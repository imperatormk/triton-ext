//===- FunctionWriter.cpp - AIR function-block writer -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits the FUNCTION_BLOCK section of the AIR bitcode for one Function:
// instruction stream, block layout, and the per-function constant table.
// Pointer instructions are emitted with their reconstructed pointee types
// from the enclosing PointeeTypeMap, not the opaque LLVM pointer type.
//
//===----------------------------------------------------------------------===//

#include "BitcodeEncoding.h"
#include "MetadataWriter.h"
#include "MetalConstraints.h"
#include "ValueEnumerator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

namespace llvm {
namespace metal {

// Forward declaration
void emitConstantsBlock(
    BitstreamWriter &W, ValueEnumerator &E,
    ArrayRef<const Constant *> Constants, unsigned CodeSize,
    const DenseMap<const Constant *, unsigned> *PoisonPtrTypeIdx = nullptr);

void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                       ValueEnumerator &E, const MetadataEnumerator &MD) {
  W.EnterSubblock(bitc::FUNCTION_BLOCK_ID, 5);

  // Reverse-post-order so every def precedes its uses: the relative value-ID
  // encoding can't represent a forward reference (underflows and corrupts the
  // bitstream). Unreachable blocks are appended so they're still emitted.
  SmallVector<const BasicBlock *, 8> BBOrder;
  {
    SmallPtrSet<const BasicBlock *, 8> Seen;
    if (!F.isDeclaration()) {
      for (const BasicBlock *BB :
           ReversePostOrderTraversal<const Function *>(&F))
        if (Seen.insert(BB).second)
          BBOrder.push_back(BB);
      for (const BasicBlock &BB : F)
        if (Seen.insert(&BB).second)
          BBOrder.push_back(&BB);
    }
  }

  // Build local value ID map
  DenseMap<const Value *, unsigned> LocalMap;
  unsigned NextID = E.globalValues.size() + E.moduleConstants.size();

  for (auto &Arg : F.args())
    LocalMap[&Arg] = NextID++;

  // Collect function-level constants, even ones that are also module constants:
  // referencing module constants directly from function instructions causes GPU
  // JIT materializeAll failures.
  SmallVector<const Constant *, 32> FuncConsts;
  auto CollectConst = [&](const Value *Op) {
    if (auto *C = dyn_cast<Constant>(Op))
      if (!isa<GlobalValue>(C) && !LocalMap.count(C) &&
          !E.globalValueMap.count(C)) {
        LocalMap[C] = NextID++;
        FuncConsts.push_back(C);
      }
  };
  // A poison/undef pointer collapses to the AS0 default; pin it to the callee
  // param's type index so its SETTYPE matches the operand the reader checks.
  DenseMap<const Constant *, unsigned> PoisonPtrTypeIdx;
  for (const BasicBlock *BB : BBOrder)
    for (auto &I : *BB) {
      for (auto &Op : I.operands())
        CollectConst(Op);
      // The shuffle mask isn't an in-memory operand but the INST_SHUFFLEVEC
      // record references it like one.
      if (auto *SV = dyn_cast<ShuffleVectorInst>(&I))
        CollectConst(SV->getShuffleMaskForBitcode());
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        auto It = E.funcTypeParamIndices.find(CI->getFunctionType());
        if (It != E.funcTypeParamIndices.end())
          for (unsigned J = 0; J < CI->arg_size() && J < It->second.size(); J++)
            if (auto *AC = dyn_cast<Constant>(CI->getArgOperand(J)))
              if (isa<UndefValue>(AC) && AC->getType()->isPointerTy())
                PoisonPtrTypeIdx.try_emplace(AC, It->second[J]);
      }
      // Same for a poison/undef load/store address: pin it to ptr(accessed
      // type).
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (auto *PC = dyn_cast<Constant>(SI->getPointerOperand()))
          if (isa<UndefValue>(PC))
            PoisonPtrTypeIdx.try_emplace(
                PC,
                E.ptrTypeIdx(PC->getType(), SI->getValueOperand()->getType()));
      }
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (auto *PC = dyn_cast<Constant>(LI->getPointerOperand()))
          if (isa<UndefValue>(PC))
            PoisonPtrTypeIdx.try_emplace(
                PC, E.ptrTypeIdx(PC->getType(), LI->getType()));
      }
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        if (auto *PC = dyn_cast<Constant>(GEP->getPointerOperand()))
          if (isa<UndefValue>(PC))
            PoisonPtrTypeIdx.try_emplace(
                PC, E.ptrTypeIdx(PC->getType(), GEP->getSourceElementType()));
      }
    }

  // Instruction results
  for (const BasicBlock *BB : BBOrder)
    for (auto &I : *BB)
      if (!I.getType()->isVoidTy())
        LocalMap[&I] = NextID++;

  auto GetAbsID = [&](const Value *V) -> unsigned {
    auto It = LocalMap.find(V);
    if (It != LocalMap.end())
      return It->second;
    if (E.globalValueMap.count(V))
      return E.globalIdx(V);
    if (auto *C = dyn_cast<Constant>(V))
      if (E.hasModuleConst(C))
        return E.moduleConstIdx(C);
    return 0;
  };

  // Relative value IDs (current instruction ID minus referenced value ID)
  unsigned CurInstID = E.globalValues.size() + E.moduleConstants.size() +
                       F.arg_size() + FuncConsts.size();
  auto GetID = [&](const Value *V) -> unsigned {
    return CurInstID - GetAbsID(V);
  };

  // DECLAREBLOCKS
  SmallVector<uint64_t, 1> DV = {BBOrder.size()};
  W.EmitRecord(bitc::FUNC_CODE_DECLAREBLOCKS, DV);

  // Function constants
  emitConstantsBlock(W, E, FuncConsts, 5, &PoisonPtrTypeIdx);

  // BB indices follow the emitted (RPO) order so branch/phi block references
  // match the stream the reader rebuilds.
  SmallVector<const BasicBlock *, 8> BBList(BBOrder.begin(), BBOrder.end());
  auto BBIdx = [&](const BasicBlock *BB) -> unsigned {
    for (unsigned I = 0; I < BBList.size(); I++)
      if (BBList[I] == BB)
        return I;
    return 0;
  };

  // Track (InstrIdx, Inst*) for any inst with attached MD; InstrIdx is 0-based
  // across the emitted stream so the reader can index it via Record[0].
  SmallVector<std::pair<unsigned, const Instruction *>, 8> Attached;
  unsigned EmittedIdx = 0;
  for (const BasicBlock *BB : BBOrder) {
    for (auto &I : *BB) {
      SmallVector<uint64_t, 16> V;

      if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        V.push_back(GetID(BO->getOperand(0)));
        V.push_back(GetID(BO->getOperand(1)));
        V.push_back(encodeBinop(BO->getOpcode()));
        if (BO->getType()->isFPOrFPVectorTy())
          V.push_back(0);
        W.EmitRecord(bitc::FUNC_CODE_INST_BINOP, V);
      } else if (auto *CI = dyn_cast<CastInst>(&I)) {
        V.push_back(GetID(CI->getOperand(0)));
        // Pointer-producing casts use the PTM-inferred pointee (Metal v1
        // needs the correct typed pointer per value).
        if (CI->getType()->isPointerTy()) {
          V.push_back(E.ptrTypeIdxForValue(CI));
        } else {
          V.push_back(E.typeIdx(CI->getDestTy()));
        }
        V.push_back(encodeCast(CI->getOpcode()));
        W.EmitRecord(bitc::FUNC_CODE_INST_CAST, V);
      } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
        V.push_back(GetID(LI->getPointerOperand()));
        // Pointer-typed loads use the per-value pointee (avoid the
        // single-pointee-per-AS mismatch, same as PHI).
        if (LI->getType()->isPointerTy())
          V.push_back(E.ptrTypeIdxForValue(LI));
        else
          V.push_back(E.typeIdx(LI->getType()));
        V.push_back(Log2_32(LI->getAlign().value()) + 1);
        V.push_back(LI->isVolatile() ? 1 : 0);
        W.EmitRecord(bitc::FUNC_CODE_INST_LOAD, V);
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        V.push_back(GetID(SI->getPointerOperand()));
        V.push_back(GetID(SI->getValueOperand()));
        V.push_back(Log2_32(SI->getAlign().value()) + 1);
        V.push_back(SI->isVolatile() ? 1 : 0);
        W.EmitRecord(bitc::FUNC_CODE_INST_STORE, V);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        V.push_back(GEP->isInBounds() ? 1 : 0);
        // Metal GPU JIT requires GEP source type to match the pointer's
        // pointee. For AS1 pointers collapsed to float*, remap i32 GEP source
        // to float (same stride), but ONLY if all terminal users consume float.
        Type *GepSrcTy = GEP->getSourceElementType();
        if (GEP->getPointerAddressSpace() == metal::AS::Device &&
            GepSrcTy->isIntegerTy(32)) {
          bool AllTerminalFloat = true;
          SmallVector<const GetElementPtrInst *, 8> Worklist;
          Worklist.push_back(GEP);
          while (!Worklist.empty() && AllTerminalFloat) {
            auto *G = Worklist.pop_back_val();
            for (auto *U : G->users()) {
              if (auto *SubGEP = dyn_cast<GetElementPtrInst>(U)) {
                Worklist.push_back(SubGEP);
              } else if (auto *LI = dyn_cast<LoadInst>(U)) {
                if (!LI->getType()->isFloatTy()) {
                  AllTerminalFloat = false;
                  break;
                }
              } else if (auto *SI = dyn_cast<StoreInst>(U)) {
                if (!SI->getValueOperand()->getType()->isFloatTy()) {
                  AllTerminalFloat = false;
                  break;
                }
              } else {
                AllTerminalFloat = false;
                break;
              }
            }
          }
          if (AllTerminalFloat)
            GepSrcTy = Type::getFloatTy(F.getContext());
        }
        V.push_back(E.typeIdx(GepSrcTy));
        for (auto &Op : GEP->operands())
          V.push_back(GetID(Op));
        W.EmitRecord(bitc::FUNC_CODE_INST_GEP, V);
      } else if (auto *EE = dyn_cast<ExtractElementInst>(&I)) {
        V.push_back(GetID(EE->getVectorOperand()));
        V.push_back(GetID(EE->getIndexOperand()));
        W.EmitRecord(bitc::FUNC_CODE_INST_EXTRACTELT, V);
      } else if (auto *IE = dyn_cast<InsertElementInst>(&I)) {
        V.push_back(GetID(IE->getOperand(0)));
        V.push_back(GetID(IE->getOperand(1)));
        V.push_back(GetID(IE->getOperand(2)));
        W.EmitRecord(bitc::FUNC_CODE_INST_INSERTELT, V);
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(&I)) {
        V.push_back(GetID(SV->getOperand(0)));
        V.push_back(GetID(SV->getOperand(1)));
        V.push_back(GetID(SV->getShuffleMaskForBitcode()));
        W.EmitRecord(bitc::FUNC_CODE_INST_SHUFFLEVEC, V);
      } else if (auto *IV = dyn_cast<InsertValueInst>(&I)) {
        V.push_back(GetID(IV->getAggregateOperand()));
        V.push_back(GetID(IV->getInsertedValueOperand()));
        for (unsigned Idx : IV->getIndices())
          V.push_back(Idx);
        W.EmitRecord(bitc::FUNC_CODE_INST_INSERTVAL, V);
      } else if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
        V.push_back(GetID(EV->getAggregateOperand()));
        for (unsigned Idx : EV->getIndices())
          V.push_back(Idx);
        W.EmitRecord(bitc::FUNC_CODE_INST_EXTRACTVAL, V);
      } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
        // Vector-condition selects MUST use VSELECT (code 29), scalar-condition
        // SELECT (code 5); emitting SELECT for a vector cond trips "Invalid
        // record". Operands [true, false, cond] for both. LowerVectorSelect
        // normally removes vector selects before reaching here.
        V.push_back(GetID(Sel->getTrueValue()));
        V.push_back(GetID(Sel->getFalseValue()));
        V.push_back(GetID(Sel->getCondition()));
        if (Sel->getCondition()->getType()->isVectorTy())
          W.EmitRecord(bitc::FUNC_CODE_INST_VSELECT, V);
        else
          W.EmitRecord(bitc::FUNC_CODE_INST_SELECT, V);
      } else if (auto *Cmp = dyn_cast<CmpInst>(&I)) {
        V.push_back(GetID(Cmp->getOperand(0)));
        V.push_back(GetID(Cmp->getOperand(1)));
        V.push_back(Cmp->getPredicate());
        W.EmitRecord(bitc::FUNC_CODE_INST_CMP2, V);
      } else if (auto *PN = dyn_cast<PHINode>(&I)) {
        // Pointer PHIs use the per-value pointee from PTM: generic typeIdx()
        // gives one pointee per address space, wrong when incoming values
        // differ (e.g. half* vs float*).
        if (PN->getType()->isPointerTy())
          V.push_back(E.ptrTypeIdxForValue(PN));
        else
          V.push_back(E.typeIdx(PN->getType()));
        for (unsigned J = 0; J < PN->getNumIncomingValues(); J++) {
          // PHI uses signed relative IDs (back-edge values give a negative
          // relative ID = forward reference).
          int64_t RelID =
              (int64_t)CurInstID - (int64_t)GetAbsID(PN->getIncomingValue(J));
          // Signed VBR: positive n → 2n, negative n → (-2n)+1
          uint64_t Encoded = (RelID >= 0) ? ((uint64_t)RelID << 1)
                                          : ((uint64_t)(-RelID) << 1) | 1;
          V.push_back(Encoded);
          V.push_back(BBIdx(PN->getIncomingBlock(J)));
        }
        W.EmitRecord(bitc::FUNC_CODE_INST_PHI, V);
      } else if (auto *UBI = dyn_cast<UncondBrInst>(&I)) {
        V.push_back(BBIdx(UBI->getSuccessor(0)));
        W.EmitRecord(bitc::FUNC_CODE_INST_BR, V);
      } else if (auto *CBI = dyn_cast<CondBrInst>(&I)) {
        V.push_back(BBIdx(CBI->getSuccessor(0)));
        V.push_back(BBIdx(CBI->getSuccessor(1)));
        V.push_back(GetID(CBI->getCondition()));
        W.EmitRecord(bitc::FUNC_CODE_INST_BR, V);
      } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
        if (RI->getReturnValue())
          V.push_back(GetID(RI->getReturnValue()));
        W.EmitRecord(bitc::FUNC_CODE_INST_RET, V);
      } else if (isa<UnreachableInst>(&I)) {
        W.EmitRecord(bitc::FUNC_CODE_INST_UNREACHABLE, V);
      } else if (auto *CI = dyn_cast<CallInst>(&I)) {
        // CALL: [paramattr, cc_flags, fnty, fnid, ...args, [sentinel]]
        // MMA load calls need paramattr=1 (nocapture+readonly on ptr param)
        bool IsMMALoadCall = false;
        if (auto *Callee = CI->getCalledFunction())
          IsMMALoadCall =
              Callee->getName().starts_with("air.simdgroup_matrix_8x8_load");
        V.push_back(IsMMALoadCall ? 1 : 0);

        // Detect MMA intrinsics that need operand bundle encoding
        bool IsMMAWithBundles = false;
        if (auto *Callee = CI->getCalledFunction()) {
          StringRef Name = Callee->getName();
          IsMMAWithBundles =
              Name.starts_with("air.simdgroup_matrix_8x8_load") ||
              Name.starts_with("air.simdgroup_matrix_8x8_multiply_accumulate");
        }

        uint64_t Flags = 0;
        if (CI->isTailCall())
          Flags |= 1;
        if (CI->getCallingConv() != CallingConv::C)
          Flags |= (uint64_t)CI->getCallingConv() << 1;
        Flags |= (1 << 15); // explicit function type
        if (IsMMAWithBundles)
          Flags |= (1 << 17); // operand bundles
        V.push_back(Flags);

        if (IsMMAWithBundles) {
          // MMA bundle encoding: sentinel 254, then function type, then callee
          V.push_back(254);
          V.push_back(E.typeIdx(CI->getFunctionType()));
        } else {
          V.push_back(E.typeIdx(CI->getFunctionType()));
        }

        V.push_back(GetID(CI->getCalledOperand()));
        for (unsigned J = 0; J < CI->arg_size(); J++)
          V.push_back(GetID(CI->getArgOperand(J)));
        W.EmitRecord(bitc::FUNC_CODE_INST_CALL, V);
      } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        // Event storage allocas (alloca ptr addrspace(3)) need allocated type
        // event_t*3, not the default float*3.
        Type *AllocTy = AI->getAllocatedType();
        if (AllocTy->isPointerTy() && AllocTy->getPointerAddressSpace() == 3) {
          if (auto *EvTy =
                  StructType::getTypeByName(AI->getContext(), "event_t"))
            V.push_back(
                E.ptrTypeIdx(PointerType::get(AI->getContext(), 3), EvTy));
          else
            V.push_back(E.typeIdx(AllocTy));
        } else {
          V.push_back(E.typeIdx(AllocTy));
        }
        V.push_back(E.typeIdx(AI->getArraySize()->getType()));
        V.push_back(GetAbsID(AI->getArraySize()));
        V.push_back((1 << 6) | (Log2_32(AI->getAlign().value()) + 1));
        W.EmitRecord(bitc::FUNC_CODE_INST_ALLOCA, V);
      } else {
        // Skipping emission while CurInstID still advances desyncs every later
        // relative operand ID (reader-side "Invalid record" far from the
        // cause). Fail loud.
        report_fatal_error(Twine("AIRWriter: unhandled instruction '") +
                           I.getOpcodeName() + "' in function '" + F.getName() +
                           "'");
      }

      if (I.hasMetadataOtherThanDebugLoc())
        Attached.push_back({EmittedIdx, &I});
      EmittedIdx++;

      if (!I.getType()->isVoidTy())
        CurInstID++;
    }
  }

  // Per-instruction metadata attachments (alias.scope, noalias, tbaa).
  if (!Attached.empty()) {
    W.EnterSubblock(bitc::METADATA_ATTACHMENT_ID, 3);
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    SmallVector<uint64_t, 16> Record;
    for (auto &P : Attached) {
      MDs.clear();
      P.second->getAllMetadataOtherThanDebugLoc(MDs);
      Record.clear();
      Record.push_back(P.first);
      for (auto &Att : MDs) {
        Record.push_back(Att.first);
        Record.push_back(MD.globalMDID(Att.second));
      }
      W.EmitRecord(bitc::METADATA_ATTACHMENT, Record);
    }
    W.ExitBlock();
  }

  W.ExitBlock();
}

} // namespace metal
} // namespace llvm
