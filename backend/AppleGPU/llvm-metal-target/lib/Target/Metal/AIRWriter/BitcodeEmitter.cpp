//===- BitcodeEmitter.cpp - Metal v1 bitcode emitter ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level orchestrator for Metal v1 bitcode emission. Delegates to
// ValueEnumerator, TypeTableWriter, ConstantsWriter, MetadataWriter,
// and FunctionWriter.
//
//===----------------------------------------------------------------------===//

#include "BitcodeEmitter.h"
#include "BitcodeEncoding.h"
#include "LowerVectorSelect.h"
#include "MetadataWriter.h"
#include "MetalConstraints.h"
#include "MetalVersion.h"
#include "ValueEnumerator.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <functional>

using namespace llvm;

namespace llvm {
namespace metal {

// The pointee the Metal reader will attribute to a pointer value.
static Type *effectivePointee(Value *Base, const PointeeTypeMap &PTM) {
  if (auto *G = dyn_cast<GetElementPtrInst>(Base))
    return G->getResultElementType();
  return PTM.get(Base);
}

// Insert an identity bitcast carrying NewPointee in the PTM. See header above.
static BitCastInst *retypePointerVia(Value *Ptr, Type *NewPointee,
                                     Instruction *BeforeI,
                                     PointeeTypeMap &PTM) {
  auto *BC = cast<BitCastInst>(CastInst::Create(
      Instruction::BitCast, Ptr, Ptr->getType(), "", BeforeI->getIterator()));
  PTM.set(BC, NewPointee);
  return BC;
}

// Collect matching instructions into a worklist. The collect-then-rewrite split
// is mandatory: the scalarize/lower passes erase the original while iterating,
// so collecting first keeps the body iterators valid.
template <typename Inst, typename PredT>
static SmallVector<Inst *, 8> collectInsts(Module &M, PredT Pred) {
  SmallVector<Inst *, 8> Out;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Cast = dyn_cast<Inst>(&I))
          if (Pred(Cast))
            Out.push_back(Cast);
  return Out;
}

// Materialize all ConstantExpr operands as instructions: Metal's GPU JIT can't
// handle constant-expression bitcode records. Byte-stride GEPs on threadgroup
// float globals are converted to float-element GEPs, since Metal v1
// typed-pointer bitcode requires the GEP source type to match the pointee.
void lowerConstantExprs(Module &M) {
  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  unsigned FloatSize = M.getDataLayout().getTypeAllocSize(FloatTy);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<std::pair<Instruction *, unsigned>, 32> Worklist;
    bool Changed = true;
    while (Changed) {
      Changed = false;
      Worklist.clear();
      for (auto &BB : F)
        for (auto &I : BB)
          for (unsigned J = 0; J < I.getNumOperands(); J++)
            if (isa<ConstantExpr>(I.getOperand(J)))
              Worklist.push_back({&I, J});
      for (auto &[I, OpIdx] : Worklist) {
        auto *CE = cast<ConstantExpr>(I->getOperand(OpIdx));
        Instruction *NewI = CE->getAsInstruction();

        // Convert byte-stride GEPs on TG float globals to float-element GEPs.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(NewI)) {
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == metal::AS::Threadgroup &&
              GEP->getNumIndices() == 1) {
            // Check if base is a TG global with float array element type
            Value *Base = GEP->getPointerOperand();
            GlobalVariable *GV = dyn_cast<GlobalVariable>(Base);
            if (GV) {
              Type *ElemTy = nullptr;
              if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                ElemTy = AT->getElementType();
              if (ElemTy && ElemTy->isFloatTy()) {
                Value *ByteIdx = GEP->idx_begin()->get();
                if (auto *CI = dyn_cast<ConstantInt>(ByteIdx)) {
                  uint64_t ByteOff = CI->getZExtValue();
                  if (ByteOff % FloatSize == 0) {
                    // Reuse an existing entry-block `gep [N x float], @GV, 0,
                    // 0`.
                    Value *FloatBase = nullptr;
                    for (auto *U : GV->users()) {
                      auto *BaseGEP = dyn_cast<GetElementPtrInst>(U);
                      if (!BaseGEP || !BaseGEP->getParent())
                        continue;
                      if (BaseGEP->getFunction() == &F &&
                          BaseGEP->getSourceElementType() ==
                              GV->getValueType() &&
                          BaseGEP->getNumIndices() == 2) {
                        FloatBase = BaseGEP;
                        break;
                      }
                    }
                    if (!FloatBase) {
                      auto *NewBaseGEP = GetElementPtrInst::CreateInBounds(
                          GV->getValueType(), GV,
                          {ConstantInt::get(I64Ty, 0),
                           ConstantInt::get(I64Ty, 0)});
                      NewBaseGEP->insertBefore(
                          F.getEntryBlock().getFirstInsertionPt());
                      FloatBase = NewBaseGEP;
                    }
                    auto *FloatGEP = GetElementPtrInst::CreateInBounds(
                        ElemTy, FloatBase,
                        {ConstantInt::get(I64Ty, ByteOff / FloatSize)});
                    FloatGEP->insertBefore(I->getIterator());
                    I->setOperand(OpIdx, FloatGEP);
                    NewI->deleteValue(); // discard the byte GEP
                    Changed = true;
                    continue;
                  }
                }
              }
            }
          }
        }

        NewI->insertBefore(I->getIterator());
        I->setOperand(OpIdx, NewI);
        Changed = true;
      }
    }
  }
}

// A pointer phi's record carries one pointee type; wrap incomings that resolve
// to a different type in an identity bitcast so the reader accepts the record.
static void fixPhiIncomingTypes(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break; // phis are at block start
        if (!PN->getType()->isPointerTy())
          continue;
        // Only intervene when an incoming carries a CONCRETE pointee. If every
        // incoming defaults to the per-AS fallback, leaving it untouched keeps
        // phi/incomings/consumers all on the same default - intervening would
        // create a spurious mismatch.
        Type *PhiPointee = PTM.get(PN);
        if (!PhiPointee) {
          for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
            Value *In = PN->getIncomingValue(J);
            if (isa<Constant>(In))
              continue;
            if (auto *GV = dyn_cast<GlobalVariable>(In))
              PhiPointee = GV->getValueType();
            else if (auto *G = dyn_cast<GetElementPtrInst>(In))
              PhiPointee = G->getResultElementType();
            else if (Type *T = PTM.get(In))
              PhiPointee = T;
            else
              continue;
            break;
          }
        }
        if (!PhiPointee)
          continue; // no concrete pointee anywhere - leave the phi alone
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          Value *In = PN->getIncomingValue(J);
          Type *InPointee = nullptr;
          if (isa<ConstantPointerNull>(In)) {
            // A typed null's SETTYPE default pointee can disagree with the phi
            // pointee → invalid record. Replace with inttoptr(0) pinned to the
            // phi pointee so the incoming is a real typed value.
            auto &Ctx = M.getContext();
            auto *Zero = ConstantInt::get(Type::getInt64Ty(Ctx), 0);
            auto *I2P = new IntToPtrInst(
                Zero, In->getType(), "",
                PN->getIncomingBlock(J)->getTerminator()->getIterator());
            PTM.set(I2P, PhiPointee);
            PN->setIncomingValue(J, I2P);
            continue;
          }
          if (isa<Constant>(In))
            continue; // other constants: leave untouched
          if (auto *GV = dyn_cast<GlobalVariable>(In))
            InPointee = GV->getValueType();
          else if (auto *G = dyn_cast<GetElementPtrInst>(In))
            InPointee = G->getResultElementType();
          else
            InPointee = PTM.get(In);
          if (InPointee == PhiPointee)
            continue;
          PN->setIncomingValue(
              J,
              retypePointerVia(In, PhiPointee,
                               PN->getIncomingBlock(J)->getTerminator(), PTM));
        }
        // Pin the phi's own pointee so the record's type index matches the
        // (now-consistent) incomings.
        PTM.set(PN, PhiPointee);
      }
  }
}

// Normalize GEP source element types to match the base pointer's typed pointee
// (the Metal v1 reader rejects mismatches). Phase ordering is load-bearing:
// phase 1 (vector linearization) runs module-wide first so phases 2/3 see the
// fresh half/i8 GEPs; phase 2 (MMA retype) is one-shot; phase 3 (byte-stride)
// iterates to a fixpoint. Shape 4 (array globals) is normalizeArrayGlobalGEPs.
static void normalizeGEPs(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);
  const DataLayout &DL = M.getDataLayout();

  bool HasMMA = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      HasMMA = true;

  // Phase 1: shape-1 vector-pointee linearization, module-wide before the
  // type-mismatch arms (see ordering note above).
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (isa<FixedVectorType>(GEP->getSourceElementType()) &&
              GEP->getNumIndices() == 2)
            ToFix.push_back(GEP);
    for (auto *GEP : ToFix) {
      auto *VT = cast<FixedVectorType>(GEP->getSourceElementType());
      IRBuilder<> B(GEP);
      Value *I0 = B.CreateSExtOrTrunc(GEP->getOperand(1), B.getInt64Ty());
      Value *I1 = B.CreateSExtOrTrunc(GEP->getOperand(2), B.getInt64Ty());
      Value *Lin =
          B.CreateAdd(B.CreateMul(I0, B.getInt64(VT->getNumElements())), I1);
      auto *NewGEP = cast<GetElementPtrInst>(
          B.CreateGEP(VT->getElementType(), GEP->getPointerOperand(), Lin));
      NewGEP->setIsInBounds(GEP->isInBounds());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
  }

  // Shape-2 (MMA element-mismatch) predicate. MMA-modules only.
  auto isMMAMismatch = [&](GetElementPtrInst *GEP) -> bool {
    if (!HasMMA)
      return false;
    Type *SrcTy = GEP->getSourceElementType();
    if (SrcTy == FloatTy || GEP->getNumIndices() != 1)
      return false;
    if (!SrcTy->isIntegerTy() && !SrcTy->isHalfTy() && !SrcTy->isBFloatTy())
      return false;
    unsigned AS = GEP->getPointerAddressSpace();
    if (AS != metal::AS::Device && AS != metal::AS::Threadgroup)
      return false;
    // Don't float-ify a genuine i32 buffer (e.g. int8-dot output): its pointee
    // stays i32 → "gep type does not match pointee" → materializeAll failure.
    if (Type *Pointee = PTM.get(GEP->getPointerOperand()))
      if (Pointee->isIntegerTy() && Pointee == SrcTy)
        return false;
    return true;
  };

  // Shape-3 (byte-stride GEP on non-global bases) predicate.
  auto isByteArray = [&](GetElementPtrInst *GEP) -> bool {
    if (GEP->getNumIndices() != 1)
      return false;
    Type *SrcTy = GEP->getSourceElementType();
    auto *AT = dyn_cast<ArrayType>(SrcTy);
    bool IsByteArray = AT && AT->getElementType()->isIntegerTy(8);
    // Two byte-stride forms: `[N x i8]` (element-count index) and `i8` (byte).
    if (!IsByteArray && !SrcTy->isIntegerTy(8))
      return false;
    // Array globals are shape 4's (normalizeArrayGlobalGEPs) job.
    if (isa<GlobalVariable>(GEP->getPointerOperand()))
      return false;
    Type *Pointee = effectivePointee(GEP->getPointerOperand(), PTM);
    return Pointee && Pointee != SrcTy;
  };

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    // Phase 2: shape 2 (MMA element-mismatch) - one-shot, NO fixpoint (a
    // half/i8 base-bitcast leaves the source unchanged; re-running would emit a
    // redundant second bitcast). Runs before phase 3 to claim overlapping GEPs.
    {
      SmallVector<GetElementPtrInst *, 8> ToFix;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            if (isMMAMismatch(GEP))
              ToFix.push_back(GEP);
      for (auto *GEP : ToFix) {
        Type *SrcTy = GEP->getSourceElementType();
        Value *Ptr = GEP->getPointerOperand();
        // Same-size (i32 vs float): retype source to float, stride unchanged.
        if (SrcTy->getPrimitiveSizeInBits() == 32) {
          GEP->setSourceElementType(FloatTy);
          GEP->setResultElementType(FloatTy);
          continue;
        }
        // Different-size (half/i8): stride differs, so give it a base typed to
        // the source rather than retyping the GEP.
        GEP->setOperand(0, retypePointerVia(Ptr, SrcTy, GEP, PTM));
      }
    }

    // Phase 3: shape 3 (byte-stride) - fixpoint, since retyping a GEP changes
    // the pointee its dependent GEPs see.
    bool Changed = true;
    while (Changed) {
      Changed = false;
      SmallVector<GetElementPtrInst *, 8> ToFix;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            if (isByteArray(GEP))
              ToFix.push_back(GEP);
      for (auto *GEP : ToFix) {
        Type *SrcTy = GEP->getSourceElementType();
        Type *Pointee = effectivePointee(GEP->getPointerOperand(), PTM);
        bool PointeeOK = Pointee->isSized() && !Pointee->isAggregateType() &&
                         DL.getTypeAllocSize(Pointee) > 0;
        if (auto *AT = dyn_cast<ArrayType>(SrcTy)) {
          // [N x i8]: same stride as the pointee iff alloc sizes match.
          if (PointeeOK &&
              DL.getTypeAllocSize(Pointee) == AT->getNumElements()) {
            GEP->setSourceElementType(Pointee);
            GEP->setResultElementType(Pointee);
            Changed = true;
            continue;
          }
        } else {
          // i8: a constant byte offset divisible by the pointee size can be
          // rescaled into an element-typed GEP.
          auto *CI = dyn_cast<ConstantInt>(GEP->idx_begin()->get());
          uint64_t ESz = PointeeOK ? DL.getTypeAllocSize(Pointee) : 0;
          if (CI && ESz && CI->getSExtValue() % (int64_t)ESz == 0) {
            GEP->setSourceElementType(Pointee);
            GEP->setResultElementType(Pointee);
            GEP->setOperand(1,
                            ConstantInt::get(CI->getType(), CI->getSExtValue() /
                                                                (int64_t)ESz));
            Changed = true;
            continue;
          }
        }
        // Fallback: give the GEP a base typed to its own source type.
        GEP->setOperand(
            0, retypePointerVia(GEP->getPointerOperand(), SrcTy, GEP, PTM));
        Changed = true;
      }
    }
  }
}

// The Metal AIR backend has no lowering for llvm.scmp/llvm.ucmp ("Undefined
// symbols: llvm.scmp.*"). Expand inline: scmp(a,b) = zext(a>b) - zext(a<b),
// signed/unsigned predicates per intrinsic, extended to the result type.
static void lowerCmpIntrinsics(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 8> Calls;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Intrinsic::ID ID = CI->getIntrinsicID();
          if (ID == Intrinsic::scmp || ID == Intrinsic::ucmp)
            Calls.push_back(CI);
        }
    for (CallInst *CI : Calls) {
      bool Signed = CI->getIntrinsicID() == Intrinsic::scmp;
      IRBuilder<> B(CI);
      Value *A = CI->getArgOperand(0);
      Value *Bv = CI->getArgOperand(1);
      Value *Gt = Signed ? B.CreateICmpSGT(A, Bv) : B.CreateICmpUGT(A, Bv);
      Value *Lt = Signed ? B.CreateICmpSLT(A, Bv) : B.CreateICmpULT(A, Bv);
      Type *RetTy = CI->getType();
      Value *Res =
          B.CreateSub(B.CreateZExt(Gt, RetTy), B.CreateZExt(Lt, RetTy));
      CI->replaceAllUsesWith(Res);
      CI->eraseFromParent();
    }
  }
  // Drop the now-unused intrinsic declarations so no symbol is referenced.
  for (auto It = M.begin(); It != M.end();) {
    Function &F = *It++;
    Intrinsic::ID ID = F.getIntrinsicID();
    if ((ID == Intrinsic::scmp || ID == Intrinsic::ucmp) && F.use_empty())
      F.eraseFromParent();
  }
}

// A `<N x ptr>` has only one pointee slot in the AIR type table but its scalar
// operands can carry conflicting pointees; lower the whole web to `<N x i64>`.
static void lowerVectorPointerToInt(Module &M) {
  auto isPtrVec = [](Type *T) -> FixedVectorType * {
    auto *VT = dyn_cast<FixedVectorType>(T);
    if (VT && VT->getElementType()->isPointerTy())
      return VT;
    return nullptr;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // Collect every instruction that produces a vector-of-pointers.
    SmallVector<Instruction *, 16> PtrVecDefs;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isPtrVec(I.getType()))
          PtrVecDefs.push_back(&I);
    if (PtrVecDefs.empty())
      continue;

    // Map each pointer-vector value to its integer-vector replacement.
    DenseMap<Value *, Value *> IntOf;
    auto intVecTy = [&](FixedVectorType *PVT) {
      return FixedVectorType::get(I64, PVT->getNumElements());
    };
    // Materialize the integer-vector form of an arbitrary pointer-vector
    // operand (constants/poison/undef and not-yet-rewritten defs).
    std::function<Value *(Value *, IRBuilder<> &)> asIntVec =
        [&](Value *V, IRBuilder<> &B) -> Value * {
      if (auto *It = IntOf.lookup(V))
        return It;
      auto *PVT = cast<FixedVectorType>(V->getType());
      if (isa<UndefValue>(V))
        return UndefValue::get(intVecTy(PVT));
      if (isa<ConstantAggregateZero>(V) ||
          (isa<Constant>(V) && cast<Constant>(V)->isNullValue()))
        return ConstantAggregateZero::get(intVecTy(PVT));
      // Fallback for any other constant/value: ptrtoint the whole vector.
      return B.CreatePtrToInt(V, intVecTy(PVT));
    };

    // First create placeholder integer phis so cycles resolve.
    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        IRBuilder<> B(PN);
        auto *NewPN =
            B.CreatePHI(intVecTy(cast<FixedVectorType>(PN->getType())),
                        PN->getNumIncomingValues());
        IntOf[PN] = NewPN;
      }

    // Rewrite the non-phi defs in program order.
    for (Instruction *I : PtrVecDefs) {
      if (isa<PHINode>(I))
        continue;
      IRBuilder<> B(I);
      Value *Repl = nullptr;
      if (auto *IE = dyn_cast<InsertElementInst>(I)) {
        Value *Vec = asIntVec(IE->getOperand(0), B);
        Value *Sc = B.CreatePtrToInt(IE->getOperand(1), I64);
        Repl = B.CreateInsertElement(Vec, Sc, IE->getOperand(2));
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(I)) {
        Value *A = asIntVec(SV->getOperand(0), B);
        Value *Bv = asIntVec(SV->getOperand(1), B);
        Repl = B.CreateShuffleVector(A, Bv, SV->getShuffleMask());
      } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
        Value *T = asIntVec(Sel->getTrueValue(), B);
        Value *Fv = asIntVec(Sel->getFalseValue(), B);
        Repl = B.CreateSelect(Sel->getCondition(), T, Fv);
      } else if (auto *BC = dyn_cast<BitCastInst>(I)) {
        // ptr-vec bitcast (e.g. addrspace-preserving): forward the int form.
        Repl = asIntVec(BC->getOperand(0), B);
      } else {
        // Unhandled producer: ptrtoint then back so users still see a ptr-vec.
        continue;
      }
      IntOf[I] = Repl;
    }

    // Fill phi incomings now that all defs have int forms.
    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        auto *NewPN = cast<PHINode>(IntOf[PN]);
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          IRBuilder<> B(PN->getIncomingBlock(J)->getTerminator());
          NewPN->addIncoming(asIntVec(PN->getIncomingValue(J), B),
                             PN->getIncomingBlock(J));
        }
      }

    // Redirect users to the int form: ptrtoint -> int (width-fixed),
    // extractelement -> scalar int back to ptr, else inttoptr-rebuilt vector.
    for (Instruction *I : PtrVecDefs) {
      Value *Int = IntOf.lookup(I);
      if (!Int)
        continue;
      SmallVector<Use *, 8> Uses;
      for (Use &U : I->uses())
        Uses.push_back(&U);
      for (Use *U : Uses) {
        auto *User = cast<Instruction>(U->getUser());
        if (IntOf.count(User))
          continue; // already rewritten to consume the int form
        IRBuilder<> B(User);
        if (auto *P2I = dyn_cast<PtrToIntInst>(User)) {
          Value *V = Int;
          if (P2I->getType() != Int->getType())
            V = B.CreateZExtOrTrunc(Int, P2I->getType());
          P2I->replaceAllUsesWith(V);
          continue; // P2I now dead; cleaned up below
        }
        if (auto *EE = dyn_cast<ExtractElementInst>(User)) {
          Value *Sc = B.CreateExtractElement(Int, EE->getIndexOperand());
          EE->replaceAllUsesWith(B.CreateIntToPtr(Sc, EE->getType()));
          continue;
        }
        // Generic consumer still expecting a pointer vector: rebuild one.
        U->set(B.CreateIntToPtr(Int, I->getType()));
      }
    }

    // Erase the now-dead pointer-vector defs and orphaned ptrtoints.
    for (Instruction *I : reverse(PtrVecDefs)) {
      if (!IntOf.count(I))
        continue;
      if (!I->use_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
    }
    // Drop ptrtoint/extractelement consumers that were replaced.
    SmallVector<Instruction *, 8> Dead;
    for (auto &BB : F)
      for (auto &I : BB)
        if ((isa<PtrToIntInst>(I) || isa<ExtractElementInst>(I)) &&
            I.use_empty() && isPtrVec(I.getOperand(0)->getType()) &&
            IntOf.count(I.getOperand(0)))
          Dead.push_back(&I);
    for (Instruction *I : Dead)
      I->eraseFromParent();
    for (Instruction *I : reverse(PtrVecDefs))
      if (IntOf.count(I) && I->use_empty())
        I->eraseFromParent();
  }
}

// The vector-condition SELECT (VSELECT) is rejected by the AGX JIT; scalarize
// each into a per-lane extract/select/insert chain.
[[maybe_unused]] static void scalarizeVectorSelects(Module &M) {
  auto Sels = collectInsts<SelectInst>(M, [](SelectInst *Sel) {
    return Sel->getCondition()->getType()->isVectorTy();
  });
  for (auto *Sel : Sels) {
    IRBuilder<> B(Sel);
    auto *VT = cast<FixedVectorType>(Sel->getType());
    Value *Res = UndefValue::get(VT);
    for (unsigned L = 0; L < VT->getNumElements(); ++L) {
      Value *C = B.CreateExtractElement(Sel->getCondition(), B.getInt64(L));
      Value *T = B.CreateExtractElement(Sel->getTrueValue(), B.getInt64(L));
      Value *F = B.CreateExtractElement(Sel->getFalseValue(), B.getInt64(L));
      Res = B.CreateInsertElement(Res, B.CreateSelect(C, T, F), B.getInt64(L));
    }
    Sel->replaceAllUsesWith(Res);
    Sel->eraseFromParent();
  }
}

// The mid-end emits >64-bit integer arithmetic for overflow-free closed
// forms (e.g. SCEV's `trunc((zext(a) * zext(b)) >> 1)` triangular sums as
// i65). The AGX JIT cannot legalize any iN > 64; expand such chains into
// (lo, hi) i64 limb pairs. Unsupported wide ops fail loud.
static void expandWideIntegers(Module &M) {
  auto isWide = [](Type *T) {
    return T->isIntegerTy() && T->getIntegerBitWidth() > 64 &&
           T->getIntegerBitWidth() <= 128;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    DenseMap<Value *, std::pair<Value *, Value *>> Limbs; // wide -> (lo, hi)
    SmallVector<Instruction *, 8> Wide;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isWide(I.getType()) || llvm::any_of(I.operands(), [&](Value *Op) {
              return isWide(Op->getType());
            }))
          Wide.push_back(&I);
    if (Wide.empty())
      continue;
    auto umulh = [&](IRBuilder<> &B, Value *A, Value *Bv) -> Value * {
      Value *Mask = ConstantInt::get(I64, 0xffffffffull);
      Value *AL = B.CreateAnd(A, Mask), *AH = B.CreateLShr(A, 32);
      Value *BL = B.CreateAnd(Bv, Mask), *BH = B.CreateLShr(Bv, 32);
      Value *LL = B.CreateMul(AL, BL);
      Value *LH = B.CreateMul(AL, BH);
      Value *HL = B.CreateMul(AH, BL);
      Value *HH = B.CreateMul(AH, BH);
      Value *Mid =
          B.CreateAdd(B.CreateAdd(B.CreateLShr(LL, 32), B.CreateAnd(LH, Mask)),
                      B.CreateAnd(HL, Mask));
      return B.CreateAdd(B.CreateAdd(HH, B.CreateAdd(B.CreateLShr(LH, 32),
                                                     B.CreateLShr(HL, 32))),
                         B.CreateLShr(Mid, 32));
    };
    for (Instruction *I : Wide) {
      IRBuilder<> B(I);
      auto limbsOf = [&](Value *V) -> std::pair<Value *, Value *> {
        auto It = Limbs.find(V);
        if (It != Limbs.end())
          return It->second;
        if (auto *C = dyn_cast<ConstantInt>(V)) {
          APInt A = C->getValue();
          return {ConstantInt::get(I64, A.trunc(64)),
                  ConstantInt::get(I64, A.lshr(64).trunc(64))};
        }
        report_fatal_error("AIRWriter: unmapped wide integer operand");
      };
      if (auto *ZE = dyn_cast<ZExtInst>(I)) {
        Limbs[I] = {B.CreateZExtOrTrunc(ZE->getOperand(0), I64),
                    ConstantInt::get(I64, 0)};
      } else if (auto *SE2 = dyn_cast<SExtInst>(I)) {
        Value *Lo = B.CreateSExtOrTrunc(SE2->getOperand(0), I64);
        Limbs[I] = {Lo, B.CreateAShr(Lo, 63)};
      } else if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        auto [L1, H1] = limbsOf(BO->getOperand(0));
        switch (BO->getOpcode()) {
        case Instruction::Mul: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateMul(L1, L2);
          Value *Hi =
              B.CreateAdd(umulh(B, L1, L2), B.CreateAdd(B.CreateMul(L1, H2),
                                                        B.CreateMul(H1, L2)));
          Limbs[I] = {Lo, Hi};
          break;
        }
        case Instruction::Add: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateAdd(L1, L2);
          Value *Carry = B.CreateZExt(B.CreateICmpULT(Lo, L1), I64);
          Limbs[I] = {Lo, B.CreateAdd(B.CreateAdd(H1, H2), Carry)};
          break;
        }
        case Instruction::Sub: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateSub(L1, L2);
          Value *Borrow = B.CreateZExt(B.CreateICmpULT(L1, L2), I64);
          Limbs[I] = {Lo, B.CreateSub(B.CreateSub(H1, H2), Borrow)};
          break;
        }
        case Instruction::LShr: {
          auto *CI = dyn_cast<ConstantInt>(BO->getOperand(1));
          if (!CI)
            report_fatal_error("AIRWriter: wide lshr by non-constant");
          uint64_t Sh = CI->getZExtValue();
          if (Sh == 0) {
            Limbs[I] = {L1, H1};
          } else if (Sh < 64) {
            Limbs[I] = {
                B.CreateOr(B.CreateLShr(L1, Sh), B.CreateShl(H1, 64 - Sh)),
                B.CreateLShr(H1, Sh)};
          } else {
            Limbs[I] = {B.CreateLShr(H1, Sh - 64), ConstantInt::get(I64, 0)};
          }
          break;
        }
        default:
          report_fatal_error(Twine("AIRWriter: unhandled wide integer op '") +
                             BO->getOpcodeName() + "'");
        }
      } else if (auto *TR = dyn_cast<TruncInst>(I)) {
        auto [Lo, Hi] = limbsOf(TR->getOperand(0));
        (void)Hi;
        Value *R = B.CreateZExtOrTrunc(Lo, TR->getType());
        TR->replaceAllUsesWith(R);
      } else {
        report_fatal_error(Twine("AIRWriter: unhandled wide integer user '") +
                           I->getOpcodeName() + "'");
      }
    }
    for (auto It = Wide.rbegin(); It != Wide.rend(); ++It)
      (*It)->eraseFromParent();
  }
}

// The AGX JIT cannot legalize bool-vector bitcasts (`bitcast <N x i1> to iN`
// and `bitcast <N x i1> to <M x iK>`); expand both into per-bit shifts.
static bool isI1VecScalarBitcast(BitCastInst *BC) {
  auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy());
  auto *DV = dyn_cast<FixedVectorType>(BC->getDestTy());
  return (SV && SV->getElementType()->isIntegerTy(1) &&
          BC->getDestTy()->isIntegerTy(SV->getNumElements())) ||
         (DV && DV->getElementType()->isIntegerTy(1) &&
          BC->getSrcTy()->isIntegerTy(DV->getNumElements()));
}

static bool isI1VecToSubByteVecBitcast(BitCastInst *BC) {
  auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy());
  auto *DV = dyn_cast<FixedVectorType>(BC->getDestTy());
  if (!SV || !DV || !SV->getElementType()->isIntegerTy(1) ||
      !DV->getElementType()->isIntegerTy())
    return false;
  unsigned K = DV->getElementType()->getIntegerBitWidth();
  return SV->getNumElements() == DV->getNumElements() * K;
}

static void scalarizeBoolVectorCasts(Module &M) {
  auto Casts = collectInsts<BitCastInst>(M, [](BitCastInst *BC) {
    return isI1VecScalarBitcast(BC) || isI1VecToSubByteVecBitcast(BC);
  });
  for (auto *BC : Casts) {
    IRBuilder<> B(BC);
    Value *R;
    if (isI1VecToSubByteVecBitcast(BC)) {
      auto *DV = cast<FixedVectorType>(BC->getDestTy());
      Type *ElemTy = DV->getElementType();
      unsigned K = ElemTy->getIntegerBitWidth();
      R = UndefValue::get(DV);
      for (unsigned E = 0; E < DV->getNumElements(); ++E) {
        Value *Packed = ConstantInt::get(ElemTy, 0);
        for (unsigned K0 = 0; K0 < K; ++K0) {
          Value *Bit = B.CreateZExt(
              B.CreateExtractElement(BC->getOperand(0), B.getInt64(E * K + K0)),
              ElemTy);
          Packed = B.CreateOr(Packed, B.CreateShl(Bit, K0));
        }
        R = B.CreateInsertElement(R, Packed, B.getInt64(E));
      }
    } else if (auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy())) {
      R = ConstantInt::get(BC->getDestTy(), 0);
      for (unsigned L = 0; L < SV->getNumElements(); ++L) {
        Value *Bit = B.CreateZExt(
            B.CreateExtractElement(BC->getOperand(0), B.getInt64(L)),
            BC->getDestTy());
        R = B.CreateOr(R, B.CreateShl(Bit, L));
      }
    } else {
      auto *DV = cast<FixedVectorType>(BC->getDestTy());
      R = UndefValue::get(DV);
      for (unsigned L = 0; L < DV->getNumElements(); ++L) {
        Value *Bit =
            B.CreateTrunc(B.CreateLShr(BC->getOperand(0), L), B.getInt1Ty());
        R = B.CreateInsertElement(R, Bit, B.getInt64(L));
      }
    }
    BC->replaceAllUsesWith(R);
    BC->eraseFromParent();
  }
}

// `zext nneg` is definitionally equal to `sext`; emit the sext form. The
// AGX JIT widens zext-fed 64-bit multiplies into 65-bit operations it then
// fails to legalize (PSO abort "unable to legalize instruction ... s65"),
// while the sext form is its long-proven path.
static void canonicalizeNNegZExt(Module &M) {
  SmallVector<ZExtInst *, 16> Zexts;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *ZE = dyn_cast<ZExtInst>(&I))
          if (ZE->hasNonNeg())
            Zexts.push_back(ZE);
  for (auto *ZE : Zexts) {
    auto *SE = CastInst::Create(Instruction::SExt, ZE->getOperand(0),
                                ZE->getType(), "", ZE->getIterator());
    SE->takeName(ZE);
    ZE->replaceAllUsesWith(SE);
    ZE->eraseFromParent();
  }
}

// AIR v1 bitcode has no freeze opcode. Replacing freeze with its operand is
// a legal refinement (freeze only matters for poison/undef inputs, where any
// fixed value is a valid choice).
static void lowerFreezeInsts(Module &M) {
  SmallVector<FreezeInst *, 8> Frozen;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *FI = dyn_cast<FreezeInst>(&I))
          Frozen.push_back(FI);
  for (auto *FI : Frozen) {
    FI->replaceAllUsesWith(FI->getOperand(0));
    FI->eraseFromParent();
  }
}

// AIR v1 bitcode has no aggregate load: the reader cannot enumerate an
// `[N x T]`-typed LOAD record. The mid-end produces them when it widens a
// small fixed-count gather into one load. Expand into per-element
// GEP+load+insertvalue so every load is scalar/vector-typed.
static void scalarizeAggregateLoads(Module &M) {
  auto Aggs = collectInsts<LoadInst>(
      M, [](LoadInst *LI) { return isa<ArrayType>(LI->getType()); });
  for (LoadInst *LI : Aggs) {
    auto *AT = cast<ArrayType>(LI->getType());
    Type *ElemTy = AT->getElementType();
    Value *Ptr = LI->getPointerOperand();
    Value *Agg = UndefValue::get(AT);
    IRBuilder<> B(LI);
    for (uint64_t E = 0; E < AT->getNumElements(); ++E) {
      Value *EP = B.CreateGEP(ElemTy, Ptr, B.getInt64(E));
      Value *EV = B.CreateLoad(ElemTy, EP);
      Agg = B.CreateInsertValue(Agg, EV, {unsigned(E)});
    }
    LI->replaceAllUsesWith(Agg);
    LI->eraseFromParent();
  }
}

static void fixSelectPointerArms(Module &M, PointeeTypeMap &PTM) {
  auto pointeeOf = [&](Value *V) -> Type * {
    if (isa<ConstantPointerNull>(V))
      return nullptr;
    return effectivePointee(V, PTM);
  };
  auto Selects = collectInsts<SelectInst>(M, [&](SelectInst *S) {
    if (!S->getType()->isPointerTy())
      return false;
    Value *T = S->getTrueValue(), *F = S->getFalseValue();
    if (isa<ConstantPointerNull>(T) || isa<ConstantPointerNull>(F))
      return true;
    Type *Use = PointeeTypeMap::inferFromUsage(S);
    return pointeeOf(T) != pointeeOf(F) ||
           (Use && (pointeeOf(T) != Use || pointeeOf(F) != Use));
  });
  for (auto *S : Selects) {
    Value *T = S->getTrueValue(), *F = S->getFalseValue();
    Type *Pointee = PointeeTypeMap::inferFromUsage(S);
    if (!Pointee)
      Pointee = pointeeOf(T);
    if (!Pointee)
      Pointee = pointeeOf(F);
    if (!Pointee)
      continue;
    if (pointeeOf(T) != Pointee || isa<ConstantPointerNull>(T))
      S->setOperand(1, retypePointerVia(T, Pointee, S, PTM));
    if (pointeeOf(F) != Pointee || isa<ConstantPointerNull>(F))
      S->setOperand(2, retypePointerVia(F, Pointee, S, PTM));
    PTM.set(S, Pointee);
  }
}

// Make every load/store's pointer pointee equal its access type; folded
// byte-GEPs leave e.g. a float store through an i8-typed pointer → "Explicit
// load/store type does not match pointee type of pointer operand". Route such
// accesses through an identity bitcast pinned to the access type.
static void fixAccessTypeMismatch(Module &M, PointeeTypeMap &PTM) {
  auto accessTypeOf = [](Instruction *I) -> Type * {
    if (auto *LI = dyn_cast<LoadInst>(I))
      return LI->getType();
    if (auto *SI = dyn_cast<StoreInst>(I))
      return SI->getValueOperand()->getType();
    return nullptr;
  };
  auto pointerOf = [](Instruction *I) -> Value * {
    if (auto *LI = dyn_cast<LoadInst>(I))
      return LI->getPointerOperand();
    return cast<StoreInst>(I)->getPointerOperand();
  };
  auto Fix = collectInsts<Instruction>(M, [&](Instruction *I) {
    Type *AccessTy = accessTypeOf(I);
    if (!AccessTy)
      return false;
    Value *Ptr = pointerOf(I);
    if (isa<BitCastInst>(Ptr))
      return false;
    // Vector accesses always retype; scalar only when the pointee provably
    // disagrees, except inttoptr/null which share a per-type default another
    // value may claim first - always retype those.
    if (!AccessTy->isVectorTy()) {
      if (!isa<IntToPtrInst>(Ptr) && !isa<ConstantPointerNull>(Ptr)) {
        Type *Pointee = effectivePointee(Ptr, PTM);
        if (!Pointee || Pointee == AccessTy)
          return false;
      }
    }
    return true;
  });
  for (Instruction *I : Fix) {
    if (auto *LI = dyn_cast<LoadInst>(I))
      LI->setOperand(
          0, retypePointerVia(LI->getPointerOperand(), LI->getType(), LI, PTM));
    else {
      auto *SI = cast<StoreInst>(I);
      SI->setOperand(1, retypePointerVia(SI->getPointerOperand(),
                                         SI->getValueOperand()->getType(), SI,
                                         PTM));
    }
  }
}

// Give every simdgroup-matrix call a pointer whose pointee matches the
// intrinsic's element suffix; a suffix mismatch (e.g. float* into a p3f16 load)
// emits an invalid record (PSO "Failed to materializeAll").
static void fixMMAPointerSuffixMismatch(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto Calls = collectInsts<CallInst>(M, [](CallInst *CI) {
    return CI->getCalledFunction() &&
           CI->getCalledFunction()->getName().starts_with(
               "air.simdgroup_matrix_8x8_");
  });
  for (auto *CI : Calls) {
    StringRef Name = CI->getCalledFunction()->getName();
    Type *Elem = nullptr;
    if (Name.contains("f16") && !Name.contains("bf16"))
      Elem = Type::getHalfTy(Ctx);
    else if (Name.contains("bf16"))
      Elem = Type::getBFloatTy(Ctx);
    else if (Name.contains("f32"))
      Elem = Type::getFloatTy(Ctx);
    if (!Elem)
      continue;
    for (unsigned J = 0; J < CI->arg_size(); J++) {
      Value *Op = CI->getArgOperand(J);
      if (!Op->getType()->isPointerTy())
        continue;
      if (Elem->isFloatTy() && !isa<Constant>(Op)) {
        // Float-suffix operands are usually float-typed; wrap only when the
        // pointee provably disagrees (byte-GEP chains leave i8* into p1f32).
        Type *Pointee = effectivePointee(Op, PTM);
        if (!Pointee || Pointee == Elem)
          continue;
      }
      if (isa<BitCastInst>(Op) || isa<AllocaInst>(Op))
        continue;
      CI->setArgOperand(J, retypePointerVia(Op, Elem, CI, PTM));
    }
  }
}

// Fix air.arg_type_name/size in kernel metadata to match PTM pointee types
// (the pipeline may label all buffers "float"); the Metal GPU JIT validates
// these against the bitcode types.
static void fixKernelArgMetadata(Module &M, const PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto *AirKernel = M.getNamedMetadata("air.kernel");
  if (!AirKernel)
    return;

  for (unsigned K = 0; K < AirKernel->getNumOperands(); K++) {
    auto *KernelMD = AirKernel->getOperand(K);
    if (KernelMD->getNumOperands() < 3)
      continue;
    // KernelMD: {fn, attrs, argDescs}
    auto *ArgDescs = dyn_cast_or_null<MDNode>(KernelMD->getOperand(2));
    if (!ArgDescs)
      continue;

    auto *FnVAM = dyn_cast_or_null<ValueAsMetadata>(KernelMD->getOperand(0));
    if (!FnVAM)
      continue;
    auto *Fn = dyn_cast<Function>(FnVAM->getValue());
    if (!Fn)
      continue;

    for (unsigned A = 0; A < ArgDescs->getNumOperands(); A++) {
      auto *ArgMD = dyn_cast_or_null<MDNode>(ArgDescs->getOperand(A));
      if (!ArgMD || ArgMD->getNumOperands() < 2)
        continue;

      // Check if this is a buffer arg (has "air.buffer" string)
      bool IsBuffer = false;
      for (unsigned I = 1; I < ArgMD->getNumOperands(); I++)
        if (auto *S = dyn_cast_or_null<MDString>(ArgMD->getOperand(I)))
          if (S->getString() == "air.buffer") {
            IsBuffer = true;
            break;
          }
      if (!IsBuffer)
        continue;

      // Get the arg index from the first operand
      auto *IdxVAM = dyn_cast_or_null<ValueAsMetadata>(ArgMD->getOperand(0));
      if (!IdxVAM)
        continue;
      auto *IdxCI = dyn_cast<ConstantInt>(IdxVAM->getValue());
      if (!IdxCI)
        continue;
      unsigned ArgIdx = IdxCI->getZExtValue();
      if (ArgIdx >= Fn->arg_size())
        continue;

      Argument *Arg = Fn->getArg(ArgIdx);
      if (!Arg->getType()->isPointerTy())
        continue;
      Type *Pointee = nullptr;
      if (auto *Ty = PTM.get(Arg))
        Pointee = Ty;
      if (!Pointee)
        Pointee = PointeeTypeMap::inferFromUsage(Arg);
      // Follow through bitcasts if inference failed on the arg directly
      if (!Pointee || Pointee->isFloatTy()) {
        for (auto *U : Arg->users()) {
          if (auto *BC = dyn_cast<BitCastInst>(U)) {
            if (auto *Ty = PTM.get(BC)) {
              if (!Ty->isFloatTy()) {
                Pointee = Ty;
                break;
              }
            }
            Type *BcTy = PointeeTypeMap::inferFromUsage(BC);
            if (BcTy && !BcTy->isFloatTy()) {
              Pointee = BcTy;
              break;
            }
          }
        }
      }
      if (!Pointee)
        continue;

      StringRef TypeName;
      unsigned TypeSize = 0, TypeAlign = 0;
      if (Pointee->isBFloatTy()) {
        TypeName = "bfloat";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isFloatTy()) {
        TypeName = "float";
        TypeSize = 4;
        TypeAlign = 4;
      } else if (Pointee->isHalfTy()) {
        TypeName = "half";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(8)) {
        TypeName = "char";
        TypeSize = 1;
        TypeAlign = 1;
      } else if (Pointee->isIntegerTy(16)) {
        TypeName = "short";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(32)) {
        TypeName = "int";
        TypeSize = 4;
        TypeAlign = 4;
      } else {
        continue; // Unknown type, don't change
      }

      SmallVector<Metadata *, 16> NewOps;
      for (unsigned I = 0; I < ArgMD->getNumOperands(); I++) {
        Metadata *Op = ArgMD->getOperand(I);
        if (I + 1 < ArgMD->getNumOperands()) {
          if (auto *PrevS = dyn_cast_or_null<MDString>(ArgMD->getOperand(I))) {
            if (PrevS->getString() == "air.arg_type_name" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(MDString::get(Ctx, TypeName));
              I++; // skip original type name
              continue;
            }
            if (PrevS->getString() == "air.arg_type_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeSize)));
              I++; // skip original size
              continue;
            }
            if (PrevS->getString() == "air.arg_type_align_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeAlign)));
              I++; // skip original align
              continue;
            }
          }
        }
        NewOps.push_back(Op);
      }
      auto *NewArgMD = MDNode::get(Ctx, NewOps);
      ArgDescs->replaceOperandWith(A, NewArgMD);
    }
  }
}

// Map LLVM's in-memory AttrKind to the AIR-v1 bitcode attr-kind encoding (the
// two numbering spaces differ). Modern attrs Apple's reader rejects as "Unknown
// attribute kind" fall outside this whitelist and are dropped (sound: hints).
static std::optional<uint64_t> airEnumAttrKind(Attribute::AttrKind K) {
  switch (K) {
  case Attribute::NoAlias:
    return 9;
  case Attribute::NoUnwind:
    return 18;
  case Attribute::ReadNone:
    return 20;
  case Attribute::ReadOnly:
    return 21;
  case Attribute::NonNull:
    return 39;
  case Attribute::Convergent:
    return 43;
  case Attribute::WriteOnly:
    return 52;
  case Attribute::WillReturn:
    return 61;
  case Attribute::NoFree:
    return 62;
  case Attribute::NoSync:
    return 63;
  case Attribute::MustProgress:
    return 70;
  default:
    return std::nullopt;
  }
}

static std::optional<uint64_t> airIntAttrKind(Attribute::AttrKind K) {
  switch (K) {
  case Attribute::Alignment:
    return 1;
  case Attribute::Dereferenceable:
    return 41;
  case Attribute::DereferenceableOrNull:
    return 42;
  default:
    return std::nullopt;
  }
}

// Append the AIR-bitcode encoding of Attr to Grp; returns false for attrs
// outside the whitelist. captures(none) maps to the legacy nocapture kind.
static bool encodeAirAttr(const Attribute &Attr,
                          SmallVectorImpl<uint64_t> *Grp) {
  if (Attr.isEnumAttribute()) {
    if (auto BK = airEnumAttrKind(Attr.getKindAsEnum())) {
      if (Grp) {
        Grp->push_back(0);
        Grp->push_back(*BK);
      }
      return true;
    }
    return false;
  }
  if (Attr.isIntAttribute()) {
    if (Attr.getKindAsEnum() == Attribute::Captures) {
      if (!capturesNothing(Attr.getCaptureInfo()))
        return false;
      if (Grp) {
        Grp->push_back(0);
        Grp->push_back(11); // nocapture
      }
      return true;
    }
    if (auto BK = airIntAttrKind(Attr.getKindAsEnum())) {
      if (Grp) {
        Grp->push_back(1);
        Grp->push_back(*BK);
        Grp->push_back(Attr.getValueAsInt());
      }
      return true;
    }
    return false;
  }
  return false;
}

// True if at least one attribute in AS survives the AIR whitelist (string
// attributes always pass through).
static bool hasAirAttrs(const AttributeSet &AS) {
  for (Attribute Attr : AS) {
    if (Attr.isStringAttribute())
      return true;
    if (encodeAirAttr(Attr, nullptr))
      return true;
  }
  return false;
}

// Defined in separate .cpp files.
void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E);
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                        ArrayRef<const Constant *> Constants,
                        unsigned CodeSize);
void emitMetadataKindBlock(BitstreamWriter &W);
void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E,
                       MetadataEnumerator &MD);
void emitOperandBundleTagsBlock(BitstreamWriter &W);
void emitSinglethreadBlock(BitstreamWriter &W);
void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                       ValueEnumerator &E, const MetadataEnumerator &MD);

// Remove ptr-to-ptr bitcasts only where PTM records the SAME pointee on both
// sides; differing-type bitcasts are typed-pointer transitions and must stay.
static void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<BitCastInst *, 16> ToRemove;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || BC->getSrcTy() != BC->getDestTy())
          continue;
        Type *SrcPT = PTM.get(BC->getOperand(0));
        Type *DstPT = PTM.get(BC);
        if (!SrcPT || !DstPT)
          continue;
        if (SrcPT != DstPT)
          continue;
        ToRemove.push_back(BC);
      }
    }
    for (auto *BC : ToRemove) {
      PTM.remove(BC);
      BC->replaceAllUsesWith(BC->getOperand(0));
      BC->eraseFromParent();
    }
  }
}

// normalizeGEPs shape 4: rewrite single-index element GEPs on array globals to
// 2-index array GEPs matching the global's array pointee. Must run AFTER
// lowerConstantExprs to cover materialized constexpr GEPs.
static void normalizeArrayGlobalGEPs(Module &M) {
  Type *I64Ty = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // Loads/stores directly through an array global have element access type vs
    // array pointee; route them through an element-0 GEP.
    SmallVector<Instruction *, 4> DirectAccess;
    for (auto &BB : F)
      for (auto &I : BB) {
        Value *Ptr = nullptr;
        if (auto *LI = dyn_cast<LoadInst>(&I))
          Ptr = LI->getPointerOperand();
        else if (auto *SI = dyn_cast<StoreInst>(&I))
          Ptr = SI->getPointerOperand();
        if (!Ptr)
          continue;
        auto *GV = dyn_cast<GlobalVariable>(Ptr);
        if (!GV || !isa<ArrayType>(GV->getValueType()))
          continue;
        DirectAccess.push_back(&I);
      }
    for (Instruction *I : DirectAccess) {
      auto *GV = cast<GlobalVariable>(
          isa<LoadInst>(I) ? cast<LoadInst>(I)->getPointerOperand()
                           : cast<StoreInst>(I)->getPointerOperand());
      auto *Base = GetElementPtrInst::CreateInBounds(
          GV->getValueType(), GV,
          {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 0)});
      Base->insertBefore(I->getIterator());
      if (isa<LoadInst>(I))
        cast<LoadInst>(I)->setOperand(0, Base);
      else
        cast<StoreInst>(I)->setOperand(1, Base);
    }
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (GEP->getNumIndices() != 1)
            continue;
          auto *GV = dyn_cast<GlobalVariable>(GEP->getPointerOperand());
          if (!GV)
            continue;
          auto *AT = dyn_cast<ArrayType>(GV->getValueType());
          if (!AT)
            continue;
          if (GEP->getSourceElementType() == AT)
            continue;
          ToFix.push_back(GEP);
        }

    for (auto *GEP : ToFix) {
      auto *GV = cast<GlobalVariable>(GEP->getPointerOperand());
      auto *AT = cast<ArrayType>(GV->getValueType());
      Type *ElemTy = AT->getElementType();
      Type *SrcTy = GEP->getSourceElementType();
      Value *Idx = GEP->idx_begin()->get();

      Value *ElemIdx = nullptr;
      if (SrcTy == ElemTy) {
        ElemIdx = Idx;
      } else {
        uint64_t SrcSize = M.getDataLayout().getTypeAllocSize(SrcTy);
        uint64_t ElemSize = M.getDataLayout().getTypeAllocSize(ElemTy);
        auto *CI = dyn_cast<ConstantInt>(Idx);
        if (!CI || ElemSize == 0)
          continue;
        uint64_t ByteOff = CI->getZExtValue() * SrcSize;
        if (ByteOff % ElemSize != 0)
          continue;
        ElemIdx = ConstantInt::get(I64Ty, ByteOff / ElemSize);
      }

      auto *NewGEP = GetElementPtrInst::Create(
          AT, GV, {ConstantInt::get(I64Ty, 0), ElemIdx}, "",
          GEP->getIterator());
      NewGEP->setIsInBounds(GEP->isInBounds());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
  }
}

std::vector<uint8_t> emitMetalBitcode(Module &M, PointeeTypeMap &PTM) {
  SmallVector<char, 0> Buf;
  // Scope the writer so its destructor's FlushToWord() pads the final partial
  // 32-bit word before Buf is read; without it readers over-read and report
  // "truncated module".
  {
    BitstreamWriter W(Buf);

    // BC magic
    W.Emit('B', 8);
    W.Emit('C', 8);
    W.Emit(0xC0, 8);
    W.Emit(0xDE, 8);

    // IDENTIFICATION
    W.EnterSubblock(bitc::IDENTIFICATION_BLOCK_ID, 5);
    emitString(W, bitc::IDENTIFICATION_CODE_STRING, "MetalIR");
    {
      SmallVector<uint64_t, 1> V = {0};
      W.EmitRecord(bitc::IDENTIFICATION_CODE_EPOCH, V);
    }
    W.ExitBlock();

    // Pre-serialization IR fixups: bring the O3 module into the subset Metal v1
    // bitcode + AGX JIT accept, refining the PTM in place. Four ordered stages;
    // stage A's lowering reshapes the IR the later type fixups inspect.
    // Stage A: legalize / lower constructs the AGX JIT can't take.
    expandWideIntegers(M);
    lowerFreezeInsts(M);
    canonicalizeNNegZExt(M);
    scalarizeBoolVectorCasts(M);
    lowerCmpIntrinsics(M);
    lowerVectorPointerToInt(M);
    // scalarizeVectorSelects(M); // disabled for now
    lowerVectorSelects(M);
    removeRedundantBitcasts(M, PTM);
    // Stage B: GEP source-type normalization (shapes 1-3, see normalizeGEPs).
    normalizeGEPs(M, PTM);
    // Stage C: pointer-pointee type agreement.
    fixPhiIncomingTypes(M, PTM);
    fixMMAPointerSuffixMismatch(M, PTM);
    fixSelectPointerArms(M, PTM);
    scalarizeAggregateLoads(M);
    fixAccessTypeMismatch(M, PTM);

    // Stage D: materialize constexprs, then post-constexpr fixups.
    lowerConstantExprs(M);
    normalizeArrayGlobalGEPs(M);
    fixKernelArgMetadata(M, PTM);

    if (const char *pw = getenv("METAL_DUMP_PREWRITE")) {
      // A path value writes the prewrite module there; otherwise stderr.
      if (pw[0] && strcmp(pw, "1") != 0) {
        std::error_code EC;
        llvm::raw_fd_ostream os(pw, EC);
        if (!EC)
          M.print(os, nullptr);
      } else {
        M.print(llvm::errs(), nullptr);
      }
    }

    ValueEnumerator E(M, PTM);

    // MODULE_BLOCK (CodeSize=4)
    W.EnterSubblock(bitc::MODULE_BLOCK_ID, 4);

    {
      SmallVector<uint64_t, 1> V = {1};
      W.EmitRecord(bitc::MODULE_CODE_VERSION, V);
    }

    // PARAMATTR blocks BEFORE TYPE_BLOCK (Metal requires this order). One
    // PARAMATTR_GRP per unique (paramIdx, AS) tuple, one list per function.
    // Group ID 1 is reserved for the legacy MMA-load nocapture+readonly entry,
    // which lives on the call site, not the function declaration.
    struct GroupKey {
      unsigned ListIdx;
      AttributeSet AS;
      bool operator==(const GroupKey &O) const {
        return ListIdx == O.ListIdx && AS == O.AS;
      }
    };
    struct GroupKeyInfo {
      static GroupKey getEmptyKey() {
        return {~0u, DenseMapInfo<AttributeSet>::getEmptyKey()};
      }
      static GroupKey getTombstoneKey() {
        return {~0u - 1, DenseMapInfo<AttributeSet>::getTombstoneKey()};
      }
      static unsigned getHashValue(const GroupKey &K) {
        return hash_combine(K.ListIdx,
                            DenseMapInfo<AttributeSet>::getHashValue(K.AS));
      }
      static bool isEqual(const GroupKey &A, const GroupKey &B) {
        return A.ListIdx == B.ListIdx && A.AS == B.AS;
      }
    };

    DenseMap<GroupKey, unsigned, GroupKeyInfo> GroupID;
    SmallVector<GroupKey, 8> GroupOrder;
    auto getGroupID = [&](unsigned ListIdx, AttributeSet AS) -> unsigned {
      GroupKey K{ListIdx, AS};
      auto It = GroupID.find(K);
      if (It != GroupID.end())
        return It->second;
      unsigned ID = GroupID.size() + 1;
      GroupID[K] = ID;
      GroupOrder.push_back(K);
      return ID;
    };

    DenseMap<const Function *, unsigned> FnAttrListID;
    SmallVector<SmallVector<unsigned, 4>, 8> AttrLists;

    // Synthesize attr groups on the simdgroup-matrix intrinsic decls to match
    // Apple's `xcrun metal` AIR, else the macOS 13/14/15 driver rejects the
    // metallib. Bitcode ATTR_KIND_* values differ from LLVM's in-memory enum
    // (convergent is 6 in-memory, 43 in bitcode).
    enum : uint64_t {
      BK_NO_CAPTURE = 11,
      BK_NO_UNWIND = 18,
      BK_READ_ONLY = 21,
      BK_CONVERGENT = 43,
      BK_WRITEONLY = 52,
      BK_WILLRETURN = 61,
      BK_NOFREE = 62,
      BK_MUSTPROGRESS = 70,
    };
    struct SynthGroup {
      uint64_t Index; // ~0u for function attrs, param index (1-based) otherwise
      SmallVector<uint64_t, 6> EnumKinds;
    };
    SmallVector<std::pair<unsigned, SynthGroup>, 8> SynthGroups;
    DenseSet<const Function *> LocalUnnamedFns;
    auto addSynthGroup = [&](SynthGroup G) -> unsigned {
      unsigned ID = GroupID.size() + 1 + SynthGroups.size();
      SynthGroups.push_back({ID, std::move(G)});
      return ID;
    };

    bool HasMMALoad = false;
    for (auto &F : M) {
      if (F.getName().starts_with("air.simdgroup_matrix_8x8_load")) {
        HasMMALoad = true;
        break;
      }
    }
    if (HasMMALoad) {
      // Reserve group ID 1 + list ID 1 for the MMA call-site paramattr (the
      // FunctionWriter unconditionally threads `paramattr=1` into those calls).
      GroupKey K{/*ListIdx=*/1, AttributeSet()};
      GroupID[K] = 1;
      GroupOrder.push_back(K);
    }

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      AttributeList AL = F.getAttributes();
      SmallVector<unsigned, 4> GroupIDs;
      for (unsigned i = 0; i < F.arg_size(); i++) {
        AttributeSet AS = AL.getParamAttrs(i);
        if (!AS.hasAttributes() || !hasAirAttrs(AS))
          continue;
        GroupIDs.push_back(getGroupID(i + 1, AS));
      }
      AttributeSet RetAS = AL.getRetAttrs();
      if (RetAS.hasAttributes() && hasAirAttrs(RetAS))
        GroupIDs.push_back(getGroupID(0, RetAS));
      if (GroupIDs.empty())
        continue;
      unsigned ListID = AttrLists.size() + 1;
      AttrLists.push_back(std::move(GroupIDs));
      FnAttrListID[&F] = ListID;
    }

    // Now synthesize declaration attribute groups for the simdgroup intrinsics.
    for (auto &F : M) {
      if (!F.isDeclaration())
        continue;
      StringRef Name = F.getName();
      if (!Name.starts_with("air.simdgroup_matrix_8x8_"))
        continue;
      bool IsLoad = Name.contains("_load");
      bool IsStore = Name.contains("_store");

      // Function-attr group; order matches Apple's textual AIR.
      SynthGroup FnG;
      FnG.Index = ~0u;
      FnG.EnumKinds.push_back(BK_CONVERGENT);
      FnG.EnumKinds.push_back(BK_MUSTPROGRESS);
      if (IsLoad)
        FnG.EnumKinds.push_back(BK_NOFREE);
      FnG.EnumKinds.push_back(BK_NO_UNWIND);
      if (IsLoad)
        FnG.EnumKinds.push_back(BK_READ_ONLY);
      FnG.EnumKinds.push_back(BK_WILLRETURN);
      if (IsStore)
        FnG.EnumKinds.push_back(BK_WRITEONLY);
      unsigned FnGID = addSynthGroup(std::move(FnG));

      SmallVector<unsigned, 4> GroupIDs;
      GroupIDs.push_back(FnGID);

      // Pointer-arg group: load ptr is arg 0, store ptr is arg 1
      // (`nocapture readonly` / `nocapture writeonly`).
      if (IsLoad || IsStore) {
        SynthGroup PG;
        PG.Index = IsStore ? 2u : 1u; // 1-based param index
        PG.EnumKinds.push_back(BK_NO_CAPTURE);
        PG.EnumKinds.push_back(IsStore ? BK_WRITEONLY : BK_READ_ONLY);
        GroupIDs.push_back(addSynthGroup(std::move(PG)));
      }

      unsigned ListID = AttrLists.size() + 1;
      AttrLists.push_back(std::move(GroupIDs));
      FnAttrListID[&F] = ListID;
      LocalUnnamedFns.insert(&F);
    }

    if (!GroupID.empty() || !SynthGroups.empty()) {
      W.EnterSubblock(bitc::PARAMATTR_GROUP_BLOCK_ID, 4);
      for (auto &K : GroupOrder) {
        unsigned ID = GroupID.lookup(K);
        SmallVector<uint64_t, 16> Grp;
        Grp.push_back(ID);
        Grp.push_back(K.ListIdx);
        if (HasMMALoad && ID == 1 && K.AS.getNumAttributes() == 0) {
          // Legacy MMA call-site group: param 1 nocapture + readonly.
          Grp.push_back(0);
          Grp.push_back(11);
          Grp.push_back(0);
          Grp.push_back(21);
        } else {
          for (Attribute Attr : K.AS) {
            if (Attr.isStringAttribute()) {
              StringRef Key = Attr.getKindAsString();
              StringRef Val = Attr.getValueAsString();
              // 3 = string-key-only, 4 = key + value (matches upstream
              // writeAttributeGroupTable encoding).
              Grp.push_back(Val.empty() ? 3 : 4);
              for (char C : Key)
                Grp.push_back((unsigned char)C);
              Grp.push_back(0);
              if (!Val.empty()) {
                for (char C : Val)
                  Grp.push_back((unsigned char)C);
                Grp.push_back(0);
              }
            } else {
              encodeAirAttr(Attr, &Grp);
            }
          }
        }
        W.EmitRecord(bitc::PARAMATTR_GRP_CODE_ENTRY, Grp);
      }
      // Synthesized decl groups: (ID, Index, [0, kind]...); 0 = enum attr.
      for (auto &IDG : SynthGroups) {
        SmallVector<uint64_t, 16> Grp;
        Grp.push_back(IDG.first);        // group ID
        Grp.push_back(IDG.second.Index); // ~0u = function, else param index
        for (uint64_t Kind : IDG.second.EnumKinds) {
          Grp.push_back(0);
          Grp.push_back(Kind);
        }
        W.EmitRecord(bitc::PARAMATTR_GRP_CODE_ENTRY, Grp);
      }
      W.ExitBlock();

      W.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 4);
      if (HasMMALoad) {
        SmallVector<uint64_t, 2> List;
        List.push_back(1);
        W.EmitRecord(2, List);
      }
      for (auto &AL : AttrLists) {
        SmallVector<uint64_t, 8> List;
        for (unsigned ID : AL)
          List.push_back(ID);
        W.EmitRecord(2, List);
      }
      W.ExitBlock();
    }

    emitTypeBlock(W, E);

    // Target triple, required by the Metal GPU JIT; derive the canonical AIR
    // triple from present OS info rather than hardcoding it.
    {
      std::string T = M.getTargetTriple().str();
      if (T.empty() || T == "air")
        T = MetalVersion::fromTriple(M.getTargetTriple().str()).tripleString();
      emitString(W, bitc::MODULE_CODE_TRIPLE, T);
    }
    {
      auto DLStr = M.getDataLayoutStr();
      if (!DLStr.empty()) {
        emitString(W, bitc::MODULE_CODE_DATALAYOUT, DLStr);
      } else {
        emitString(W, bitc::MODULE_CODE_DATALAYOUT,
                   "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                   "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32"
                   "-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
                   "-v192:256:256-v256:256:256-v512:512:512"
                   "-v1024:1024:1024-n8:16:32");
      }
    }

    if (!M.getSourceFileName().empty())
      emitString(W, bitc::MODULE_CODE_SOURCE_FILENAME, M.getSourceFileName());

    // GLOBALVAR/FUNCTION records in globalValues order (globals first, matching
    // value ID assignment).
    for (auto *V : E.globalValues) {
      if (auto *G = dyn_cast<GlobalVariable>(V)) {
        SmallVector<uint64_t, 14> Ops;
        Ops.push_back(E.globalPtrTypeIdx(G)); // ptr-to-valueType
        Ops.push_back(G->isConstant() ? 1 : 0);
        Ops.push_back(G->hasInitializer()
                          ? E.moduleConstIdx(G->getInitializer()) + 1
                          : 0);
        Ops.push_back(encodeLinkage(G->getLinkage()));
        Ops.push_back(G->getAlign() ? Log2_32(G->getAlign()->value()) + 1 : 0);
        for (int J = 0; J < 3; J++)
          Ops.push_back(0);
        Ops.push_back(G->hasGlobalUnnamedAddr() ? 1 : 0);
        Ops.push_back(G->isExternallyInitialized() ? 1 : 0);
        Ops.push_back(0);
        Ops.push_back(0);
        Ops.push_back(G->getAddressSpace());
        Ops.push_back(0);
        W.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Ops);
      } else if (auto *Fn = dyn_cast<Function>(V)) {
        SmallVector<uint64_t, 17> Ops;
        Ops.push_back(E.typeIdx(Fn->getFunctionType()));
        Ops.push_back(Fn->getCallingConv());
        Ops.push_back(Fn->isDeclaration() ? 1 : 0);
        Ops.push_back(encodeLinkage(Fn->getLinkage()));
        // paramattr list ID (0 = none); +1 when HasMMALoad reserves list 1.
        unsigned ListID = 0;
        auto It = FnAttrListID.find(Fn);
        if (It != FnAttrListID.end())
          ListID = It->second + (HasMMALoad ? 1 : 0);
        Ops.push_back(ListID);
        Ops.push_back(0); // align
        // Field 9 (unnamed_addr) encoding: None=0, Global=1, Local=2. Apple's
        // simdgroup intrinsic decls are local_unnamed_addr (=2); the macOS-14
        // driver expects this to match.
        Ops.push_back(0); // 6: section
        Ops.push_back(0); // 7: visibility
        Ops.push_back(0); // 8: gc
        Ops.push_back(LocalUnnamedFns.contains(Fn) ? 2u
                                                   : 0u); // 9: unnamed_addr
        for (int J = 10; J < 16; J++)
          Ops.push_back(0);
        Ops.push_back(Fn->getAddressSpace());
        W.EmitRecord(bitc::MODULE_CODE_FUNCTION, Ops);
      }
    }

    emitConstantsBlock(W, E, E.moduleConstants, 5);
    emitMetadataKindBlock(W);

    // Share one MetadataEnumerator between the module METADATA_BLOCK (nodes
    // emitted) and per-function attachment blocks (referenced by ID).
    MetadataEnumerator MDEnum;
    MDEnum.collect(M, E);
    emitMetadataBlock(W, M, E, MDEnum);
    emitOperandBundleTagsBlock(W);
    emitSinglethreadBlock(W);

    for (auto *V : E.globalValues)
      if (auto *F = dyn_cast<Function>(V))
        if (!F->isDeclaration())
          emitFunctionBlock(W, *F, E, MDEnum);

    // VALUE_SYMTAB
    W.EnterSubblock(bitc::VALUE_SYMTAB_BLOCK_ID, 4);
    for (unsigned I = 0; I < E.globalValues.size(); I++) {
      if (!E.globalValues[I]->hasName())
        continue;
      SmallVector<uint64_t, 32> NV;
      NV.push_back(I);
      for (char C : E.globalValues[I]->getName())
        NV.push_back((uint64_t)(unsigned char)C);
      W.EmitRecord(bitc::VST_CODE_ENTRY, NV);
    }
    W.ExitBlock();

    W.ExitBlock(); // MODULE_BLOCK
  } // ~BitstreamWriter flushes the final word into Buf here.

  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metal
} // namespace llvm
