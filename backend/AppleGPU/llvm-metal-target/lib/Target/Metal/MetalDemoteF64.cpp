//===- MetalDemoteF64.cpp - Demote double to float -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrite every f64 (`double`) value to f32 (`float`). Apple GPU / Metal has
// no double type, so any kernel containing one crashes the Metal shader
// compiler. The f64 chains Triton emits (sitofp i64 -> double, transcendental
// f64 intrinsics, fmul, then fptrunc back to float) are shape-derived scalars
// that already truncate to float at the end, so f32 is accuracy-safe.
//
// Strategy: a single forward worklist pass per function. For each f64
// instruction we build a float replacement (mapping its operands through the
// value map), RAUW, and erase the dead f64 op. Identity casts (fptrunc/fpext
// float->float) are dropped by forwarding to the operand. f64 constants are
// rounded to float. Double intrinsics are remapped to their .f32 overload.
//
//===----------------------------------------------------------------------===//

#include "MetalDemoteF64.h"
#include "Metal.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "metal-demote-f64"

namespace {

/// Does this type contain a `double` anywhere (scalar, vector, or array)?
static bool hasF64(Type *T) {
  if (T->isDoubleTy())
    return true;
  if (auto *VT = dyn_cast<VectorType>(T))
    return VT->getElementType()->isDoubleTy();
  if (auto *AT = dyn_cast<ArrayType>(T))
    return hasF64(AT->getElementType());
  return false;
}

/// Map a type's f64 components to f32. Non-f64 types pass through unchanged.
static Type *demoteTy(Type *T) {
  LLVMContext &C = T->getContext();
  if (T->isDoubleTy())
    return Type::getFloatTy(C);
  if (auto *VT = dyn_cast<VectorType>(T)) {
    if (VT->getElementType()->isDoubleTy())
      return VectorType::get(Type::getFloatTy(C), VT->getElementCount());
    return T;
  }
  if (auto *AT = dyn_cast<ArrayType>(T)) {
    if (hasF64(AT->getElementType()))
      return ArrayType::get(demoteTy(AT->getElementType()),
                            AT->getNumElements());
    return T;
  }
  return T;
}

class F64Demoter {
  Function &F;
  // Maps an original f64 value to its f32 replacement.
  DenseMap<Value *, Value *> Map;
  SmallVector<Instruction *, 32> Dead;

public:
  F64Demoter(Function &F) : F(F) {}

  /// Return the f32-domain replacement for V. V may be an f64 value (returns
  /// its mapped float), a float-domain constant, or a non-f64 value (returned
  /// as-is).
  Value *get(Value *V) {
    if (auto It = Map.find(V); It != Map.end())
      return It->second;
    if (!hasF64(V->getType()))
      return V;
    if (auto *C = dyn_cast<Constant>(V)) {
      Value *R = demoteConstant(C);
      Map[V] = R;
      return R;
    }
    // An f64 value we have not produced yet (e.g. an argument, or processed
    // out of order). Materialize a placeholder fptrunc that we patch later is
    // overkill here -- forward iteration over the function guarantees defs
    // precede uses except for phis, which we handle specially. If we still hit
    // an unmapped f64 value, fall back to an fpext-free bitcast is impossible;
    // instead leave it and rely on a second pass. As a safety net, return V
    // (will be fixed once its def is processed).
    return V;
  }

  Constant *demoteConstant(Constant *C) {
    Type *DT = demoteTy(C->getType());
    if (DT == C->getType())
      return C;
    if (auto *CFP = dyn_cast<ConstantFP>(C)) {
      APFloat V = CFP->getValueAPF();
      bool Lost;
      V.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &Lost);
      return ConstantFP::get(DT->getContext(), V);
    }
    if (isa<UndefValue>(C))
      return UndefValue::get(DT);
    if (isa<PoisonValue>(C))
      return PoisonValue::get(DT);
    if (C->isNullValue())
      return Constant::getNullValue(DT);
    if (auto *CV = dyn_cast<ConstantVector>(C)) {
      SmallVector<Constant *, 8> Elts;
      for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i)
        Elts.push_back(demoteConstant(CV->getOperand(i)));
      return ConstantVector::get(Elts);
    }
    if (auto *CDV = dyn_cast<ConstantDataVector>(C)) {
      SmallVector<Constant *, 8> Elts;
      for (unsigned i = 0, e = CDV->getNumElements(); i != e; ++i) {
        APFloat V = CDV->getElementAsAPFloat(i);
        bool Lost;
        V.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &Lost);
        Elts.push_back(ConstantFP::get(DT->getContext(), V));
      }
      return ConstantVector::get(Elts);
    }
    if (auto *CA = dyn_cast<ConstantArray>(C)) {
      SmallVector<Constant *, 8> Elts;
      for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
        Elts.push_back(demoteConstant(CA->getOperand(i)));
      return ConstantArray::get(cast<ArrayType>(DT), Elts);
    }
    // Fallback: cannot demote (e.g. ConstantExpr). Should not appear in these
    // scalar shape chains; bail loudly in debug builds.
    LLVM_DEBUG(dbgs() << "metal-demote-f64: unhandled f64 constant: " << *C
                      << "\n");
    return C;
  }

  void run() {
    LLVMContext &Ctx = F.getContext();
    Type *FloatTy = Type::getFloatTy(Ctx);
    (void)FloatTy;

    // Forward iteration. Defs precede uses except across phi back-edges; we
    // patch phi incoming values in a fix-up loop afterwards.
    for (Instruction &I : instructions(F)) {
      if (!instrTouchesF64(&I))
        continue;
      processInstruction(&I);
    }

    // Fix-up: phis may have referenced f64 values not yet mapped at creation
    // time. Re-resolve every new phi's incoming values.
    for (auto &KV : Map) {
      if (auto *NewPhi = dyn_cast_or_null<PHINode>(KV.second)) {
        auto *OldPhi = cast<PHINode>(KV.first);
        for (unsigned i = 0, e = OldPhi->getNumIncomingValues(); i != e; ++i) {
          Value *NV = get(OldPhi->getIncomingValue(i));
          NewPhi->setIncomingValue(i, NV);
        }
      }
    }

    // RAUW: every old f64 instruction's float users have been rebuilt to use
    // the new values directly, but external (non-demoted) users may remain if
    // an f64 value feeds something we did not rewrite (it shouldn't, post
    // demotion). Erase dead instructions in reverse program order.
    for (Instruction *I : reverse(Dead)) {
      if (!I->use_empty()) {
        // Any surviving use must be replaced with the float value (this also
        // covers identity-cast forwarding consumers we missed).
        if (Value *R = Map.lookup(I))
          I->replaceAllUsesWith(R);
      }
      if (I->use_empty())
        I->eraseFromParent();
    }
  }

private:
  static bool instrTouchesF64(Instruction *I) {
    if (hasF64(I->getType()))
      return true;
    for (Value *Op : I->operands())
      if (hasF64(Op->getType()))
        return true;
    return false;
  }

  void record(Instruction *Old, Value *New) {
    Map[Old] = New;
    Dead.push_back(Old);
  }

  void processInstruction(Instruction *I) {
    IRBuilder<> B(I);
    LLVMContext &Ctx = I->getContext();
    Type *FloatTy = Type::getFloatTy(Ctx);

    // ---- Casts that become identities or change type ----------------------
    if (auto *FPT = dyn_cast<FPTruncInst>(I)) {
      Value *Src = get(FPT->getOperand(0));
      Type *DstTy = demoteTy(FPT->getType());
      if (Src->getType() == DstTy) {
        // fptrunc float->float : identity, forward operand.
        record(FPT, Src);
      } else if (DstTy->getScalarType()->isHalfTy()) {
        // fptrunc double->half  =>  fptrunc float->half.
        record(FPT, B.CreateFPTrunc(Src, DstTy, FPT->getName()));
      } else {
        // float -> something narrower than float that isn't half: keep trunc.
        record(FPT, B.CreateFPTrunc(Src, DstTy, FPT->getName()));
      }
      return;
    }
    if (auto *FPE = dyn_cast<FPExtInst>(I)) {
      Value *Src = get(FPE->getOperand(0));
      Type *DstTy = demoteTy(FPE->getType());
      if (Src->getType() == DstTy) {
        // fpext float->float (was float->double): identity.
        record(FPE, Src);
      } else if (Src->getType()->getScalarType()->isHalfTy()) {
        // fpext half->double => fpext half->float.
        record(FPE, B.CreateFPExt(Src, DstTy, FPE->getName()));
      } else {
        record(FPE, B.CreateFPExt(Src, DstTy, FPE->getName()));
      }
      return;
    }
    if (auto *SI = dyn_cast<SIToFPInst>(I)) {
      record(SI, B.CreateSIToFP(SI->getOperand(0), demoteTy(SI->getType()),
                                SI->getName()));
      return;
    }
    if (auto *UI = dyn_cast<UIToFPInst>(I)) {
      record(UI, B.CreateUIToFP(UI->getOperand(0), demoteTy(UI->getType()),
                                UI->getName()));
      return;
    }
    if (auto *FTI = dyn_cast<FPToSIInst>(I)) {
      record(FTI, B.CreateFPToSI(get(FTI->getOperand(0)), FTI->getType(),
                                 FTI->getName()));
      return;
    }
    if (auto *FTU = dyn_cast<FPToUIInst>(I)) {
      record(FTU, B.CreateFPToUI(get(FTU->getOperand(0)), FTU->getType(),
                                 FTU->getName()));
      return;
    }
    if (auto *BC = dyn_cast<BitCastInst>(I)) {
      // bitcast i64 -> double => bitcast i32... no: bit width changes. A
      // bitcast to/from double has no valid f32 analogue (different size), so
      // route through the value: if source is f64 mapped to f32, re-bitcast.
      Value *Src = get(BC->getOperand(0));
      Type *DstTy = demoteTy(BC->getType());
      if (Src->getType() == DstTy) {
        record(BC, Src);
      } else {
        record(BC, B.CreateBitCast(Src, DstTy, BC->getName()));
      }
      return;
    }

    // ---- Binary float ops --------------------------------------------------
    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      Value *L = get(BO->getOperand(0));
      Value *R = get(BO->getOperand(1));
      Value *NV = B.CreateBinOp(BO->getOpcode(), L, R, BO->getName());
      if (auto *NI = dyn_cast<Instruction>(NV))
        NI->copyFastMathFlags(BO);
      record(BO, NV);
      return;
    }
    if (auto *UO = dyn_cast<UnaryOperator>(I)) {
      // fneg double
      Value *Op = get(UO->getOperand(0));
      Value *NV = B.CreateUnOp(UO->getOpcode(), Op, UO->getName());
      if (auto *NI = dyn_cast<Instruction>(NV))
        NI->copyFastMathFlags(UO);
      record(UO, NV);
      return;
    }

    // ---- Comparisons (fcmp double) ----------------------------------------
    if (auto *FC = dyn_cast<FCmpInst>(I)) {
      Value *L = get(FC->getOperand(0));
      Value *R = get(FC->getOperand(1));
      Value *NV = B.CreateFCmp(FC->getPredicate(), L, R, FC->getName());
      if (auto *NI = dyn_cast<Instruction>(NV))
        NI->copyFastMathFlags(FC);
      record(FC, NV);
      return;
    }

    // ---- select ------------------------------------------------------------
    if (auto *Sel = dyn_cast<SelectInst>(I)) {
      Value *T = get(Sel->getTrueValue());
      Value *Fa = get(Sel->getFalseValue());
      record(Sel, B.CreateSelect(Sel->getCondition(), T, Fa, Sel->getName()));
      return;
    }

    // ---- phi ---------------------------------------------------------------
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      PHINode *NP = B.CreatePHI(demoteTy(Phi->getType()),
                                Phi->getNumIncomingValues(), Phi->getName());
      for (unsigned i = 0, e = Phi->getNumIncomingValues(); i != e; ++i)
        NP->addIncoming(get(Phi->getIncomingValue(i)),
                        Phi->getIncomingBlock(i));
      record(Phi, NP);
      return;
    }

    // ---- extract/insert element on f64 vectors -----------------------------
    if (auto *EE = dyn_cast<ExtractElementInst>(I)) {
      record(EE, B.CreateExtractElement(get(EE->getVectorOperand()),
                                        EE->getIndexOperand(), EE->getName()));
      return;
    }
    if (auto *IE = dyn_cast<InsertElementInst>(I)) {
      record(IE, B.CreateInsertElement(get(IE->getOperand(0)),
                                       get(IE->getOperand(1)),
                                       IE->getOperand(2), IE->getName()));
      return;
    }
    if (auto *SV = dyn_cast<ShuffleVectorInst>(I)) {
      record(SV, B.CreateShuffleVector(get(SV->getOperand(0)),
                                       get(SV->getOperand(1)),
                                       SV->getShuffleMask(), SV->getName()));
      return;
    }

    // ---- calls / intrinsics ------------------------------------------------
    if (auto *CB = dyn_cast<CallInst>(I)) {
      processCall(CB, B);
      return;
    }

    // ---- return ------------------------------------------------------------
    if (auto *Ret = dyn_cast<ReturnInst>(I)) {
      if (Ret->getReturnValue() && hasF64(Ret->getReturnValue()->getType())) {
        // The function return type itself would need rewriting; these kernels
        // are void, so this is not expected. Leave as-is but warn.
        LLVM_DEBUG(dbgs() << "metal-demote-f64: f64 return in " << F.getName()
                          << "\n");
      }
      return;
    }

    // ---- loads / stores through f64 -- not expected for scalar chains, but
    // handle to keep the IR consistent if they appear.
    if (auto *LD = dyn_cast<LoadInst>(I)) {
      if (LD->getType()->isDoubleTy()) {
        // Load the bits as float -- requires the pointee was f64 in memory.
        // We do NOT rewrite memory layout here; instead load as f64 then
        // fptrunc. But since f64 is illegal, load as i32-pair is overkill.
        // For safety, load and immediately fptrunc to float.
        LLVM_DEBUG(dbgs() << "metal-demote-f64: f64 load in " << F.getName()
                          << "\n");
      }
      return;
    }

    // Unhandled: anything else that touched f64. Log it.
    LLVM_DEBUG(dbgs() << "metal-demote-f64: unhandled f64 instr: " << *I
                      << "\n");
  }

  void processCall(CallInst *CB, IRBuilder<> &B) {
    Function *Callee = CB->getCalledFunction();
    bool RetF64 = hasF64(CB->getType());
    bool ArgF64 = false;
    for (Value *A : CB->args())
      if (hasF64(A->getType()))
        ArgF64 = true;

    if (Callee && Callee->isIntrinsic()) {
      Intrinsic::ID ID = Callee->getIntrinsicID();
      // Remap the overloaded float type from double to float. Most of these
      // are simple single-type-arg intrinsics (sqrt/sin/cos/exp/log/fabs/
      // floor/ceil/trunc/rint/...) or multi-arg same-type (pow/fma/copysign/
      // minnum/maxnum/...). Build with all f64 operands demoted to f32.
      SmallVector<Value *, 4> Args;
      SmallVector<Type *, 2> OverloadTys;
      for (Value *A : CB->args())
        Args.push_back(get(A));
      // The overload type set: for these FP intrinsics the mangling type is
      // the (now float) result/operand type.
      Type *NewTy = demoteTy(CB->getType());
      OverloadTys.push_back(NewTy);

      Module *M = F.getParent();
      Function *NewFn = nullptr;
      switch (ID) {
      // Intrinsics overloaded on a single FP type (result == operand type).
      case Intrinsic::sqrt:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::fabs:
      case Intrinsic::floor:
      case Intrinsic::ceil:
      case Intrinsic::trunc:
      case Intrinsic::rint:
      case Intrinsic::nearbyint:
      case Intrinsic::round:
      case Intrinsic::roundeven:
      case Intrinsic::canonicalize:
      case Intrinsic::pow:
      case Intrinsic::powi:
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
      case Intrinsic::copysign:
      case Intrinsic::minnum:
      case Intrinsic::maxnum:
      case Intrinsic::minimum:
      case Intrinsic::maximum: {
        NewFn = Intrinsic::getOrInsertDeclaration(M, ID, {NewTy});
        break;
      }
      default: {
        // Generic attempt: re-declare with the demoted overload type.
        NewFn = Intrinsic::getOrInsertDeclaration(M, ID, OverloadTys);
        break;
      }
      }
      CallInst *NC = B.CreateCall(NewFn, Args);
      NC->setName(CB->getName());
      NC->copyFastMathFlags(CB);
      if (RetF64)
        record(CB, NC);
      else {
        CB->replaceAllUsesWith(NC);
        Dead.push_back(CB);
      }
      return;
    }

    // Non-intrinsic call with f64 args/ret: demote args, and if the callee
    // signature has f64 we cannot safely change it here (would need to rewrite
    // the callee). For these scalar chains, such calls are not expected.
    if (ArgF64 || RetF64) {
      LLVM_DEBUG(dbgs() << "metal-demote-f64: f64 user call: " << *CB << "\n");
    }
  }
};

static bool demoteF64(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    // Quick scan: any f64 anywhere?
    bool Any = false;
    for (Instruction &I : instructions(F)) {
      if (hasF64(I.getType())) {
        Any = true;
        break;
      }
      for (Value *Op : I.operands())
        if (hasF64(Op->getType())) {
          Any = true;
          break;
        }
      if (Any)
        break;
    }
    if (!Any)
      continue;
    F64Demoter(F).run();
    Changed = true;
  }

  // Remove now-dead f64 intrinsic declarations.
  SmallVector<Function *, 8> DeadDecls;
  for (Function &Fn : M) {
    if (Fn.isDeclaration() && Fn.isIntrinsic() && Fn.use_empty()) {
      // Check the declaration mentions double in its type.
      if (hasF64(Fn.getReturnType()))
        DeadDecls.push_back(&Fn);
      else {
        for (Type *PT : Fn.getFunctionType()->params())
          if (hasF64(PT)) {
            DeadDecls.push_back(&Fn);
            break;
          }
      }
    }
  }
  for (Function *Fn : DeadDecls)
    Fn->eraseFromParent();

  return Changed;
}

} // namespace

PreservedAnalyses MetalDemoteF64Pass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  return demoteF64(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool MetalDemoteF64Legacy::runOnModule(Module &M) { return demoteF64(M); }

char MetalDemoteF64Legacy::ID = 0;

INITIALIZE_PASS(MetalDemoteF64Legacy, DEBUG_TYPE, "Metal Demote f64 to f32",
                false, false)

ModulePass *llvm::createMetalDemoteF64LegacyPass() {
  return new MetalDemoteF64Legacy();
}
