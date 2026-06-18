//===- MetalNaNMinMax.cpp - NaN-propagating min/max lowering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalNaNMinMax.h"
#include "Metal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-nan-min-max"

// AIR's native vector fmin/fmax tops out at width 4 (Apple's own frontend emits
// air.fast_fmax.v4f32 and nothing wider; verified against an emit-llvm probe).
// The .td rename table covers v2/v3/v4 only, so a wider min/max intrinsic from
// the SLP mid-end (e.g. llvm.maximum.v8f32) would otherwise reach the writer
// unrenamed and surface as an undefined symbol at PSO link. Split any min/max
// wider than 4 into <=4-wide chunks so the native vector path still applies.
static bool splitWideVectorMinMax(Module &M) {
  bool Changed = false;
  Intrinsic::ID Ids[] = {Intrinsic::minimum, Intrinsic::maximum,
                         Intrinsic::minnum, Intrinsic::maxnum};
  for (Intrinsic::ID ID : Ids) {
    SmallVector<Function *, 4> Decls;
    for (Function &F : M) {
      if (F.getIntrinsicID() != ID)
        continue;
      auto *VTy = dyn_cast<FixedVectorType>(F.getReturnType());
      if (VTy && VTy->getNumElements() > 4)
        Decls.push_back(&F);
    }
    for (Function *F : Decls) {
      auto *VTy = cast<FixedVectorType>(F->getReturnType());
      Type *EltTy = VTy->getElementType();
      unsigned N = VTy->getNumElements();
      unsigned Chunk = (N % 4 == 0) ? 4 : ((N % 2 == 0) ? 2 : 1);
      auto *ChunkTy = FixedVectorType::get(EltTy, Chunk);
      Function *Narrow = Intrinsic::getOrInsertDeclaration(&M, ID, {ChunkTy});
      SmallVector<CallInst *, 8> Calls;
      for (User *U : F->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (CI->getCalledFunction() == F)
            Calls.push_back(CI);
      for (CallInst *CI : Calls) {
        IRBuilder<> B(CI);
        Value *A = CI->getArgOperand(0);
        Value *Bv = CI->getArgOperand(1);
        Value *Acc = PoisonValue::get(VTy);
        for (unsigned Base = 0; Base < N; Base += Chunk) {
          SmallVector<int, 4> Idx;
          for (unsigned I = 0; I < Chunk; ++I)
            Idx.push_back(Base + I);
          Value *Ac = B.CreateShuffleVector(A, Idx);
          Value *Bc = B.CreateShuffleVector(Bv, Idx);
          Value *S = B.CreateCall(Narrow, {Ac, Bc});
          for (unsigned I = 0; I < Chunk; ++I)
            Acc = B.CreateInsertElement(Acc, B.CreateExtractElement(S, I),
                                        Base + I);
        }
        CI->replaceAllUsesWith(Acc);
        CI->eraseFromParent();
        Changed = true;
      }
      if (F->use_empty())
        F->eraseFromParent();
    }
  }
  return Changed;
}

// AIR has no llvm.vector.reduce.* counterpart (Apple's frontend lowers a
// horizontal reduction to a scalar tree). Expand each reduce intrinsic into a
// sequential element fold so the resulting scalar ops go through the normal
// rename/NaN-guard machinery. Covers the float (fmaximum/fminimum/fmax/fmin/
// fadd/fmul) and integer (add/mul/and/or/xor/umax/umin/smax/smin) forms the
// SLP mid-end produces.
static Value *foldReduce(IRBuilder<> &B, Intrinsic::ID ID, Value *Vec,
                         Value *Start) {
  auto *VTy = cast<FixedVectorType>(Vec->getType());
  unsigned N = VTy->getNumElements();
  Value *Acc = Start ? Start : B.CreateExtractElement(Vec, uint64_t(0));
  unsigned First = Start ? 0 : 1;
  for (unsigned I = First; I < N; ++I) {
    Value *E = B.CreateExtractElement(Vec, I);
    switch (ID) {
    case Intrinsic::vector_reduce_fadd:
      Acc = B.CreateFAdd(Acc, E);
      break;
    case Intrinsic::vector_reduce_fmul:
      Acc = B.CreateFMul(Acc, E);
      break;
    case Intrinsic::vector_reduce_add:
      Acc = B.CreateAdd(Acc, E);
      break;
    case Intrinsic::vector_reduce_mul:
      Acc = B.CreateMul(Acc, E);
      break;
    case Intrinsic::vector_reduce_and:
      Acc = B.CreateAnd(Acc, E);
      break;
    case Intrinsic::vector_reduce_or:
      Acc = B.CreateOr(Acc, E);
      break;
    case Intrinsic::vector_reduce_xor:
      Acc = B.CreateXor(Acc, E);
      break;
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_fmaximum:
      Acc = B.CreateBinaryIntrinsic(ID == Intrinsic::vector_reduce_fmaximum
                                        ? Intrinsic::maximum
                                        : Intrinsic::maxnum,
                                    Acc, E);
      break;
    case Intrinsic::vector_reduce_fmin:
    case Intrinsic::vector_reduce_fminimum:
      Acc = B.CreateBinaryIntrinsic(ID == Intrinsic::vector_reduce_fminimum
                                        ? Intrinsic::minimum
                                        : Intrinsic::minnum,
                                    Acc, E);
      break;
    case Intrinsic::vector_reduce_umax:
      Acc = B.CreateBinaryIntrinsic(Intrinsic::umax, Acc, E);
      break;
    case Intrinsic::vector_reduce_umin:
      Acc = B.CreateBinaryIntrinsic(Intrinsic::umin, Acc, E);
      break;
    case Intrinsic::vector_reduce_smax:
      Acc = B.CreateBinaryIntrinsic(Intrinsic::smax, Acc, E);
      break;
    case Intrinsic::vector_reduce_smin:
      Acc = B.CreateBinaryIntrinsic(Intrinsic::smin, Acc, E);
      break;
    default:
      return nullptr;
    }
  }
  return Acc;
}

static bool expandVectorReductions(Module &M) {
  static const Intrinsic::ID Reduces[] = {
      Intrinsic::vector_reduce_fadd,     Intrinsic::vector_reduce_fmul,
      Intrinsic::vector_reduce_add,      Intrinsic::vector_reduce_mul,
      Intrinsic::vector_reduce_and,      Intrinsic::vector_reduce_or,
      Intrinsic::vector_reduce_xor,      Intrinsic::vector_reduce_fmax,
      Intrinsic::vector_reduce_fmin,     Intrinsic::vector_reduce_fmaximum,
      Intrinsic::vector_reduce_fminimum, Intrinsic::vector_reduce_umax,
      Intrinsic::vector_reduce_umin,     Intrinsic::vector_reduce_smax,
      Intrinsic::vector_reduce_smin};
  bool Changed = false;
  for (Intrinsic::ID ID : Reduces) {
    SmallVector<Function *, 4> Decls;
    for (Function &F : M)
      if (F.getIntrinsicID() == ID)
        Decls.push_back(&F);
    for (Function *F : Decls) {
      SmallVector<CallInst *, 8> Calls;
      for (User *U : F->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (CI->getCalledFunction() == F)
            Calls.push_back(CI);
      for (CallInst *CI : Calls) {
        IRBuilder<> B(CI);
        bool Ordered = ID == Intrinsic::vector_reduce_fadd ||
                       ID == Intrinsic::vector_reduce_fmul;
        Value *Vec = CI->getArgOperand(Ordered ? 1 : 0);
        Value *Start = Ordered ? CI->getArgOperand(0) : nullptr;
        if (Value *R = foldReduce(B, ID, Vec, Start)) {
          CI->replaceAllUsesWith(R);
          CI->eraseFromParent();
          Changed = true;
        }
      }
      if (F->use_empty())
        F->eraseFromParent();
    }
  }
  return Changed;
}

// AIR has no bf16 fmin/fmax; the scalar .td renames only cover f32/f16. Promote
// a bf16 min/max to f32 (fpext, scalar op, fptrunc) so it reaches air.fmax.f32.
static bool promoteBF16MinMax(Module &M) {
  bool Changed = false;
  Intrinsic::ID Ids[] = {Intrinsic::minimum, Intrinsic::maximum,
                         Intrinsic::minnum, Intrinsic::maxnum};
  LLVMContext &Ctx = M.getContext();
  for (Intrinsic::ID ID : Ids) {
    SmallVector<Function *, 4> Decls;
    for (Function &F : M)
      if (F.getIntrinsicID() == ID && F.getReturnType()->isBFloatTy())
        Decls.push_back(&F);
    for (Function *F : Decls) {
      Function *F32 =
          Intrinsic::getOrInsertDeclaration(&M, ID, {Type::getFloatTy(Ctx)});
      SmallVector<CallInst *, 8> Calls;
      for (User *U : F->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (CI->getCalledFunction() == F)
            Calls.push_back(CI);
      for (CallInst *CI : Calls) {
        IRBuilder<> B(CI);
        Value *A = B.CreateFPExt(CI->getArgOperand(0), B.getFloatTy());
        Value *Bv = B.CreateFPExt(CI->getArgOperand(1), B.getFloatTy());
        Value *R = B.CreateFPTrunc(B.CreateCall(F32, {A, Bv}), CI->getType());
        CI->replaceAllUsesWith(R);
        CI->eraseFromParent();
        Changed = true;
      }
      if (F->use_empty())
        F->eraseFromParent();
    }
  }
  return Changed;
}

static bool nanMinMax(Module &M) {
  bool Changed = expandVectorReductions(M);
  Changed |= splitWideVectorMinMax(M);
  Changed |= promoteBF16MinMax(M);
  SmallVector<CallInst *, 16> Calls;
  for (Function &F : M) {
    Intrinsic::ID ID = F.getIntrinsicID();
    if (ID != Intrinsic::minimum && ID != Intrinsic::maximum)
      continue;
    for (User *U : F.users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (CI->getCalledFunction() == &F)
          Calls.push_back(CI);
  }
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI->getNextNode());
    Value *A = CI->getArgOperand(0);
    Value *Bv = CI->getArgOperand(1);
    Value *IsNaN = B.CreateFCmpUNO(A, Bv, "nan_check");
    Value *NaN = ConstantFP::getNaN(CI->getType());
    Value *Sel = B.CreateSelect(IsNaN, NaN, CI, CI->getName() + ".nan");
    CI->replaceAllUsesWith(Sel);
    cast<SelectInst>(Sel)->setOperand(2, CI);
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses MetalNaNMinMaxPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  return nanMinMax(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool MetalNaNMinMaxLegacy::runOnModule(Module &M) { return nanMinMax(M); }

char MetalNaNMinMaxLegacy::ID = 0;

INITIALIZE_PASS(MetalNaNMinMaxLegacy, DEBUG_TYPE, "Metal NaN-safe min/max",
                false, false)

ModulePass *llvm::createMetalNaNMinMaxLegacyPass() {
  return new MetalNaNMinMaxLegacy();
}
