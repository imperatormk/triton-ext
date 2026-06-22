//===- NormalizeGEPs.cpp - Normalize GEP source element types -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Metal v1 typed-pointer reader rejects a GEP whose source element type
// differs from the base pointer's typed pointee. These transforms reconcile the
// two across four GEP shapes. Shape ordering inside normalizeGEPs is
// load-bearing (see the inline note); shape 4 (array globals) runs separately
// after lowerConstantExprs.
//
//===----------------------------------------------------------------------===//

#include "NormalizeGEPs.h"
#include "MetalConstraints.h"
#include "PointeeRules.h"
#include "PointerRepairUtil.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

// Normalize GEP source element types to match the base pointer's typed pointee
// (the Metal v1 reader rejects mismatches). Phase ordering is load-bearing:
// phase 1 (vector linearization) runs module-wide first so phases 2/3 see the
// fresh half/i8 GEPs; phase 2 (MMA retype) is one-shot; phase 3 (byte-stride)
// iterates to a fixpoint. Shape 4 (array globals) is normalizeArrayGlobalGEPs.
void normalizeGEPs(Module &M, PointeeTypeMap &PTM) {
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
    if (auto *AT = dyn_cast<ArrayType>(SrcTy))
      return AT->getElementType()->isIntegerTy(8) &&
             (GEP->getPointerAddressSpace() == metal::AS::Device ||
              GEP->getPointerAddressSpace() == metal::AS::Threadgroup);
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
        bool SameSize = SrcTy->getPrimitiveSizeInBits() == 32 ||
                        (SrcTy->isArrayTy() && DL.getTypeAllocSize(SrcTy) == 4);
        if (SameSize) {
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

    {
      SmallVector<GetElementPtrInst *, 8> GEPs;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
            GEPs.push_back(GEP);
      for (auto *GEP : GEPs)
        reconcileGEPBaseType(GEP, PTM);
    }
  }
}

// normalizeGEPs shape 4: rewrite single-index element GEPs on array globals to
// 2-index array GEPs matching the global's array pointee. Must run AFTER
// lowerConstantExprs to cover materialized constexpr GEPs.
void normalizeArrayGlobalGEPs(Module &M) {
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

} // namespace metal
} // namespace llvm
