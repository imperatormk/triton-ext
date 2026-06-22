//===- PointerPointeeRepair.cpp - Pointer/pointee type agreement ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Metal v1 typed-pointer reader requires every pointer-consuming record
// (phi, select, load/store, call arg) to agree with the pointee the writer
// emits for the pointer. These transforms pin disagreeing edges through an
// identity bitcast (recorded in the PTM) or repoint them, one mismatch class
// per function.
//
//===----------------------------------------------------------------------===//

#include "PointerPointeeRepair.h"
#include "PointeeRules.h"
#include "PointerRepairUtil.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

// A pointer phi's record carries one pointee type; wrap incomings that resolve
// to a different type in an identity bitcast so the reader accepts the record.
void fixPhiIncomingTypes(Module &M, PointeeTypeMap &PTM) {
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
        // Only intervene when a concrete pointee exists. If every incoming
        // defaults to the per-AS fallback, leaving it untouched keeps
        // phi/incomings/consumers all on the same default - intervening would
        // create a spurious mismatch.
        Type *PhiPointee = requiredPhiPointee(PN, PTM);
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

void fixSelectPointerArms(Module &M, PointeeTypeMap &PTM) {
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
      Pointee = requiredSelectPointee(S, PTM);
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
void fixAccessTypeMismatch(Module &M, PointeeTypeMap &PTM) {
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
void fixMMAPointerSuffixMismatch(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto Calls = collectInsts<CallInst>(M, [](CallInst *CI) {
    return CI->getCalledFunction() &&
           CI->getCalledFunction()->getName().starts_with(
               "air.simdgroup_matrix_8x8_");
  });
  for (auto *CI : Calls) {
    StringRef Name = CI->getCalledFunction()->getName();
    Type *Elem = mmaElemFromName(Name, Ctx);
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

// Remove ptr-to-ptr bitcasts only where PTM records the SAME pointee on both
// sides; differing-type bitcasts are typed-pointer transitions and must stay.
void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM) {
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

} // namespace metal
} // namespace llvm
