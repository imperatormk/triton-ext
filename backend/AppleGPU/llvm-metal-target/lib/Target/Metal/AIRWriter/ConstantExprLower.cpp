//===- ConstantExprLower.cpp - Lower ConstantExprs to instructions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstantExprLower.h"
#include "MetalConstraints.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

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

} // namespace metal
} // namespace llvm
