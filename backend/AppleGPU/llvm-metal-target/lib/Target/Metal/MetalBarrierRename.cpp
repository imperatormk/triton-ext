//===- MetalBarrierRename.cpp - Rename threadgroup barrier ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalBarrierRename.h"
#include "Metal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-barrier-rename"

static constexpr StringLiteral kBarrier("air.wg.barrier");
static constexpr StringLiteral kBarrierOld("air.threadgroup.barrier");

// The Metal 4 JIT accepts air.wg.barrier with either (2,1) or legacy (1,4)
// args, but rejects calls to air.threadgroup.barrier as "unlowered function
// call" - only the function name must be modern. See PASS_GUARDS.md sub-track
// J.
static bool barrierRename(Module &M) {
  Function *OldBarrier = M.getFunction(kBarrierOld);
  if (!OldBarrier)
    return false;

  Function *NewBarrier = M.getFunction(kBarrier);
  if (!NewBarrier)
    NewBarrier = Function::Create(OldBarrier->getFunctionType(),
                                  OldBarrier->getLinkage(), kBarrier, &M);

  bool Changed = false;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() == OldBarrier) {
            CI->setCalledFunction(NewBarrier);
            Changed = true;
          }

  if (OldBarrier->use_empty())
    OldBarrier->eraseFromParent();

  return Changed;
}

// Two adjacent identical barriers (same callee/constant args, nothing between)
// rendezvous the same threads twice; the second is a no-op. Only the
// exact-adjacent case is folded.
static bool dropAdjacentDuplicateBarriers(Module &M) {
  Function *Barrier = M.getFunction(kBarrier);
  if (!Barrier)
    return false;
  SmallVector<CallInst *, 8> Dead;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || CI->getCalledFunction() != Barrier)
          continue;
        auto *Next = dyn_cast_or_null<CallInst>(CI->getNextNode());
        if (!Next || Next->getCalledFunction() != Barrier ||
            Next->arg_size() != CI->arg_size())
          continue;
        bool Same = true;
        for (unsigned A = 0; A < CI->arg_size(); ++A) {
          auto *CA = dyn_cast<ConstantInt>(CI->getArgOperand(A));
          auto *NA = dyn_cast<ConstantInt>(Next->getArgOperand(A));
          if (!CA || !NA || CA->getValue() != NA->getValue()) {
            Same = false;
            break;
          }
        }
        if (Same)
          Dead.push_back(Next);
      }
  for (CallInst *CI : Dead)
    CI->eraseFromParent();
  return !Dead.empty();
}

PreservedAnalyses MetalBarrierRenamePass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  bool Changed = barrierRename(M);
  Changed |= dropAdjacentDuplicateBarriers(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool MetalBarrierRenameLegacy::runOnModule(Module &M) {
  bool Changed = barrierRename(M);
  Changed |= dropAdjacentDuplicateBarriers(M);
  return Changed;
}

char MetalBarrierRenameLegacy::ID = 0;

INITIALIZE_PASS(MetalBarrierRenameLegacy, DEBUG_TYPE, "Metal Barrier Rename",
                false, false)

ModulePass *llvm::createMetalBarrierRenameLegacyPass() {
  return new MetalBarrierRenameLegacy();
}
