//===- MetalBarrierRename.cpp - Rename threadgroup barrier ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalBarrierRename.h"
#include "Metal.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-barrier-rename"

static constexpr StringLiteral kBarrier("air.wg.barrier");
static constexpr StringLiteral kBarrierOld("air.threadgroup.barrier");

// Sub-track J: arg-rewrite portion dropped. Real-context bisect (post-rename
// IR on dot3d-int8 / scan2d-bf16) showed the Metal 4 JIT accepts
// air.wg.barrier with either (2,1) or legacy (1,4) args; only the function
// name needs to be the modern one (call sites to air.threadgroup.barrier are
// rejected as "unlowered function call"). See PASS_GUARDS.md sub-track J.
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

PreservedAnalyses MetalBarrierRenamePass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return barrierRename(M) ? PreservedAnalyses::none()
                          : PreservedAnalyses::all();
}

bool MetalBarrierRenameLegacy::runOnModule(Module &M) {
  return barrierRename(M);
}

char MetalBarrierRenameLegacy::ID = 0;

INITIALIZE_PASS(MetalBarrierRenameLegacy, DEBUG_TYPE, "Metal Barrier Rename",
                false, false)

ModulePass *llvm::createMetalBarrierRenameLegacyPass() {
  return new MetalBarrierRenameLegacy();
}
