//===- MetalLegalizeUnsupportedIR.h - Strip unsupported IR ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strip/lower newer-LLVM constructs the AIR v1 bitcode and AGX JIT can't
/// encode: lifetime intrinsics, >64-bit integer arithmetic, freeze, `nneg`
/// zext, `disjoint` or-flags, and llvm.scmp/llvm.ucmp.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALLEGALIZEUNSUPPORTEDIR_H
#define LLVM_LIB_TARGET_METAL_METALLEGALIZEUNSUPPORTEDIR_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalLegalizeUnsupportedIRPass
    : public PassInfoMixin<MetalLegalizeUnsupportedIRPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalLegalizeUnsupportedIRLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalLegalizeUnsupportedIRLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALLEGALIZEUNSUPPORTEDIR_H
