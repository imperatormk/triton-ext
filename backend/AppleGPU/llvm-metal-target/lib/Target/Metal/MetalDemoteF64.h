//===- MetalDemoteF64.h - Demote double to float ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Apple GPU / Metal has no `double` type; any kernel containing f64
/// operations crashes the Metal shader compiler (XPC connection interrupted).
/// This pass rewrites all f64 values/types to f32, dropping the now-identity
/// float<->float casts. Triton emits these f64 chains for shape-derived
/// scalars that `fptrunc` back to float anyway, so computing in f32 is safe.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALDEMOTEF64_H
#define LLVM_LIB_TARGET_METAL_METALDEMOTEF64_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalDemoteF64Pass : public PassInfoMixin<MetalDemoteF64Pass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalDemoteF64Legacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalDemoteF64Legacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALDEMOTEF64_H
