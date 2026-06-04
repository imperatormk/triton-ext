//===- MetalAsyncCopyToCooperative.h - lower async copy ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Lower air.simdgroup_async_copy_2d into an inline cooperative threadgroup
/// copy (and air.wait_simdgroup_events into a threadgroup barrier) when the
/// module also uses air.simdgroup_matrix_8x8_* ops.  The AGX PSO compiler fails
/// "materializeAll" on any module that merely DECLARES air.simdgroup_async_copy
/// alongside simdgroup-matrix loads, so removing the intrinsic is required for
/// the pipeline-shared MMA path.  On Apple GPUs the async DMA carries no
/// measured perf benefit over a cooperative copy, so this is loss-free.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALASYNCCOPYTOCOOPERATIVE_H
#define LLVM_LIB_TARGET_METAL_METALASYNCCOPYTOCOOPERATIVE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalAsyncCopyToCooperativePass
    : public PassInfoMixin<MetalAsyncCopyToCooperativePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalAsyncCopyToCooperativeLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalAsyncCopyToCooperativeLegacy() : ModulePass(ID) {}
  static char ID;
};

ModulePass *createMetalAsyncCopyToCooperativeLegacyPass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALASYNCCOPYTOCOOPERATIVE_H
