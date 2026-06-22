//===- CoopTensorLowering.h - matmul2d / cooperative_tensor glue -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// All writer plumbing specific to the matmul2d / cooperative_tensor feature
// (env METAL_COOP_MMA, default-off), kept out of the generic always-on
// pipeline: the `__tensorops_impl_` externally_defined section re-tag, the
// tensor-builtin arg→pointee table, and the tensor-handle pinning consulted by
// PointeeTypeMap::inferFromUsage. Each entry no-ops when no tensor-ops symbols
// are present.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_COOPTENSORLOWERING_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_COOPTENSORLOWERING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Type.h"

namespace llvm {
class Module;
class Value;
class LLVMContext;
namespace metal {
class PointeeTypeMap;

// Section name marking a declaration as airdyld-resolved. Set by
// retagTensorOpsExternallyDefined and consumed ~400 lines away by the irsymtab
// emit in BitcodeEmitter; shared here so the producer/consumer handshake is one
// string, not two literals.
inline constexpr llvm::StringRef kExternallyDefinedSection =
    "air.externally_defined";

// Re-tag `__tensorops_impl_*` declarations with no section as
// kExternallyDefinedSection so airdyld binds the real import (vs a noop stub)
// and the irsymtab emits. The MLIR frontend can drop the section in
// translation. No-op when no such declarations exist.
void retagTensorOpsExternallyDefined(Module &M);

// Tensor-builtin arg pinning: which pointee a given arg position of a tensor
// runtime builtin must carry (handle → %struct._tensor_t, data/extent/stride →
// i8). Null when the arg is unpinned. ONE table shared by tensorHandlePointee
// and fixTensorRuntimeArgTypes.
Type *requiredTensorArgPointee(StringRef Name, unsigned ArgNo,
                               LLVMContext &Ctx);

// The tensor-handle pinning consulted by inferFromUsage for a call arg: maps a
// tensor-builtin callee + arg position to its required pointee
// (%struct._tensor_t / i8 / matmul2d arg0's alloca'd type). Null when the
// callee/arg is not a tensor-handle case.
Type *tensorHandlePointee(StringRef CalleeName, unsigned ArgNo, Value *Ptr,
                          LLVMContext &Ctx);

// Route a tensor-builtin arg whose pointee disagrees with the required pointee
// through an identity bitcast pinned to it.
void fixTensorRuntimeArgTypes(Module &M, PointeeTypeMap &PTM);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_COOPTENSORLOWERING_H
