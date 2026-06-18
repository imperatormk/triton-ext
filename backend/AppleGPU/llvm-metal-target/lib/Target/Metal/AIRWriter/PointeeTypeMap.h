//===- PointeeTypeMap.h - Typed-pointer reconstruction ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_POINTEETYPEMAP_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_POINTEETYPEMAP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
namespace metal {

// Side table mapping each opaque pointer to its reconstructed pointee type.
// Metal GPU JIT requires typed POINTER records in bitcode but LLVM 19+ only
// has opaque pointers; the custom bitcode writer consumes this map to emit
// typed records while the in-memory Module stays valid with opaque pointers.
//
// Metal GPU JIT rules encoded here: device (AS1) and TG (AS3) ptrs must be
// typed; when MMA intrinsics are present ALL device ptrs become float*; i1*
// crashes the JIT so remap to i8*.
//
// The pointee-type logic lives in TWO overlapping places that must stay in
// sync: PointeeTypeAnalysis::run (cached LLVM Analysis, must be fully
// self-contained as the PM may recompute it) and InferTypedPointersPass::run
// (Transform Pass that refines the map after IR mutations). The MMA + async
// override blocks are near-duplicated; change BOTH. Shared intrinsic-name
// constants below prevent at least those from diverging.
namespace mma_intrinsics {
inline constexpr const char *kLoad =
    "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
inline constexpr const char *kStore =
    "air.simdgroup_matrix_8x8_store.v64f32.p3f32";
inline constexpr const char *kLoadDev =
    "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
inline constexpr const char *kStoreDev =
    "air.simdgroup_matrix_8x8_store.v64f32.p1f32";
inline constexpr const char *kPrefix = "air.simdgroup_matrix_8x8_";
} // namespace mma_intrinsics

class PointeeTypeMap {
public:
  // Set the pointee type for a pointer value.
  void set(llvm::Value *ptr, llvm::Type *pointeeTy) { map[ptr] = pointeeTy; }

  // Get the pointee type, or nullptr if unknown.
  llvm::Type *get(llvm::Value *ptr) const {
    auto it = map.find(ptr);
    return it != map.end() ? it->second : nullptr;
  }

  // Check if a pointer has a known pointee type.
  bool has(llvm::Value *ptr) const { return map.count(ptr); }

  // Remove a pointer's entry (used when erasing instructions).
  void remove(llvm::Value *ptr) { map.erase(ptr); }

  // Iterate all entries.
  auto begin() const { return map.begin(); }
  auto end() const { return map.end(); }
  size_t size() const { return map.size(); }

  // ── Inference helpers ────────────────────────────────────────────────

  // Infer pointee type from how a pointer is used (loads, stores, GEPs).
  // Returns nullptr if no usage gives a clear type.
  static llvm::Type *inferFromUsage(llvm::Value *Ptr);
  static llvm::Type *
  inferFromUsage(llvm::Value *Ptr,
                 llvm::SmallPtrSetImpl<llvm::Value *> &Visited);

  // Apply the "MMA present → all device ptrs are float*" rule.
  void collapseDevicePointersToFloat(llvm::Module &M);

  // Apply the "i1* → i8*" rule.
  void remapI1ToI8(llvm::Module &M);

private:
  llvm::DenseMap<llvm::Value *, llvm::Type *> map;
};

// Pure function of the module (no analysis-manager state), so callers outside
// the new-PM machinery (e.g. the legacy metallib writer pass) can call it.
PointeeTypeMap buildPointeeTypeMap(llvm::Module &M);

struct PointeeTypeAnalysis : llvm::AnalysisInfoMixin<PointeeTypeAnalysis> {
  using Result = PointeeTypeMap;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_POINTEETYPEMAP_H
