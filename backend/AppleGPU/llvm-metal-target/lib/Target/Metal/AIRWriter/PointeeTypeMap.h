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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
namespace metal {

// Side table mapping each opaque pointer to its reconstructed pointee type, so
// the custom bitcode writer can emit the typed POINTER records the Metal GPU
// JIT requires while the in-memory Module keeps opaque pointers. Rules: device
// (AS1)/TG (AS3) ptrs must be typed; with MMA intrinsics all device ptrs become
// float*; i1* crashes the JIT so remap to i8*.
//
// The pointee-typing rules shared by the one-shot analysis and the
// post-mutation repairs live once in PointeeRules.h; both call those functions.
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

// The async-copy event handle named-struct. The single owner of the "event_t"
// named type: the analysis (Phase 7) creates it via getOrCreateEventType when a
// module uses async copy; the enumerator and function writer only look it up by
// kEventTypeName and stay no-op when absent (creating an unused event_t would
// leak it into the emitted type table). LLVM 14 readers reject forward
// references, so it must exist before function-type processing.
inline constexpr llvm::StringRef kEventTypeName = "event_t";
llvm::StructType *getOrCreateEventType(llvm::LLVMContext &Ctx);

struct PointeeTypeAnalysis : llvm::AnalysisInfoMixin<PointeeTypeAnalysis> {
  using Result = PointeeTypeMap;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_POINTEETYPEMAP_H
