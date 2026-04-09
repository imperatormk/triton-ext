#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace metalir {

// ── Pass declarations ────────────────────────────────────────────────────
//
// Each pass has:
//   run()      — do the transform
//   needsRun() — static check: does the module still contain constructs
//                this pass should eliminate? Used as postcondition in
//                debug builds to catch pass misordering.

// Metal can't call functions — inline everything.
struct InlineNonKernelFunctionsPass
    : llvm::PassInfoMixin<InlineNonKernelFunctionsPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// GPU JIT crashes on struct phi nodes — split to scalar phis.
struct DecomposeStructPhisPass : llvm::PassInfoMixin<DecomposeStructPhisPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// ~63 ptr phi limit — convert ptr phis to i64.
struct PtrPhiToI64Pass : llvm::PassInfoMixin<PtrPhiToI64Pass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Pointer selects crash GPU JIT — convert to i64 selects.
struct PtrSelectToI64Pass : llvm::PassInfoMixin<PtrSelectToI64Pass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// LLVM intrinsics → AIR intrinsics (rename declarations).
struct LLVMToAIRIntrinsicsPass : llvm::PassInfoMixin<LLVMToAIRIntrinsicsPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Opaque → typed pointers (Metal GPU JIT requires typed).
struct InferTypedPointersPass : llvm::PassInfoMixin<InferTypedPointersPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Scalar params → ptr addrspace(2) constant buffer + load.
struct ScalarBufferPackingPass : llvm::PassInfoMixin<ScalarBufferPackingPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// System value calls → kernel params + !air.kernel metadata.
struct AIRSystemValuesPass : llvm::PassInfoMixin<AIRSystemValuesPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// i1 GEPs → i8 GEPs (Metal has no i1 memory type).
struct NormalizeI1PointersPass : llvm::PassInfoMixin<NormalizeI1PointersPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Alloca i64 sizes → i32 (Metal v1 bitcode requirement).
struct NormalizeAllocasPass : llvm::PassInfoMixin<NormalizeAllocasPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// Mark device loads volatile to prevent GPU JIT hoisting.
struct DeviceLoadsVolatilePass : llvm::PassInfoMixin<DeviceLoadsVolatilePass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool needsRun(llvm::Module &M);
};

// ── Analysis: MMA Presence ────────────────────────────────────────────────
// Cached answer to "does this module have simdgroup_matrix_8x8 intrinsics?"
// Queried by InferTypedPointers and the metalir bridge.

struct MMAPresence {
  bool hasMMA = false;
};

struct MMAPresenceAnalysis : llvm::AnalysisInfoMixin<MMAPresenceAnalysis> {
  using Result = MMAPresence;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

// ── Analysis: TG Memory Budget ───────────────────────────────────────────

struct TGMemoryBudget {
  static constexpr unsigned kMaxBytes = 32 * 1024;
  unsigned usedBytes = 0;
  void addGlobal(llvm::StringRef name, unsigned bytes);
  bool fits(unsigned additionalBytes) const;
};

struct TGMemoryAnalysis : llvm::AnalysisInfoMixin<TGMemoryAnalysis> {
  using Result = TGMemoryBudget;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

// ── Pipeline builder ─────────────────────────────────────────────────────

void buildMetalIRPipeline(llvm::ModulePassManager &MPM);

} // namespace metalir
