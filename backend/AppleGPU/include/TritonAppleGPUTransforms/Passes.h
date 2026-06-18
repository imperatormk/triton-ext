#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

// Threadgroup-resident operand budget for the dot TG path (32KB Metal cap
// minus convert-scratch margin). Shared by the lowering gates and the
// WidenPipelinedStaging widen gate; they must not drift.
inline constexpr int64_t kTGResidentBudgetBytes = 32768;

// Rewrite tt.dot with BlockedEncoding → AppleMmaEncoding
std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass();

// Strip efficient_layout from large gather ops
std::unique_ptr<mlir::Pass> createSimplifyGatherLayoutPass();

// Re-lay MMA epilogue stores into the simdgroup-shuffle blocked layout so the
// #mma -> #blocked convert_layout becomes a within-simdgroup shuffle.
std::unique_ptr<mlir::Pass> createStoreShuffleLayoutPass();

// Grow num_stages=2 pipelined dot staging to 2 rotating SMEM slots
std::unique_ptr<mlir::Pass> createWidenPipelinedStagingPass();

} // namespace mlir::triton::applegpu

// Generated pass declarations
#include "TritonAppleGPUTransforms/Passes.h.inc"
