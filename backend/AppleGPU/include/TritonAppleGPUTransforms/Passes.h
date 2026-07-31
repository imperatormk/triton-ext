#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

// Rewrite tt.dot with BlockedEncoding → AppleMmaEncoding
std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass();

// Strip efficient_layout from large gather ops
std::unique_ptr<mlir::Pass> createSimplifyGatherLayoutPass();

// Re-lay MMA epilogue stores into the simdgroup-shuffle blocked layout so the
// #mma -> #blocked convert_layout becomes a within-simdgroup shuffle.
std::unique_ptr<mlir::Pass> createStoreShuffleLayoutPass();

// Rotate the smaller dot operand's load one iteration ahead of the K-loop so
// it retires under the MMA block.
std::unique_ptr<mlir::Pass> createPrefetchDotOperandPass();

} // namespace mlir::triton::applegpu

// Generated pass declarations
#include "TritonAppleGPUTransforms/Passes.h.inc"
