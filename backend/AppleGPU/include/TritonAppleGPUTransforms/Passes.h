#ifndef TRITON_APPLEGPU_TRANSFORMS_PASSES_H
#define TRITON_APPLEGPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass();
std::unique_ptr<mlir::Pass> createStoreShuffleLayoutPass();
std::unique_ptr<mlir::Pass> createMaskSelectArmLoadsPass();

} // namespace mlir::triton::applegpu

#include "TritonAppleGPUTransforms/Passes.h.inc"

#endif // TRITON_APPLEGPU_TRANSFORMS_PASSES_H
