#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

// Emit MSL (Metal Shading Language) source directly from TritonGPU IR. Writes
// the source to the file named by the TRITON_MSL_OUT env var (or stderr if
// unset). This is the sole terminal codegen path.
std::unique_ptr<mlir::Pass> createEmitMSLPass();

} // namespace mlir::triton::applegpu
