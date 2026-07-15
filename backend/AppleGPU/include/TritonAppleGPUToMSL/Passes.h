#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

std::unique_ptr<mlir::Pass> createEmitMSLPass();

} // namespace mlir::triton::applegpu
