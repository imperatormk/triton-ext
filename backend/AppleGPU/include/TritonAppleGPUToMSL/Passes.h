#ifndef TRITON_APPLEGPU_TO_MSL_PASSES_H
#define TRITON_APPLEGPU_TO_MSL_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir::triton::applegpu {

std::unique_ptr<mlir::Pass> createEmitMSLPass(std::string outPath = {});

} // namespace mlir::triton::applegpu

#endif // TRITON_APPLEGPU_TO_MSL_PASSES_H
