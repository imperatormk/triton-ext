#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::applegpu {

// Lower AppleMmaEncoding tt.dot → simdgroup_multiply_accumulate LLVM calls
std::unique_ptr<mlir::Pass> createConvertTritonAppleGPUToLLVMPass();

// Lower remaining gpu.thread_id / gpu.block_dim → air intrinsics / constants
std::unique_ptr<mlir::Pass> createLowerGPUToAirPass();

// Emit MSL (Metal Shading Language) source directly from TritonGPU IR, with no
// dependence on air.* intrinsics. Writes the source to the file named by the
// TRITON_MSL_OUT env var (or stderr if unset). Terminal alternative to the
// AIR/metallib path.
std::unique_ptr<mlir::Pass> createEmitMSLPass();

// Populate just the dot op patterns (for use in combined lowering passes)
void populateDotOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                 mlir::RewritePatternSet &patterns,
                                 mlir::PatternBenefit benefit = 1);

// Populate load/store/addptr patterns
void populateLoadStoreToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                     mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit = 1);

// Register all Apple GPU → LLVM passes with the MLIR pass registry
// (for use with triton-opt / mlir-opt command line tools).
void registerTritonAppleGPUToLLVMPasses();

} // namespace mlir::triton::applegpu
