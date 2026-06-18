// DotOpToLLVM: lower tt.dot to air simdgroup matrix intrinsics via TG memory.
// Tiled TG-scatter, register-resident C: process 8-row strips so the TG buffer
// is 8*max(K,N) floats (phases alias it), fitting Apple's 32KB cap. M,N,K must
// be multiples of 8. Batch routing: A/B dot_op operands use compile-time batch
// indices; C blocked operands use runtime warpId.
//
// Two patterns (in DotOpBlockedConversion.cpp / DotOpAppleMmaConversion.cpp;
// shared helpers in DotCommon.{h,cpp}):
//   DotOpBlockedConversion  — blocked encoding on C, any rank (batch-aware)
//   DotOpAppleMmaConversion — AppleMmaEncoding on C, rank-2 only

#include "DotCommon.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::applegpu {

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit) {
  populateDotOpBlockedPattern(typeConverter, patterns, benefit);
  populateDotOpAppleMmaPatterns(typeConverter, patterns, benefit);
}

} // namespace mlir::triton::applegpu
