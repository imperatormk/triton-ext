// DotOpToLLVM: lower tt.dot to air simdgroup matrix intrinsics via TG memory.
//
// Strategy (tiled TG scatter, register-resident C):
//   Scatter/load 8 rows at a time to minimize TG memory usage.
//   TG buffer = 8 * max(K, N) floats -- phases alias the same memory.
//
//   1. For each 8-row strip tm: scatter A[8xK] -> TG, barrier, load A[tm][*],
//   barrier
//   2. For each 8-row strip tm: scatter C[8xN] -> TG, barrier, load C[tm][*],
//   barrier
//   3. For each 8-row strip tk: scatter B[8xN] -> TG, barrier, load B[tk][*],
//   barrier
//      then MMA: C[tm][tn] += A[tm][tk] * B[tk][tn] for all tm,tn
//   4. For each 8-row strip tm: store C[tm][*] -> TG, barrier, gather, barrier
//
// For 64x64x64: TG = 8*64 = 512 floats = 2 KB (vs 16 KB untiled).
// Fits within Apple's 32 KB TG limit even with large tiles.
//
// Supports arbitrary M*K x K*N where M,N,K are multiples of 8.
// Handles any blocked encoding (reads
// sizePerThread/threadsPerWarp/warpsPerCTA). Supports batched (3D+) dot: each
// batch gets its own TG region. Batch routing: dot_op operands (A/B) with mixed
// batch offsets use compile-time batch indices; blocked operands (C) with
// uniform offsets use runtime warpId.
//
// Two patterns:
//   DotOpBlockedConversion  — blocked encoding on C, any rank (batch-aware)
//   DotOpAppleMmaConversion — AppleMmaEncoding on C, rank-2 only
//
// The two pattern families live in DotOpBlockedConversion.cpp and
// DotOpAppleMmaConversion.cpp; shared helpers live in DotCommon.{h,cpp}.

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
