// Shared surface for the TritonAppleGPU -> LLVM conversion patterns: the
// fragment-ABI eligibility analysis, the cross-pattern lowering helpers, and
// the per-group pattern registration entry points.
#pragma once

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "triton/Analysis/AxisInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::applegpu {

namespace ttg = mlir::triton::gpu;

// Per-TYPE fragment-ABI gate: returns the set of #mma tensor types whose every
// module-wide value rides the simdgroup-fragment ABI (admitted only when every
// such value is produced and consumed by recognized dot-chain / fragment ops).
llvm::DenseSet<Type> computeFragmentEligibleTypes(ModuleOp mod);

// air.simd_shuffle.<ty>(val, i16 srcLane): pull `val` from absolute lane.
Value emitFragShuffle(ConversionPatternRewriter &rewriter, Location loc,
                      ModuleOp mod, Value val, Value srcLaneI16);
// thread_index_in_simdgroup (lane id, i32).
Value emitLaneId(ConversionPatternRewriter &rewriter, Location loc,
                 ModuleOp mod);

// Predicate selecting the canonical thread per redundant free-variable group.
Value emitAppleRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod);

// Combine two predicates with AND, tolerating null operands.
Value maybeAnd(ConversionPatternRewriter &rewriter, Location loc, Value a,
               Value b);

// First-element scalar / compile-time constant of a 1D index tensor.
Value extractFirstElemScalar(Value tensor, bool peelModulo = false);
bool extractFirstElemConst(Value tensor, int64_t &out, bool peelModulo = false);

// Whole-tile store guard reconstructed from a rectangular mask.
Value computeRectStoreFullGuard(triton::StoreOp op,
                                ConversionPatternRewriter &rewriter,
                                Location loc);

// Reconstructed scalar base/stride/origin of an async-copy source pointer.
struct AsyncCopyPtrInfo {
  Value stride;   // Row stride scalar (MLIR, in elements)
  Value basePtr;  // Scalar base pointer (MLIR, tt.ptr)
  Value rowStart; // Scalar first-row index (MLIR, i32/i64), or nullptr if 0
  Value colStart; // Scalar first-col index (MLIR, i32/i64), or nullptr if 0
  // Compile-time fallbacks when the first-element offset is a FOLDED splat
  // constant (dense<C>) not a tt.splat of an SSA scalar. The pipeliner emits
  // the prefetched K-block offset this way; without capturing it every
  // prefetched slab read K-block 0, corrupting num_stages>=3.
  int64_t rowStartConst = 0;
  int64_t colStartConst = 0;
  // Row stride as a compile-time constant (inductor dense<C>); INT64_MIN means
  // use the `stride` SSA Value instead.
  int64_t strideConst = INT64_MIN;
};

bool extractAsyncCopyPtrInfo(Value ptrTensor, AsyncCopyPtrInfo &info,
                             bool allowModulo);

// Per-group pattern registration. The pass driver owns the centralized
// RewritePatternSet and calls each of these; the benefits live with the
// registration so the driver's relative-priority view stays in one place.
void populateConvertLayoutOpPattern(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns);
void populateAtomicOpPatterns(LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns);
void populateMemoryOpPatterns(LLVMTypeConverter &typeConverter,
                              RewritePatternSet &patterns);
void populateMiscOpPatterns(LLVMTypeConverter &typeConverter,
                            RewritePatternSet &patterns,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis);
void populateFragmentElementwisePatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns);
void populateAsyncCopyPatterns(LLVMTypeConverter &typeConverter,
                               RewritePatternSet &patterns,
                               ModuleAxisInfoAnalysis &axisInfoAnalysis);

} // namespace mlir::triton::applegpu
