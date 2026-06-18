// Shared helpers for the tt.dot -> AIR simdgroup-matrix lowering patterns.
#pragma once

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::triton::applegpu {

void populateDotOpBlockedPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);
void populateDotOpAppleMmaPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

namespace dotcommon {

// Strength reduction: replace div/rem by constant with shift/mask when
// power-of-2.
bool isPowerOf2(int64_t v);
unsigned log2(int64_t v);

// divByConst: v / c  ->  v >> log2(c) when c is power-of-2, else DivUIOp.
Value divByConst(OpBuilder &b, Location loc, Value v, int64_t c);
// remByConst: v % c  ->  v & (c-1) when c is power-of-2, else RemUIOp.
Value remByConst(OpBuilder &b, Location loc, Value v, int64_t c);

// Bank-conflict padding: add 16 bytes per TG row (Apple GPU = 32 banks x 4B,
// 128-byte period) so successive rows shift banks. Pad is always 16 bytes:
// f32 -> 4 elems, f16/bf16 -> 8, int8 -> 16. Integer types skip padding: int8
// dots stage TG as f32 but async copy writes raw bytes, and padding would
// corrupt that stride mismatch. Only applied when the buffer fits 32KB.
inline constexpr int64_t TG_PAD = 4;

// Compute element-type-aware pad amount.  Returns 0 for integer element types
// (padding disabled), otherwise 16 / elemBytes so the pad is always 16 bytes.
int64_t tgPadForType(Type elemTy);

// Compute padded stride: only pad when stride is a multiple of 8 (likely
// bank-conflicting) AND total buffer size stays within 32KB TG budget.
// budget_bytes: estimated remaining TG budget for this buffer.
int64_t paddedStride(int64_t stride, int64_t budget_bytes,
                     int64_t pad = TG_PAD);

Type getSimdgroupMatrixType(MLIRContext *ctx);

Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val);
Value makeI64(OpBuilder &b, Location loc, int64_t v);
Value makeI1False(OpBuilder &b, Location loc);

// air.simdgroup_matrix_8x8_{load,store} signature differs by OS: macOS<=15 uses
// (elements_per_row, origin, transpose); macOS>=16 uses 3-vector
// shape/stride/origin. Selected via TRITON_MPS_TARGET_OS_MAJOR (default 16).
unsigned getTargetOSMajor();
bool useCanonicalSimdgroupSig();

LLVM::LLVMFuncOp getOrInsertIntrinsic(ConversionPatternRewriter &rewriter,
                                      ModuleOp mod, StringRef name,
                                      LLVM::LLVMFunctionType fnTy);

// One zero-filled simdgroup_matrix fragment (<64 x f32>), the identity C-in for
// an empty accumulator and the per-slot init for the fragment ABI struct.
Value getOrInsertSimdgroupInitFilled(ConversionPatternRewriter &rewriter,
                                     Location loc, ModuleOp mod,
                                     float fill = 0.0f);

LLVM::GlobalOp getOrCreateTGGlobal(ConversionPatternRewriter &rewriter,
                                   ModuleOp mod, StringRef name, int64_t size);

// Create a TG global with the specified element type.
// For bf16/f16, the array has 2x as many elements as f32 (same byte footprint
// since f32 is 4 bytes and bf16/f16 are 2 bytes).
LLVM::GlobalOp getOrCreateTypedTGGlobal(ConversionPatternRewriter &rewriter,
                                        ModuleOp mod, StringRef name,
                                        int64_t numElements, Type elemTy);

// Get the TG MMA load intrinsic and MMA multiply intrinsic for a given input
// element type. Returns: (loadIntrinsicName, mmaIntrinsicName, mmaMatVecType).
// The accumulator is always f32.
struct MMAIntrinsicInfo {
  const char *tgLoadName;
  const char *mmaName;
  Type matVecTy; // <64 x elemTy> for load, <64 x f32> for accumulator
};

MMAIntrinsicInfo getMMAIntrinsicInfo(MLIRContext *ctx, Type elemTy);

// Convert a value to the target MMA input type.
// For bf16/f16: truncate from f32 or leave as-is.
// For f32: extend from bf16/f16 or leave as-is.
Value toMmaInputType(OpBuilder &rewriter, Location loc, Value val,
                     Type targetTy);

// Convert a value to f32. Handles both float and integer element types.
Value toF32(OpBuilder &rewriter, Location loc, Value val, Type f32Ty);

// Convert f32 to the target element type. Handles both float and integer types.
Value fromF32(OpBuilder &rewriter, Location loc, Value val, Type targetTy);

unsigned &getDotCounter(MLIRContext *ctx);

} // namespace dotcommon
} // namespace mlir::triton::applegpu
