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

// Bank conflict padding: add PAD elements per row in TG buffers.
// Apple GPU has 32 banks of 4 bytes each (128-byte bank period).
// For stride = N (e.g. 64), all threads in the same column hit the same bank.
// Adding PAD elements (16 bytes) shifts successive rows by PAD banks,
// eliminating most bank conflicts for 8-row strips.
// Only applied when the padded buffer fits within 32KB TG limit.
//
// The pad amount must be 16 bytes regardless of element type:
//   float32 (4B) -> 4 elements, float16/bf16 (2B) -> 8, int8 (1B) -> 16.
// For integer types we skip padding entirely: int8 dots go through TG as f32
// (scatter converts i8->f32), but async copy writes raw bytes with a stride
// mismatch against the f32 MMA load.  Disabling padding avoids corrupting
// the stride arithmetic for these types.
inline constexpr int64_t TG_PAD = 4;

// Compute element-type-aware pad amount.  Returns 0 for integer element types
// (padding disabled), otherwise 16 / elemBytes so the pad is always 16 bytes.
int64_t tgPadForType(Type elemTy);

// Compute padded stride: only pad when stride is a multiple of 8 (likely
// bank-conflicting) AND total buffer size stays within 32KB TG budget.
// budget_bytes: estimated remaining TG budget for this buffer.
int64_t paddedStride(int64_t stride, int64_t budget_bytes, int64_t pad = TG_PAD);

Type getSimdgroupMatrixType(MLIRContext *ctx);

Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val);
Value makeI64(OpBuilder &b, Location loc, int64_t v);
Value makeI1False(OpBuilder &b, Location loc);

// The air.simdgroup_matrix_8x8_{load,store} intrinsic signature changed at
// macOS 16:
//   macOS <= 15 (canonical, matching Metal's metal_simdgroup_matrix header):
//     load:  (ptr, i64 elements_per_row, <2 x i64> origin, i1 transpose)
//     store: (<64 x T>, ptr, i64 elements_per_row, <2 x i64> origin, i1
//     transpose)
//   macOS >= 16 (3-vector shape/stride/offset form, current shipping target):
//     load:  (ptr, <2 x i64> shape, <2 x i64> stride, <2 x i64> origin)
//     store: (<64 x T>, ptr, <2 x i64> shape, <2 x i64> stride, <2 x i64>
//     origin)
// Selected at runtime via TRITON_MPS_TARGET_OS_MAJOR (default 16 = 3-vector).
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
