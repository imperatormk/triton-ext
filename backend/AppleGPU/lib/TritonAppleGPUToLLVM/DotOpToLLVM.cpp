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

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdlib>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
using namespace mlir::triton::applegpu;

namespace {

// Strength reduction: replace div/rem by constant with shift/mask when
// power-of-2.
static bool isPowerOf2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }
static unsigned log2(int64_t v) {
  unsigned r = 0;
  while ((1LL << r) < v)
    ++r;
  return r;
}

// Morton (Z-order) deinterleaving: extract even/odd bits from an integer.
// For warp grid mapping: even bits → row index, odd bits → column index.
// This gives cache-local (Z-curve) traversal of the output tile grid.
// mortonDeinterleaveEven: extract bits 0, 2, 4, ... → compact them to 0, 1, 2,
// ... mortonDeinterleaveOdd:  extract bits 1, 3, 5, ... → compact them to 0, 1,
// 2, ...
static Value mortonDeinterleaveEven(OpBuilder &b, Location loc, Value id,
                                    unsigned numBits) {
  // result = bit0 | (bit2 << 1) | (bit4 << 2) | ...
  Value result = arith::ConstantIntOp::create(b, loc, 0, 32);
  for (unsigned i = 0; i < numBits; ++i) {
    unsigned srcBit = i * 2; // even bits: 0, 2, 4, ...
    // Extract bit: (id >> srcBit) & 1
    Value shifted =
        (srcBit > 0)
            ? arith::ShRUIOp::create(
                  b, loc, id, arith::ConstantIntOp::create(b, loc, srcBit, 32))
            : id;
    Value bit = arith::AndIOp::create(
        b, loc, shifted, arith::ConstantIntOp::create(b, loc, 1, 32));
    // Place at destination position i
    if (i > 0)
      bit = arith::ShLIOp::create(b, loc, bit,
                                  arith::ConstantIntOp::create(b, loc, i, 32));
    result = arith::OrIOp::create(b, loc, result, bit);
  }
  return result;
}

static Value mortonDeinterleaveOdd(OpBuilder &b, Location loc, Value id,
                                   unsigned numBits) {
  // result = bit1 | (bit3 << 1) | (bit5 << 2) | ...
  Value result = arith::ConstantIntOp::create(b, loc, 0, 32);
  for (unsigned i = 0; i < numBits; ++i) {
    unsigned srcBit = i * 2 + 1; // odd bits: 1, 3, 5, ...
    Value shifted = arith::ShRUIOp::create(
        b, loc, id, arith::ConstantIntOp::create(b, loc, srcBit, 32));
    Value bit = arith::AndIOp::create(
        b, loc, shifted, arith::ConstantIntOp::create(b, loc, 1, 32));
    if (i > 0)
      bit = arith::ShLIOp::create(b, loc, bit,
                                  arith::ConstantIntOp::create(b, loc, i, 32));
    result = arith::OrIOp::create(b, loc, result, bit);
  }
  return result;
}

// Check if Morton order is applicable: both dimensions must be equal powers of
// 2 (square warp grid). For non-square grids, Morton deinterleaving doesn't
// produce a valid bijection — e.g., for 1x4, Morton yields only 2 distinct
// column values instead of 4, causing warp collisions. Returns the number of
// bits per dimension (log2 of dimension size). Returns 0 if Morton is not
// applicable.
static unsigned mortonBitsPerDim(int64_t wM, int64_t wN) {
  if (wM != wN)
    return 0; // non-square: can't use Morton
  if (!isPowerOf2(wM))
    return 0;
  if (wM <= 1)
    return 0; // single warp: nothing to reorder
  return log2(wM);
}

// divByConst: v / c  →  v >> log2(c) when c is power-of-2, else DivUIOp.
static Value divByConst(OpBuilder &b, Location loc, Value v, int64_t c) {
  if (isPowerOf2(c)) {
    unsigned shift = log2(c);
    if (shift == 0)
      return v;
    Value shAmt = arith::ConstantIntOp::create(b, loc, shift, 32);
    return arith::ShRUIOp::create(b, loc, v, shAmt);
  }
  Value cv = arith::ConstantIntOp::create(b, loc, c, 32);
  return arith::DivUIOp::create(b, loc, v, cv);
}

// remByConst: v % c  →  v & (c-1) when c is power-of-2, else RemUIOp.
static Value remByConst(OpBuilder &b, Location loc, Value v, int64_t c) {
  if (isPowerOf2(c)) {
    Value mask = arith::ConstantIntOp::create(b, loc, c - 1, 32);
    return arith::AndIOp::create(b, loc, v, mask);
  }
  Value cv = arith::ConstantIntOp::create(b, loc, c, 32);
  return arith::RemUIOp::create(b, loc, v, cv);
}

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
static constexpr int64_t TG_PAD = 4;

// Compute element-type-aware pad amount.  Returns 0 for integer element types
// (padding disabled), otherwise 16 / elemBytes so the pad is always 16 bytes.
static int64_t tgPadForType(Type elemTy) {
  if (isa<IntegerType>(elemTy))
    return 0; // no padding for int8/int16/int32
  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  if (elemBytes == 0)
    return TG_PAD;       // fallback
  return 16 / elemBytes; // 4 for f32, 8 for f16/bf16
}

// Compute padded stride: only pad when stride is a multiple of 8 (likely
// bank-conflicting) AND total buffer size stays within 32KB TG budget.
// budget_bytes: estimated remaining TG budget for this buffer.
static int64_t paddedStride(int64_t stride, int64_t budget_bytes,
                            int64_t pad = TG_PAD) {
  if (pad == 0)
    return stride;
  if (stride % 8 != 0)
    return stride; // odd strides don't bank-conflict
  int64_t padded = stride + pad;
  // 8 rows * padded * 4 bytes/f32 must fit in budget
  if (8 * padded * 4 > budget_bytes)
    return stride;
  return padded;
}

static Type getSimdgroupMatrixType(MLIRContext *ctx) {
  return LLVM::getVectorType(Float32Type::get(ctx), 64);
}

static Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val) {
  auto ty = LLVM::getVectorType(IntegerType::get(b.getContext(), 64), 2);
  Value vec = UndefOp::create(b, loc, ty);
  Value va = arith::ConstantIntOp::create(b, loc, a, 64);
  Value vb = arith::ConstantIntOp::create(b, loc, b_val, 64);
  Value i0 = arith::ConstantIntOp::create(b, loc, 0, 32);
  Value i1 = arith::ConstantIntOp::create(b, loc, 1, 32);
  vec = InsertElementOp::create(b, loc, ty, vec, va, i0);
  vec = InsertElementOp::create(b, loc, ty, vec, vb, i1);
  return vec;
}

static Value makeI64(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantIntOp::create(b, loc, v, 64);
}
static Value makeI1False(OpBuilder &b, Location loc) {
  return arith::ConstantIntOp::create(b, loc, 0, 1);
}

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
static unsigned getTargetOSMajor() {
  if (const char *e = std::getenv("TRITON_MPS_TARGET_OS_MAJOR")) {
    unsigned v = std::atoi(e);
    if (v)
      return v;
  }
  return 16; // default = current shipping target (3-vector signature)
}
static bool useCanonicalSimdgroupSig() { return getTargetOSMajor() <= 15; }

static LLVMFuncOp getOrInsertIntrinsic(ConversionPatternRewriter &rewriter,
                                       ModuleOp mod, StringRef name,
                                       LLVMFunctionType fnTy) {
  if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto fn =
      LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy, Linkage::External);
  // The simdgroup-matrix intrinsics need the same attributes Apple's `xcrun
  // metal` emits — the macOS 13/14/15 Metal driver rejects the declarations
  // otherwise ("Compiler encountered an internal error"); macOS 26 is lenient.
  // The attributes are universal (Apple emits them on all OSes), so safe both.
  if (name.contains("simdgroup")) {
    auto *ctx = mod.getContext();
    auto unit = UnitAttr::get(ctx);
    bool isLoad = name.contains("_load");
    bool isStore = name.contains("_store");

    SmallVector<Attribute> pass;
    auto add = [&](StringRef kw) { pass.push_back(StringAttr::get(ctx, kw)); };
    add("convergent");
    add("mustprogress");
    if (isLoad)
      add("nofree");
    add("nounwind");
    add("willreturn");
    fn.setPassthroughAttr(ArrayAttr::get(ctx, pass));

    // readonly/writeonly: not valid as function keyword attrs in modern LLVM,
    // expressed via memory effects instead (semantics identical).
    if (isLoad || isStore) {
      auto mr = isStore ? LLVM::ModRefInfo::Mod : LLVM::ModRefInfo::Ref;
      fn.setMemoryEffectsAttr(
          LLVM::MemoryEffectsAttr::get(ctx, {mr, mr, mr, mr, mr, mr}));
    }

    fn.setUnnamedAddr(LLVM::UnnamedAddr::Local);

    // Matrix pointer arg: nocapture readonly (load) / nocapture writeonly
    // (store). Load ptr is arg 0; store ptr is arg 1.
    if (isLoad || isStore) {
      unsigned ptrArg = isStore ? 1u : 0u;
      unsigned nArgs = fnTy.getNumParams();
      SmallVector<Attribute> argDicts(nArgs, DictionaryAttr::get(ctx, {}));
      SmallVector<NamedAttribute> ptrAttrs;
      ptrAttrs.push_back(NamedAttribute(
          StringAttr::get(ctx, LLVM::LLVMDialect::getNoCaptureAttrName()),
          unit));
      ptrAttrs.push_back(NamedAttribute(
          StringAttr::get(ctx, isStore
                                   ? LLVM::LLVMDialect::getWriteOnlyAttrName()
                                   : LLVM::LLVMDialect::getReadonlyAttrName()),
          unit));
      argDicts[ptrArg] = DictionaryAttr::get(ctx, ptrAttrs);
      fn.setArgAttrsAttr(ArrayAttr::get(ctx, argDicts));
    }
  }
  return fn;
}

static LLVM::GlobalOp getOrCreateTGGlobal(ConversionPatternRewriter &rewriter,
                                          ModuleOp mod, StringRef name,
                                          int64_t size) {
  if (auto g = mod.lookupSymbol<LLVM::GlobalOp>(name))
    return g;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto f32Ty = Float32Type::get(mod.getContext());
  auto arrTy = LLVMArrayType::get(f32Ty, size);
  return LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy,
                                /*isConstant=*/false, LLVM::Linkage::Internal,
                                name,
                                /*value=*/Attribute(),
                                /*alignment=*/4,
                                /*addrspace=*/3u);
}

// Create a TG global with the specified element type.
// For bf16/f16, the array has 2x as many elements as f32 (same byte footprint
// since f32 is 4 bytes and bf16/f16 are 2 bytes).
static LLVM::GlobalOp
getOrCreateTypedTGGlobal(ConversionPatternRewriter &rewriter, ModuleOp mod,
                         StringRef name, int64_t numElements, Type elemTy) {
  if (auto g = mod.lookupSymbol<LLVM::GlobalOp>(name))
    return g;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto arrTy = LLVMArrayType::get(elemTy, numElements);
  unsigned alignment = isa<Float32Type>(elemTy) ? 4 : 2;
  return LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy,
                                /*isConstant=*/false, LLVM::Linkage::Internal,
                                name,
                                /*value=*/Attribute(),
                                /*alignment=*/alignment,
                                /*addrspace=*/3u);
}

// Get the TG MMA load intrinsic and MMA multiply intrinsic for a given input
// element type. Returns: (loadIntrinsicName, mmaIntrinsicName, mmaMatVecType).
// The accumulator is always f32.
struct MMAIntrinsicInfo {
  const char *tgLoadName;
  const char *mmaName;
  Type matVecTy; // <64 x elemTy> for load, <64 x f32> for accumulator
};

static MMAIntrinsicInfo getMMAIntrinsicInfo(MLIRContext *ctx, Type elemTy) {
  MMAIntrinsicInfo info;
  if (elemTy.isBF16()) {
    info.tgLoadName = "air.simdgroup_matrix_8x8_load.v64bf16.p3bf16";
    info.mmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32."
                   "v64bf16.v64bf16.v64f32";
    info.matVecTy = LLVM::getVectorType(BFloat16Type::get(ctx), 64);
  } else if (elemTy.isF16()) {
    info.tgLoadName = "air.simdgroup_matrix_8x8_load.v64f16.p3f16";
    info.mmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16."
                   "v64f16.v64f32";
    info.matVecTy = LLVM::getVectorType(Float16Type::get(ctx), 64);
  } else {
    info.tgLoadName = "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
    info.mmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32."
                   "v64f32.v64f32";
    info.matVecTy = LLVM::getVectorType(Float32Type::get(ctx), 64);
  }
  return info;
}

// Convert a value to the target MMA input type.
// For bf16/f16: truncate from f32 or leave as-is.
// For f32: extend from bf16/f16 or leave as-is.
static Value toMmaInputType(OpBuilder &rewriter, Location loc, Value val,
                            Type targetTy) {
  auto valTy = val.getType();
  if (valTy == targetTy)
    return val;
  if (isa<FloatType>(valTy) && isa<FloatType>(targetTy)) {
    unsigned srcBits = cast<FloatType>(valTy).getWidth();
    unsigned dstBits = cast<FloatType>(targetTy).getWidth();
    if (srcBits < dstBits)
      return arith::ExtFOp::create(rewriter, loc, targetTy, val);
    else
      return arith::TruncFOp::create(rewriter, loc, targetTy, val);
  }
  // Integer type path: convert to f32 first, then truncate if needed
  auto f32Ty = Float32Type::get(rewriter.getContext());
  Value f32Val = arith::SIToFPOp::create(rewriter, loc, f32Ty, val);
  if (targetTy == f32Ty)
    return f32Val;
  return arith::TruncFOp::create(rewriter, loc, targetTy, f32Val);
}

// Convert a value to f32. Handles both float and integer element types.
static Value toF32(OpBuilder &rewriter, Location loc, Value val, Type f32Ty) {
  auto valTy = val.getType();
  if (valTy == f32Ty)
    return val;
  if (isa<FloatType>(valTy))
    return arith::ExtFOp::create(rewriter, loc, f32Ty, val);
  // Integer type (e.g. i8, i16, i32) -- use signed conversion
  return arith::SIToFPOp::create(rewriter, loc, f32Ty, val);
}

// Convert f32 to the target element type. Handles both float and integer types.
static Value fromF32(OpBuilder &rewriter, Location loc, Value val,
                     Type targetTy) {
  if (val.getType() == targetTy)
    return val;
  if (isa<FloatType>(targetTy))
    return arith::TruncFOp::create(rewriter, loc, targetTy, val);
  // Integer type -- use signed conversion
  return arith::FPToSIOp::create(rewriter, loc, targetTy, val);
}

static unsigned &getDotCounter(MLIRContext *ctx) {
  static llvm::DenseMap<MLIRContext *, unsigned> counters;
  return counters[ctx];
}

// ============================================================================
// DotOpBlockedConversion: blocked encoding on C, any rank (batch-aware).
// This is the original batch-aware lowering that handles 2D and 3D+ dots.
// ============================================================================
struct DotOpBlockedConversion : public ConvertOpToLLVMPattern<tt::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tt::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto cType = cast<RankedTensorType>(op.getC().getType());
    auto cEnc = dyn_cast<ttg::BlockedEncodingAttr>(cType.getEncoding());
    // Only handle blocked encoding on C. MMA encoding handled separately.
    if (!cEnc)
      return failure();

    auto aType = cast<RankedTensorType>(op.getA().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());

    // ── Extract M, N, K from last two dimensions ────────────────────
    unsigned rank = cType.getRank();
    unsigned rowDim = rank - 2;
    unsigned colDim = rank - 1;

    int64_t M = cType.getShape()[rowDim];
    int64_t N = cType.getShape()[colDim];
    int64_t K = aType.getShape()[colDim]; // A is [..., M, K]

    // Compute batch size (product of all dims except last 2)
    int64_t batchSize = 1;
    for (unsigned d = 0; d < rowDim; ++d)
      batchSize *= cType.getShape()[d];

    auto f32Ty = Float32Type::get(ctx);
    auto tgPtrTy = LLVMPointerType::get(ctx, 3);
    auto matTy = getSimdgroupMatrixType(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);

    // ── Declare air intrinsics ────────────────────────────────────────

    auto laneIdFn =
        getOrInsertIntrinsic(rewriter, mod, "air.thread_index_in_simdgroup",
                             LLVMFunctionType::get(i32Ty, {}, false));

    auto voidTy = LLVMVoidType::get(ctx);
    auto barrTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
    auto tgBarrFn =
        getOrInsertIntrinsic(rewriter, mod, "air.threadgroup.barrier", barrTy);
    (void)getOrInsertIntrinsic(rewriter, mod, "air.simdgroup.barrier", barrTy);

    auto vec2i64Ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
    auto i1Ty = IntegerType::get(ctx, 1);
    bool canonSG = useCanonicalSimdgroupSig();
    // Build the simdgroup load/store arg-type lists for the selected target
    // ABI.
    auto sgLoadArgTys = [&](Type ptrTy) -> SmallVector<Type> {
      if (canonSG)
        return {ptrTy, i64Ty, vec2i64Ty, i1Ty};
      return {ptrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty};
    };
    auto sgStoreArgTys = [&](Type matVecTy, Type ptrTy) -> SmallVector<Type> {
      if (canonSG)
        return {matVecTy, ptrTy, i64Ty, vec2i64Ty, i1Ty};
      return {matVecTy, ptrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty};
    };
    // f32 TG load/store/MMA (always needed for C accumulator)
    auto loadFn = getOrInsertIntrinsic(
        rewriter, mod, "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
        LLVMFunctionType::get(matTy, sgLoadArgTys(tgPtrTy), false));
    auto storeFn = getOrInsertIntrinsic(
        rewriter, mod, "air.simdgroup_matrix_8x8_store.v64f32.p3f32",
        LLVMFunctionType::get(voidTy, sgStoreArgTys(matTy, tgPtrTy), false));

    // Type-specific A/B MMA intrinsics (bf16/f16 use native MMA).
    // For batchSize > 1, fall back to f32 to avoid batch offset mismatch
    // between scatter (element-indexed) and MMA load (byte-addressed).
    auto aElemTy = aType.getElementType();
    bool useNativeABType =
        (batchSize == 1) && (aElemTy.isF16() || aElemTy.isBF16());
    auto abMmaInfo = useNativeABType ? getMMAIntrinsicInfo(ctx, aElemTy)
                                     : getMMAIntrinsicInfo(ctx, f32Ty);
    auto abLoadFn = getOrInsertIntrinsic(
        rewriter, mod, abMmaInfo.tgLoadName,
        LLVMFunctionType::get(abMmaInfo.matVecTy, sgLoadArgTys(tgPtrTy),
                              false));
    auto abMmaFn = getOrInsertIntrinsic(
        rewriter, mod, abMmaInfo.mmaName,
        LLVMFunctionType::get(
            matTy, {abMmaInfo.matVecTy, abMmaInfo.matVecTy, matTy}, false));
    // Determine TG element type for A/B scatter.
    Type abTgElemTy = useNativeABType ? aElemTy : f32Ty;

    // simdgroup load/store emitters: pick the canonical (macOS<=15) or the
    // 3-vector (macOS>=16) argument list at runtime.  shapeDim is the first
    // lane of the 3-vector shape; pitch is the row pitch (3-vector stride lane1
    // / canonical elements_per_row).
    auto emitSGLoad = [&](LLVMFuncOp fn, Value ptr, int64_t shapeDim,
                          int64_t pitch, Value off) -> Value {
      SmallVector<Value> args;
      if (canonSG)
        args = {ptr, makeI64(rewriter, loc, pitch), off,
                makeI1False(rewriter, loc)};
      else
        args = {ptr, makeI64Vec2(rewriter, loc, shapeDim, 8),
                makeI64Vec2(rewriter, loc, 1, pitch), off};
      return LLVM::CallOp::create(rewriter, loc, fn, args).getResult();
    };
    auto emitSGStore = [&](LLVMFuncOp fn, Value mat, Value ptr,
                           int64_t shapeDim, int64_t pitch, Value off) {
      SmallVector<Value> args;
      if (canonSG)
        args = {mat, ptr, makeI64(rewriter, loc, pitch), off,
                makeI1False(rewriter, loc)};
      else
        args = {mat, ptr, makeI64Vec2(rewriter, loc, shapeDim, 8),
                makeI64Vec2(rewriter, loc, 1, pitch), off};
      LLVM::CallOp::create(rewriter, loc, fn, args);
    };

    // ── Constants ────────────────────────────────────────────────────

    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

    // ── Thread identification ─────────────────────────────────────────

    Value laneId =
        LLVM::CallOp::create(rewriter, loc, laneIdFn, ValueRange{}).getResult();

    auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto tidFn = getOrInsertIntrinsic(
        rewriter, mod, "air.thread_position_in_threadgroup",
        LLVMFunctionType::get(arrI32x3Ty, {}, false));
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                               ArrayRef<int64_t>{0});
    Value warpId = divByConst(rewriter, loc, tid32, 32); // tid / 32

    // ── Get blocked encoding params for A, B, C ───────────────────────

    // Unpack struct elements
    auto unpack = [&](Value v) -> SmallVector<Value> {
      SmallVector<Value> elems;
      if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
        for (unsigned i = 0; i < sTy.getBody().size(); ++i)
          elems.push_back(
              ExtractValueOp::create(rewriter, loc, sTy.getBody()[i], v,
                                     ArrayRef<int64_t>{(int64_t)i}));
      } else {
        elems = {v};
      }
      return elems;
    };

    // resolveOperand returns: (elements, offsets, encoding, dotOpIdx)
    // dotOpIdx: -1 if not from DotOperandEncoding, 0 for A-matrix, 1 for
    // B-matrix
    auto resolveOperand = [&](Value tritonVal, Value adaptorVal,
                              RankedTensorType opTy)
        -> std::tuple<SmallVector<Value>, SmallVector<SmallVector<unsigned>>,
                      ttg::BlockedEncodingAttr, int> {
      // Path 1: convert_layout -- look through to source blocked values
      if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>()) {
        Value mapped = rewriter.getRemappedValue(cvt.getSrc());
        if (mapped) {
          auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
          auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
          if (srcEnc) {
            auto offsets = emitOffsetForLayout(srcEnc, srcTy);
            return {unpack(mapped), offsets, srcEnc, -1};
          }
        }
      }
      // Path 2: DotOperandEncoding (e.g. local_load after
      // optimize_dot_operands)
      auto enc = opTy.getEncoding();
      if (auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(enc)) {
        auto parentEnc = dyn_cast<ttg::BlockedEncodingAttr>(dotEnc.getParent());
        if (parentEnc) {
          auto offsets = emitOffsetForLayout(enc, opTy);
          return {unpack(adaptorVal), offsets, parentEnc,
                  (int)dotEnc.getOpIdx()};
        }
      }
      // Path 3: direct blocked encoding
      if (auto blk = dyn_cast<ttg::BlockedEncodingAttr>(enc)) {
        auto offsets = emitOffsetForLayout(blk, opTy);
        return {unpack(adaptorVal), offsets, blk, -1};
      }
      return {{}, {}, nullptr, -1};
    };

    auto [elemsA, aOffsets, aSrcEnc, aDotOpIdx] =
        resolveOperand(op.getA(), adaptor.getA(), aType);
    auto [elemsB, bOffsets, bSrcEnc, bDotOpIdx] =
        resolveOperand(op.getB(), adaptor.getB(), bType);
    auto elemsC = unpack(adaptor.getC());
    auto cOffsets = emitOffsetForLayout(cEnc, cType);

    if (!aSrcEnc || !bSrcEnc)
      return failure();

    // Verify element counts match
    if ((int64_t)elemsA.size() != (int64_t)aOffsets.size() ||
        (int64_t)elemsB.size() != (int64_t)bOffsets.size() ||
        (int64_t)elemsC.size() != (int64_t)cOffsets.size())
      return failure();

    // ── Try to resolve device pointers for async copy ──────────────
    auto resolveDevicePointers = [&](Value tritonVal) -> SmallVector<Value> {
      Value src = tritonVal;
      if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      auto loadOp = src.getDefiningOp<tt::LoadOp>();
      if (!loadOp)
        return {};
      Value ptrTensor = loadOp.getPtr();
      Value mappedPtrs = rewriter.getRemappedValue(ptrTensor);
      if (!mappedPtrs)
        return {};
      return unpack(mappedPtrs);
    };

    // ── Resolve device pointers for async copy (A and B) ──────────────
    auto aPtrs = resolveDevicePointers(op.getA());
    auto bPtrs = resolveDevicePointers(op.getB());

    // Compute row stride from pointer differences for async copy.
    // This uses the same shuffle-based approach as the MMA conversion.
    auto computeRowStrideBlocked =
        [&](SmallVector<Value> &ptrs,
            SmallVector<SmallVector<unsigned>> &offsets, Type elemTy,
            ttg::BlockedEncodingAttr enc) -> Value {
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      // Strategy 1: same-thread pair with same col, different row
      for (size_t i = 0; i < offsets.size(); ++i) {
        for (size_t j = i + 1; j < offsets.size(); ++j) {
          if (offsets[i][colDim] == offsets[j][colDim] &&
              offsets[i][rowDim] != offsets[j][rowDim]) {
            int64_t rowDiff =
                (int64_t)offsets[j][rowDim] - (int64_t)offsets[i][rowDim];
            Value ptrI64_i =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[i]);
            Value ptrI64_j =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[j]);
            Value byteDiff =
                arith::SubIOp::create(rewriter, loc, ptrI64_j, ptrI64_i);
            Value elemSize = arith::ConstantIntOp::create(
                rewriter, loc, (int64_t)elemBytes, 64);
            Value elemDiff =
                arith::DivSIOp::create(rewriter, loc, byteDiff, elemSize);
            Value rowDiffVal =
                arith::ConstantIntOp::create(rewriter, loc, rowDiff, 64);
            return arith::DivSIOp::create(rewriter, loc, elemDiff, rowDiffVal);
          }
        }
      }
      // Strategy 2: simd_shuffle_xor
      auto order = enc.getOrder();
      auto tpw = enc.getThreadsPerWarp();
      auto spt = enc.getSizePerThread();
      unsigned encRank = spt.size();
      unsigned encColDim = encRank - 1;
      unsigned encRowDim = encRank - 2;
      bool colFastest = (order[0] == (unsigned)encColDim);
      int64_t xorMask = colFastest ? tpw[encColDim] : 1;
      int64_t rowDiff = spt[encRowDim];

      Value ptrI64 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[0]);
      auto i16Ty = IntegerType::get(ctx, 16);
      Value xorVal = arith::ConstantIntOp::create(rewriter, loc, xorMask, 16);
      Value lo = arith::TruncIOp::create(rewriter, loc, i32Ty, ptrI64);
      Value hi = arith::TruncIOp::create(
          rewriter, loc, i32Ty,
          arith::ShRUIOp::create(
              rewriter, loc, ptrI64,
              arith::ConstantIntOp::create(rewriter, loc, 32, 64)));
      auto shuffleFn32 = getOrInsertIntrinsic(
          rewriter, mod, "air.simd_shuffle_xor.s.i32",
          LLVMFunctionType::get(i32Ty, {i32Ty, i16Ty}, false));
      Value shufLo = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{lo, xorVal})
                         .getResult();
      Value shufHi = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{hi, xorVal})
                         .getResult();
      Value loExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufLo);
      Value hiExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufHi);
      Value hiShl = arith::ShLIOp::create(
          rewriter, loc, hiExt,
          arith::ConstantIntOp::create(rewriter, loc, 32, 64));
      Value otherPtrI64 = arith::OrIOp::create(rewriter, loc, loExt, hiShl);
      Value byteDiff =
          arith::SubIOp::create(rewriter, loc, otherPtrI64, ptrI64);
      Value zero64 =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
      Value isNeg = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, byteDiff, zero64);
      Value negDiff = arith::SubIOp::create(rewriter, loc, zero64, byteDiff);
      Value absDiff =
          arith::SelectOp::create(rewriter, loc, isNeg, negDiff, byteDiff);
      Value elemSize =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
      Value elemDiff = arith::DivUIOp::create(rewriter, loc, absDiff, elemSize);
      if (rowDiff > 1) {
        Value rowDiffVal =
            arith::ConstantIntOp::create(rewriter, loc, rowDiff, 64);
        elemDiff = arith::DivUIOp::create(rewriter, loc, elemDiff, rowDiffVal);
      }
      return elemDiff;
    };

    // Compute column stride: distance between adjacent columns in same row.
    auto computeColStrideBlocked =
        [&](SmallVector<Value> &ptrs,
            SmallVector<SmallVector<unsigned>> &offsets, Type elemTy,
            ttg::BlockedEncodingAttr enc) -> Value {
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      // Strategy 1: same-thread pair with same row, different col
      for (size_t i = 0; i < offsets.size(); ++i) {
        for (size_t j = i + 1; j < offsets.size(); ++j) {
          if (offsets[i][rowDim] == offsets[j][rowDim] &&
              offsets[i][colDim] != offsets[j][colDim]) {
            int64_t colDiff =
                (int64_t)offsets[j][colDim] - (int64_t)offsets[i][colDim];
            Value ptrI64_i =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[i]);
            Value ptrI64_j =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[j]);
            Value byteDiff =
                arith::SubIOp::create(rewriter, loc, ptrI64_j, ptrI64_i);
            Value elemSz = arith::ConstantIntOp::create(rewriter, loc,
                                                        (int64_t)elemBytes, 64);
            Value elemDiff =
                arith::DivSIOp::create(rewriter, loc, byteDiff, elemSz);
            Value colDiffVal =
                arith::ConstantIntOp::create(rewriter, loc, colDiff, 64);
            return arith::DivSIOp::create(rewriter, loc, elemDiff, colDiffVal);
          }
        }
      }
      // Strategy 2: simd_shuffle_xor to get col-adjacent thread
      auto order = enc.getOrder();
      auto tpw = enc.getThreadsPerWarp();
      auto spt = enc.getSizePerThread();
      unsigned encRank = spt.size();
      unsigned encColDim = encRank - 1;
      unsigned encRowDim = encRank - 2;
      bool colFastest = (order[0] == (unsigned)encColDim);
      int64_t xorMask = colFastest ? 1 : tpw[encRowDim];
      int64_t colDiff = spt[encColDim];

      Value ptrI64 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[0]);
      auto i16Ty = IntegerType::get(ctx, 16);
      Value xorVal = arith::ConstantIntOp::create(rewriter, loc, xorMask, 16);
      Value lo = arith::TruncIOp::create(rewriter, loc, i32Ty, ptrI64);
      Value hi = arith::TruncIOp::create(
          rewriter, loc, i32Ty,
          arith::ShRUIOp::create(
              rewriter, loc, ptrI64,
              arith::ConstantIntOp::create(rewriter, loc, 32, 64)));
      auto shuffleFn32 = getOrInsertIntrinsic(
          rewriter, mod, "air.simd_shuffle_xor.s.i32",
          LLVMFunctionType::get(i32Ty, {i32Ty, i16Ty}, false));
      Value shufLo = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{lo, xorVal})
                         .getResult();
      Value shufHi = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{hi, xorVal})
                         .getResult();
      Value loExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufLo);
      Value hiExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufHi);
      Value hiShl = arith::ShLIOp::create(
          rewriter, loc, hiExt,
          arith::ConstantIntOp::create(rewriter, loc, 32, 64));
      Value otherPtrI64 = arith::OrIOp::create(rewriter, loc, loExt, hiShl);
      Value byteDiff =
          arith::SubIOp::create(rewriter, loc, otherPtrI64, ptrI64);
      Value zero64 =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
      Value isNeg = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, byteDiff, zero64);
      Value negDiff = arith::SubIOp::create(rewriter, loc, zero64, byteDiff);
      Value absDiff =
          arith::SelectOp::create(rewriter, loc, isNeg, negDiff, byteDiff);
      Value elemSz =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
      Value elemDiff = arith::DivUIOp::create(rewriter, loc, absDiff, elemSz);
      if (colDiff > 1) {
        Value colDiffVal =
            arith::ConstantIntOp::create(rewriter, loc, colDiff, 64);
        elemDiff = arith::DivUIOp::create(rewriter, loc, elemDiff, colDiffVal);
      }
      return elemDiff;
    };

    // Determine if async copy is usable for A/B (only for non-batched).
    // Async copy requires row-major layout (contiguous columns) because the
    // DMA copies contiguous bytes per row. Column-major data has non-contiguous
    // columns, which async_copy_2d can't handle.
    // Heuristic: if encoding's col dim is fastest-varying (order[0]==colDim),
    // data is likely row-major.
    auto isRowMajorEncoding = [&](ttg::BlockedEncodingAttr enc) -> bool {
      auto order = enc.getOrder();
      unsigned encColDim = enc.getSizePerThread().size() - 1;
      return order[0] == (unsigned)encColDim;
    };

    // Compute matWarpsC early: number of warps assigned to the MxN tile.
    // When matWarpsC > 1, multiple simdgroups share the same TG strip,
    // causing concurrent async copies to race on the same TG region.
    auto cWpcEarly = cEnc.getWarpsPerCTA();
    int64_t matWarpsCEarly = cWpcEarly[rowDim] * cWpcEarly[colDim];

    // Env var to disable async copy for debugging
    bool asyncCopyEnabled = true;
    {
      const char *envVar = std::getenv("TRITON_ASYNC_COPY");
      if (envVar && std::string(envVar) == "0")
        asyncCopyEnabled = false;
    }
    // Async copy writes raw bytes to TG, but MMA loads read as f32.
    // For integer types (int8 etc.), the byte width mismatch corrupts data.
    if (isa<IntegerType>(aElemTy))
      asyncCopyEnabled = false;
    // When multiple simdgroups handle the same MxN tile (matWarpsC > 1),
    // all issue identical async copies to the same TG region concurrently.
    // This races on Apple GPUs and corrupts data. Fall back to scatter.
    if (matWarpsCEarly > 1)
      asyncCopyEnabled = false;

    bool useAsyncA = false, useAsyncB = false;
    Value aRowStride, bRowStride, aColStride, bColStride;
    if (asyncCopyEnabled && batchSize == 1 &&
        (int64_t)aPtrs.size() == (int64_t)elemsA.size() && !aPtrs.empty() &&
        isRowMajorEncoding(aSrcEnc)) {
      aRowStride = computeRowStrideBlocked(aPtrs, aOffsets,
                                           aType.getElementType(), aSrcEnc);
      aColStride = computeColStrideBlocked(aPtrs, aOffsets,
                                           aType.getElementType(), aSrcEnc);
      useAsyncA = (aRowStride != nullptr && aColStride != nullptr);
    }
    if (asyncCopyEnabled && batchSize == 1 &&
        (int64_t)bPtrs.size() == (int64_t)elemsB.size() && !bPtrs.empty() &&
        isRowMajorEncoding(bSrcEnc)) {
      bRowStride = computeRowStrideBlocked(bPtrs, bOffsets,
                                           bType.getElementType(), bSrcEnc);
      bColStride = computeColStrideBlocked(bPtrs, bOffsets,
                                           bType.getElementType(), bSrcEnc);
      useAsyncB = (bRowStride != nullptr && bColStride != nullptr);
    }

    // ── Compute runtime thread base position ──────────────────────────
    // For 3D+ tensors, use only the last two dims of the encoding for
    // the MMA row/col base. The batch dims are handled via compile-time
    // offset matching.
    auto makeBase = [&](ttg::BlockedEncodingAttr enc, int64_t rows,
                        int64_t cols) -> std::pair<Value, Value> {
      auto spt = enc.getSizePerThread();
      auto tpw = enc.getThreadsPerWarp();
      auto wpc = enc.getWarpsPerCTA();
      auto order = enc.getOrder();

      unsigned encRank = spt.size();
      unsigned encRowDim = encRank - 2;
      unsigned encColDim = encRank - 1;

      int64_t sM = spt[encRowDim], sN = spt[encColDim];
      int64_t tM = tpw[encRowDim], tN = tpw[encColDim];
      int64_t wM = wpc[encRowDim], wN = wpc[encColDim];
      int64_t tileM = wM * tM * sM;
      int64_t tileN = wN * tN * sN;

      // For 3D+ encodings, strip batch warp component from warpId.
      // Batch dims use warps for batch distribution; the 2D row/col
      // decomposition should only use the row/col warps.
      int64_t batchWarps = 1;
      for (unsigned d = 0; d < encRowDim; ++d)
        batchWarps *= wpc[d];
      Value matWarpId = warpId;
      if (batchWarps > 1) {
        // mat_warp = warpId % (wM * wN)
        int64_t matWarps = wM * wN;
        matWarpId = remByConst(rewriter, loc, warpId, matWarps);
      }

      // Similarly strip batch lanes from laneId
      int64_t batchLanes = 1;
      for (unsigned d = 0; d < encRowDim; ++d)
        batchLanes *= tpw[d];
      Value matLaneId = laneId;
      if (batchLanes > 1) {
        int64_t matLanes = tM * tN;
        matLaneId = remByConst(rewriter, loc, laneId, matLanes);
      }

      // For the last two dims, check if col is the fastest-varying dim.
      // order[0] is the fastest dim index. For 2D: order[0]==1 means col-fast.
      // For 3D with order=[2,1,0]: order[0]==2 means colDim is fastest.
      bool colFastest = (order[0] == (unsigned)encColDim);

      Value tMsM = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
      Value sM_val = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
      Value tNsN = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
      Value sN_val = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

      // Warp decomposition: Morton order when both dims are power-of-2,
      // otherwise linear div/mod.
      Value wR, wC;
      unsigned mortonBits = mortonBitsPerDim(wM, wN);
      if (mortonBits > 0) {
        // Morton Z-order: deinterleave even bits → row, odd bits → col
        // For colFastest: even bits = row, odd bits = col
        // For rowFastest: even bits = col, odd bits = row
        if (colFastest) {
          wR = mortonDeinterleaveEven(rewriter, loc, matWarpId, mortonBits);
          wC = mortonDeinterleaveOdd(rewriter, loc, matWarpId, mortonBits);
        } else {
          wC = mortonDeinterleaveEven(rewriter, loc, matWarpId, mortonBits);
          wR = mortonDeinterleaveOdd(rewriter, loc, matWarpId, mortonBits);
        }
        // Mask to valid range if grid is non-square
        if (wM < (1LL << mortonBits))
          wR = remByConst(rewriter, loc, wR, wM);
        if (wN < (1LL << mortonBits))
          wC = remByConst(rewriter, loc, wC, wN);
      } else if (colFastest) {
        wR = divByConst(rewriter, loc, matWarpId, wN);
        wC = remByConst(rewriter, loc, matWarpId, wN);
      } else {
        wR = remByConst(rewriter, loc, matWarpId, wM);
        wC = divByConst(rewriter, loc, matWarpId, wM);
      }
      // Lane decomposition: shift/mask when power-of-2
      Value lR, lC;
      if (colFastest) {
        lR = divByConst(rewriter, loc, matLaneId, tN);
        lC = remByConst(rewriter, loc, matLaneId, tN);
      } else {
        lR = remByConst(rewriter, loc, matLaneId, tM);
        lC = divByConst(rewriter, loc, matLaneId, tM);
      }

      Value baseRow = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wR, tMsM),
          arith::MulIOp::create(rewriter, loc, lR, sM_val));
      Value baseCol = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wC, tNsN),
          arith::MulIOp::create(rewriter, loc, lC, sN_val));

      // Wrap to handle redundant threads (tileM > rows)
      if (tileM > rows)
        baseRow = remByConst(rewriter, loc, baseRow, rows);
      if (tileN > cols)
        baseCol = remByConst(rewriter, loc, baseCol, cols);

      return {baseRow, baseCol};
    };

    auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
    auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
    auto [cBaseRow, cBaseCol] = makeBase(cEnc, M, N);

    // For DotOperandEncoding (Path 2), the contracting dimension (K)
    // is fully replicated per thread. The offsets from emitOffsetForLayout
    // already span the full K range, so the base for the K dim must be 0.
    // Otherwise base + offset overshoots the K dimension.
    Value zero32 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    if (aDotOpIdx == 0) {
      // A is [M, K]: col dim is K (contracting) -> zero base
      aBaseCol = zero32;
    }
    if (bDotOpIdx == 1) {
      // B is [K, N]: row dim is K (contracting) -> zero base
      bBaseRow = zero32;
    }

    // ── Create threadgroup global ─────────────────────────────────────
    unsigned id = getDotCounter(ctx)++;
    // Padded strides for bank conflict avoidance.
    // Only pad when the padded buffer (including batch slices) fits in 16KB,
    // leaving headroom for other TG allocations within the 32KB hardware limit.
    int64_t pad = tgPadForType(aElemTy);
    int64_t maxStride = std::max(K, N);
    int64_t unpaddedSize = 8 * maxStride * batchSize + 1;
    int64_t paddedMaxStride = maxStride + pad;
    int64_t paddedSize = 8 * paddedMaxStride * batchSize + 1;
    bool canPad = (pad > 0) && (paddedSize * 4 <= 16384); // 16KB budget
    int64_t Kpad = canPad ? K + pad : K;
    int64_t Npad = canPad ? N + pad : N;
    int64_t tgStripStride = canPad ? paddedMaxStride : maxStride;
    int64_t tgStripSize = 8 * tgStripStride;

    // Double-buffering for B strips in Phase 3: overlap DMA with MMA.
    // Only when async copy is used for B, K > 8 (multiple strips), and
    // the doubled TG buffer fits in 16KB budget.
    // batchSize is always 1 when async copy is enabled (checked above).
    int64_t bStripBytes = 8 * Npad * 4; // one B strip in bytes (f32)
    int64_t tilesKEarly = K / 8;
    // Also require matWarpsCEarly > 1: single-warp configs can't overlap
    // load+compute (one simdgroup), and the doubled TG global gets trimmed
    // by the IR pipeline's dead-GEP elimination, causing PSO crashes.
    bool useDoubleBufB = useAsyncB && (tilesKEarly > 1) &&
                         (2 * bStripBytes <= 16384) && (batchSize == 1) &&
                         (matWarpsCEarly > 1);
    // Env var to disable double-buffering for debugging
    {
      const char *envVar = std::getenv("TRITON_DOUBLE_BUF");
      if (envVar && std::string(envVar) == "0")
        useDoubleBufB = false;
    }

    // Each batch slice needs its own TG region so MMA ops don't
    // cross-contaminate between warps assigned to different batches.
    // With double-buffering, Phase 3 needs 2 B strip slots.
    // The TG buffer must fit the max of: Phase 1 (A strip), Phase 2 (C strip),
    // Phase 3 (1 or 2 B strips), Phase 4 (C strip).
    int64_t phase3Strips = useDoubleBufB ? 2 : 1;
    int64_t tgSizeNeeded = std::max(tgStripSize, phase3Strips * 8 * Npad);
    int64_t tgSize = tgSizeNeeded * batchSize + 1; // +1 garbage slot
    auto tgBuf = getOrCreateTGGlobal(
        rewriter, mod, ("__tg_dot_ab_" + llvm::Twine(id)).str(), tgSize);

    Value ptrTG =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgBuf.getName());

    // ── Async copy intrinsics (when device pointers available) ────────
    auto devPtrTy = LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVMPointerType::get(ctx, 0);
    LLVMFuncOp asyncCopyFn, waitFn;
    Value evAlloca;
    if (useAsyncA || useAsyncB) {
      auto asyncCopyFnTy = LLVMFunctionType::get(
          tgPtrTy,
          {i64Ty, i64Ty, tgPtrTy, i64Ty, i64Ty, vec2i64Ty, devPtrTy, i64Ty,
           i64Ty, vec2i64Ty, vec2i64Ty, i32Ty},
          false);
      asyncCopyFn = getOrInsertIntrinsic(
          rewriter, mod, "air.simdgroup_async_copy_2d.p3i8.p1i8",
          asyncCopyFnTy);

      auto waitFnTy = LLVMFunctionType::get(voidTy, {i32Ty, ptrTy0}, false);
      waitFn = getOrInsertIntrinsic(rewriter, mod, "air.wait_simdgroup_events",
                                    waitFnTy);

      // Event alloca: thread-local storage for async event pointer
      auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
      OpBuilder::InsertionGuard guard(rewriter);
      if (funcOp) {
        // Check for existing alloca
        auto &entryBlock = funcOp.getBody().front();
        bool found = false;
        for (auto &existingOp : entryBlock) {
          if (auto alloca = dyn_cast<LLVM::AllocaOp>(existingOp)) {
            if (alloca.getElemType() == tgPtrTy) {
              evAlloca = alloca.getResult();
              found = true;
              break;
            }
          }
        }
        if (!found) {
          rewriter.setInsertionPointToStart(&entryBlock);
          Value one64 = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
          evAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, tgPtrTy,
                                            one64, /*alignment=*/8)
                         .getResult();
        }
      }
    }

    // Compute device strip base pointer (uniform across all threads).
    // For row-major: addr = base + row * rowStride + col * colStride
    // stripBase = ptrs[0] - ((baseRow + offsets[0][rowDim] - stripRowStart) *
    // rowStride
    //                        + (baseCol + offsets[0][colDim]) * colStride)
    // The baseRow/baseCol terms cancel with ptrs[0]'s offset, giving:
    //   matrixBase + stripRowStart * rowStride (same for all threads).
    auto computeStripDevPtr = [&](SmallVector<Value> &ptrs,
                                  SmallVector<SmallVector<unsigned>> &offsets,
                                  Value baseRow, Value baseCol, Value rowStride,
                                  Value colStride, int64_t stripRowStart,
                                  Type elemTy) -> Value {
      int64_t refRowOff = offsets[0][rowDim];
      int64_t refColOff = offsets[0][colDim];

      Value baseRowExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseRow);
      Value baseColExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseCol);

      // rowDelta = baseRow + refRowOff - stripRowStart
      Value rowDelta = arith::AddIOp::create(
          rewriter, loc, baseRowExt,
          arith::ConstantIntOp::create(rewriter, loc, refRowOff - stripRowStart,
                                       64));
      // colDelta = baseCol + refColOff
      Value colDelta = arith::AddIOp::create(
          rewriter, loc, baseColExt,
          arith::ConstantIntOp::create(rewriter, loc, refColOff, 64));
      // elemOff = rowDelta * rowStride + colDelta * colStride
      Value elemOff = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(rewriter, loc, rowDelta, rowStride),
          arith::MulIOp::create(rewriter, loc, colDelta, colStride));
      // Negate to go back to strip base
      Value negElemOff = arith::SubIOp::create(
          rewriter, loc,
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64), elemOff);
      return LLVM::GEPOp::create(rewriter, loc, devPtrTy, elemTy, ptrs[0],
                                 ArrayRef<LLVM::GEPArg>{negElemOff});
    };

    // Emit async copy from device to TG for one 8-row strip (fire only).
    // Returns without waiting -- caller must call emitAsyncCopyWait.
    // tgPadStride: the padded stride used in TG (stripCols + padding).
    auto emitAsyncCopyFire = [&](Value stripDevPtr, Value rowStrideElems,
                                 Value tgDst, int64_t stripCols,
                                 int64_t tgPadStride, Type elemTy) {
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      int64_t tileWidthBytes = stripCols * elemBytes;

      Value sizeOf = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value alignOf = arith::ConstantIntOp::create(rewriter, loc, 1, 64);

      // Destination (TG): packed with padding
      Value dstStrideBytes = arith::ConstantIntOp::create(
          rewriter, loc, tgPadStride * elemBytes, 64);
      Value dstElemStride = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value dstTile = makeI64Vec2(rewriter, loc, tileWidthBytes, 8);

      // Source (device): stride in bytes
      Value elemBytesVal =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
      Value srcStrideBytes =
          arith::MulIOp::create(rewriter, loc, rowStrideElems, elemBytesVal);
      Value srcElemStride = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value srcTile = makeI64Vec2(rewriter, loc, tileWidthBytes, 8);

      Value offsetVec = makeI64Vec2(rewriter, loc, 0, 0);
      Value clamp = arith::ConstantIntOp::create(rewriter, loc, 0, 32);

      Value evPtr =
          LLVM::CallOp::create(
              rewriter, loc, asyncCopyFn,
              ValueRange{sizeOf, alignOf, tgDst, dstStrideBytes, dstElemStride,
                         dstTile, stripDevPtr, srcStrideBytes, srcElemStride,
                         srcTile, offsetVec, clamp})
              .getResult();

      // Store event pointer for later wait
      LLVM::StoreOp::create(rewriter, loc, evPtr, evAlloca);
    };

    // Wait for the last fired async copy to complete.
    auto emitAsyncCopyWait = [&]() {
      Value oneI32 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      LLVM::CallOp::create(rewriter, loc, waitFn, ValueRange{oneI32, evAlloca});
    };

    // Emit async copy from device to TG for one 8-row strip (fire + wait).
    auto emitAsyncCopy = [&](Value stripDevPtr, Value rowStrideElems,
                             Value tgDst, int64_t stripCols,
                             int64_t tgPadStride, Type elemTy) {
      emitAsyncCopyFire(stripDevPtr, rowStrideElems, tgDst, stripCols,
                        tgPadStride, elemTy);
      emitAsyncCopyWait();
    };

    // Compute runtime batch-offset pointer for SIMD matrix load/store.
    // Each batch slice gets its own tgStripSize region in TG memory so
    // MMA ops don't cross-contaminate between warps assigned to different
    // batches.
    //
    // numBatchWarps = numTotalWarps / matWarpsC.
    // When batchSize > numBatchWarps, we must process multiple batches
    // per warp via an unrolled batch loop (see below).
    auto cWpc = cEnc.getWarpsPerCTA();
    int64_t matWarpsC = cWpc[rowDim] * cWpc[colDim];
    int64_t numTotalWarps = 1;
    for (auto w : cWpc)
      numTotalWarps *= w;
    int64_t numBatchWarps = numTotalWarps / matWarpsC;

    // Compute per-operand batch warp index (runtime).
    // batchWarpIdx = warpId / matWarps, where matWarps is the product
    // of warpsPerCTA for the row and col dims of each operand's encoding.
    // This determines which batch slice each warp handles.
    auto makeBatchWarpIdx = [&](ttg::BlockedEncodingAttr enc) -> Value {
      auto wpc = enc.getWarpsPerCTA();
      unsigned encRank = wpc.size();
      unsigned encRowDim = encRank - 2;
      unsigned encColDim = encRank - 1;
      int64_t matW = wpc[encRowDim] * wpc[encColDim];
      int64_t batchW = 1;
      for (unsigned d = 0; d < encRowDim; ++d)
        batchW *= wpc[d];
      if (batchW <= 1)
        return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      return divByConst(rewriter, loc, warpId, matW);
    };

    Value cBatchWarpIdx = makeBatchWarpIdx(cEnc);
    Value aBatchWarpIdx = makeBatchWarpIdx(aSrcEnc);
    Value bBatchWarpIdx = makeBatchWarpIdx(bSrcEnc);

    Value ptrTGBatch = ptrTG;
    Value batchTGOffset64 =
        arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
    if (batchSize > 1 && numBatchWarps >= batchSize) {
      // Warp-distributed: each warp handles one batch via C's batchWarpIdx.
      Value batchOff32 = arith::MulIOp::create(
          rewriter, loc, cBatchWarpIdx,
          arith::ConstantIntOp::create(rewriter, loc, tgStripSize, 32));
      batchTGOffset64 =
          arith::ExtUIOp::create(rewriter, loc, i64Ty, batchOff32);
      ptrTGBatch = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty, ptrTG,
                                       ArrayRef<LLVM::GEPArg>{batchTGOffset64});
    }

    // ── GEP helpers ───────────────────────────────────────────────────

    auto gather1 = [&](Value ptr, Value flatIdx64) -> Value {
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty, ptr,
                                      ArrayRef<LLVM::GEPArg>{flatIdx64});
      return LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult();
    };

    // stripFlatIdx: (baseRow + rowOff - stripRowStart) * stride + (baseCol +
    // colOff)
    auto stripFlatIdx = [&](Value baseRow, Value baseCol, int64_t rowOff,
                            int64_t colOff, int64_t stride,
                            int64_t stripRowStart) -> Value {
      Value row32 =
          arith::AddIOp::create(rewriter, loc, baseRow,
                                arith::ConstantIntOp::create(
                                    rewriter, loc, rowOff - stripRowStart, 32));
      Value col32 = arith::AddIOp::create(
          rewriter, loc, baseCol,
          arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
      Value flat32 = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(
              rewriter, loc, row32,
              arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
          col32);
      return arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
    };

    int64_t tilesM = M / 8;
    int64_t tilesN = N / 8;
    int64_t tilesK = K / 8;

    // Garbage bin index -- last slot in TG, used for out-of-strip stores.
    // Points to the very last slot in the full (batch-expanded) TG buffer.
    Value garbageIdx = arith::ConstantIntOp::create(
        rewriter, loc, tgStripSize * batchSize, 64);

    // Determine if an operand has batch warps (runtime batch component).
    // If wpc[0] == 1, all batches are in compile-time offsets (no runtime
    // batch). If wpc[0] > 1, batch is partially runtime (batchWarpIdx
    // contributes).
    auto hasBatchWarps = [&](ttg::BlockedEncodingAttr enc) -> bool {
      if (rowDim == 0)
        return false;
      auto wpc = enc.getWarpsPerCTA();
      int64_t bw = 1;
      for (unsigned d = 0; d < rowDim; ++d)
        bw *= wpc[d];
      return bw > 1;
    };

    // "hasBatchWarps" means the operand needs runtime batch filtering
    // in sequential mode. If false, compile-time elemBatchIndex covers
    // all batches and compile-time filtering suffices.
    bool aHasBatchWarps = hasBatchWarps(aSrcEnc);
    bool bHasBatchWarps = hasBatchWarps(bSrcEnc);
    bool cHasBatchWarps = hasBatchWarps(cEnc);

    // For warp-distributed mode (TG offset routing), we still need to know
    // if offsets have mixed batch values for the TG offset computation.
    auto hasMixedBatches =
        [&](const SmallVector<SmallVector<unsigned>> &offsets) -> bool {
      if (rowDim == 0 || offsets.empty())
        return false;
      for (size_t i = 1; i < offsets.size(); ++i) {
        for (unsigned d = 0; d < rowDim; ++d) {
          if (offsets[i][d] != offsets[0][d])
            return true;
        }
      }
      return false;
    };

    bool aMixed = hasMixedBatches(aOffsets);
    bool bMixed = hasMixedBatches(bOffsets);
    bool cMixed = hasMixedBatches(cOffsets);

    // Helper: compute the TG batch offset for an element.
    // For operands with mixed batch offsets (e.g. dot_op A/B), use the
    // compile-time offset as the global batch index.
    // For operands with uniform batch offsets (e.g. C blocked), use the
    // runtime batchTGOffset64 derived from batchWarpIdx.
    auto elemBatchTGOffset =
        [&](const SmallVector<SmallVector<unsigned>> &offsets, size_t i,
            bool mixed) -> Value {
      if (rowDim == 0)
        return arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
      if (mixed) {
        // Compute flat batch index from compile-time offset dims
        int64_t batchOff = 0;
        int64_t stride = 1;
        for (int d = (int)rowDim - 1; d >= 0; --d) {
          batchOff += offsets[i][d] * stride;
          stride *= cType.getShape()[d];
        }
        return arith::ConstantIntOp::create(rewriter, loc,
                                            batchOff * tgStripSize, 64);
      }
      // Uniform batch -- use runtime warp-based offset
      return batchTGOffset64;
    };

    // Helper: compute flat batch index for element i from compile-time offsets.
    auto elemBatchIndex = [&](const SmallVector<SmallVector<unsigned>> &offsets,
                              size_t i) -> int64_t {
      int64_t batchIdx = 0;
      int64_t stride = 1;
      for (int d = (int)rowDim - 1; d >= 0; --d) {
        batchIdx += offsets[i][d] * stride;
        stride *= cType.getShape()[d];
      }
      return batchIdx;
    };

    // Helper: scatter elements into TG for an 8-row strip.
    // scatterTy: element type to scatter as (f32 for C, native type for A/B).
    // In sequential batch mode (curBatchRound >= 0), only scatter elements
    // matching the current batch. Data goes to TG base (no per-batch regions).
    // For operands without batch warps: compile-time filter by elemBatchIndex.
    // For operands with batch warps: runtime filter using batchWarpIdx.
    // In warp-distributed mode (curBatchRound < 0), scatter all elements,
    // each to its own batch region based on elemBatchTGOffset.
    auto stripScatter = [&](Value baseRow, Value baseCol,
                            SmallVector<Value> &elems,
                            SmallVector<SmallVector<unsigned>> &offsets,
                            int64_t stride, int64_t rowStart, bool mixed,
                            int64_t curBatchRound, Value operandBatchWarpIdx,
                            bool opHasBatchWarps, Type scatterTy) {
      for (size_t i = 0; i < elems.size(); ++i) {
        int64_t eb = (rowDim > 0) ? elemBatchIndex(offsets, i) : 0;

        // In sequential batch mode, skip elements not in current batch.
        if (curBatchRound >= 0 && rowDim > 0 && !opHasBatchWarps) {
          // No batch warps: compile-time batch index IS the actual batch.
          if (eb != curBatchRound)
            continue;
        }
        // For operands WITH batch warps: can't skip at compile time.
        // Runtime check is added below.

        int64_t rowOff = offsets[i][rowDim];
        int64_t colOff = offsets[i][colDim];
        Value actualRow = arith::AddIOp::create(
            rewriter, loc, baseRow,
            arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
        Value inStrip = arith::AndIOp::create(
            rewriter, loc,
            arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
            arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::ult, actualRow,
                arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));

        // For operands with batch warps in sequential mode, add runtime batch
        // check. actual batch = elemBatchIndex + batchWarpIdx match condition:
        // batchWarpIdx == curBatchRound - elemBatchIndex
        if (curBatchRound >= 0 && rowDim > 0 && opHasBatchWarps) {
          Value targetBatchWarp = arith::ConstantIntOp::create(
              rewriter, loc, curBatchRound - eb, 32);
          Value batchMatch =
              arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    operandBatchWarpIdx, targetBatchWarp);
          inStrip = arith::AndIOp::create(rewriter, loc, inStrip, batchMatch);
        }

        Value idx =
            stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride, rowStart);
        // Convert element to scatter type
        Value val = (scatterTy == f32Ty)
                        ? toF32(rewriter, loc, elems[i], f32Ty)
                        : toMmaInputType(rewriter, loc, elems[i], scatterTy);
        if (curBatchRound >= 0) {
          // Sequential mode: all data goes to TG base.
          Value safeIdx =
              arith::SelectOp::create(rewriter, loc, inStrip, idx, garbageIdx);
          Value gep =
              LLVM::GEPOp::create(rewriter, loc, tgPtrTy, scatterTy, ptrTG,
                                  ArrayRef<LLVM::GEPArg>{safeIdx});
          LLVM::StoreOp::create(rewriter, loc, val, gep);
        } else {
          // Warp-distributed mode: add per-element batch TG offset.
          Value batchOff = elemBatchTGOffset(offsets, i, mixed);
          Value batchIdx = arith::AddIOp::create(rewriter, loc, idx, batchOff);
          Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip,
                                                  batchIdx, garbageIdx);
          Value gep =
              LLVM::GEPOp::create(rewriter, loc, tgPtrTy, scatterTy, ptrTG,
                                  ArrayRef<LLVM::GEPArg>{safeIdx});
          LLVM::StoreOp::create(rewriter, loc, val, gep);
        }
      }
    };

    // ── Initialize result to zero ────────────────────────────────────
    auto outElemTy = cType.getElementType();
    SmallVector<Value> resultElems(elemsC.size());
    for (size_t i = 0; i < elemsC.size(); ++i)
      resultElems[i] = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(outElemTy));

    // ── MMA computation (batch-aware) ─────────────────────────────
    // When numBatchWarps >= batchSize, each warp handles one batch
    // (ptrTGBatch already points to that batch's TG region) -- single pass.
    // When numBatchWarps < batchSize, process one batch per round.
    // Each round: scatter/load/MMA/store/gather for a single batch.
    // All data goes to TG base (single region). Runtime batch filtering
    // ensures only the correct batch's data is scattered.

    bool batchConsistent = true;
    if (batchSize > 1) {
      auto aWpc = aSrcEnc.getWarpsPerCTA();
      auto bWpc = bSrcEnc.getWarpsPerCTA();
      int64_t matWarpsA = aWpc[rowDim] * aWpc[colDim];
      int64_t matWarpsB = bWpc[rowDim] * bWpc[colDim];
      if (matWarpsA != matWarpsC || matWarpsB != matWarpsC)
        batchConsistent = false;
    }

    int64_t batchRounds = 1;
    if (batchSize > 1 && (numBatchWarps < batchSize || !batchConsistent))
      batchRounds = batchSize;

    for (int64_t batchRound = 0; batchRound < batchRounds; ++batchRound) {
      Value curPtrTGBatch = ptrTGBatch;
      int64_t scatterBatchRound = -1; // -1 = warp-distributed (no filter)
      if (batchRounds > 1) {
        curPtrTGBatch = ptrTG; // sequential: single TG region at base
        scatterBatchRound = batchRound;
      }

      // Phase 1: Load A tiles (8-row strips) — native type MMA
      SmallVector<SmallVector<Value>> matA_tiles(tilesM);
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        matA_tiles[tm].resize(tilesK);
        int64_t rowStart = tm * 8;

        if (useAsyncA) {
          // Async copy: DMA from device to TG
          Value stripPtr = computeStripDevPtr(aPtrs, aOffsets, aBaseRow,
                                              aBaseCol, aRowStride, aColStride,
                                              rowStart, aType.getElementType());
          emitAsyncCopy(stripPtr, aRowStride, curPtrTGBatch, K, Kpad,
                        aType.getElementType());
        } else {
          stripScatter(aBaseRow, aBaseCol, elemsA, aOffsets, Kpad, rowStart,
                       aMixed, scatterBatchRound, aBatchWarpIdx, aHasBatchWarps,
                       abTgElemTy);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});

        for (int64_t tk = 0; tk < tilesK; ++tk) {
          Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
          matA_tiles[tm][tk] =
              emitSGLoad(abLoadFn, curPtrTGBatch, Kpad, Kpad, aOff);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
      }

      // Phase 2: Load C tiles (8-row strips)
      SmallVector<SmallVector<Value>> matC_tiles(tilesM);
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        matC_tiles[tm].resize(tilesN);
        int64_t rowStart = tm * 8;

        stripScatter(cBaseRow, cBaseCol, elemsC, cOffsets, Npad, rowStart,
                     cMixed, scatterBatchRound, cBatchWarpIdx, cHasBatchWarps,
                     f32Ty);
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});

        for (int64_t tn = 0; tn < tilesN; ++tn) {
          Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
          matC_tiles[tm][tn] =
              emitSGLoad(loadFn, curPtrTGBatch, Npad, Npad, cOff);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
      }

      // Phase 3: B strips + MMA — native type MMA
      // Double-buffered path: overlap DMA of B[k+1] with MMA on B[k].
      // Two TG slots of size 8*Npad alternate: slot 0 and slot 1.
      // Prologue loads B[0] into slot 0; each iteration fires prefetch
      // of B[k+1] into slot[(k+1)%2] then MMA on slot[k%2], then waits.
      // Falls back to single-buffer when double-buffering is not active.
      if (useDoubleBufB) {
        // Slot pointers: slot0 = curPtrTGBatch, slot1 = curPtrTGBatch + 8*Npad
        int64_t slotOffset = 8 * Npad; // offset in f32 elements
        Value ptrSlot0 = curPtrTGBatch;
        Value ptrSlot1 =
            LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty, curPtrTGBatch,
                                ArrayRef<LLVM::GEPArg>{(int64_t)slotOffset});

        // Prologue: load B[0] into slot 0, wait, barrier
        {
          Value stripPtr = computeStripDevPtr(bPtrs, bOffsets, bBaseRow,
                                              bBaseCol, bRowStride, bColStride,
                                              0, bType.getElementType());
          emitAsyncCopy(stripPtr, bRowStride, ptrSlot0, N, Npad,
                        bType.getElementType());
          LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                               ValueRange{fenceTG, execMod});
        }

        for (int64_t tk = 0; tk < tilesK; ++tk) {
          Value curSlotPtr = (tk % 2 == 0) ? ptrSlot0 : ptrSlot1;
          Value nextSlotPtr = (tk % 2 == 0) ? ptrSlot1 : ptrSlot0;

          // Prefetch B[tk+1] into next slot (fire only, no wait)
          if (tk + 1 < tilesK) {
            int64_t nextRowStart = (tk + 1) * 8;
            Value stripPtr = computeStripDevPtr(
                bPtrs, bOffsets, bBaseRow, bBaseCol, bRowStride, bColStride,
                nextRowStart, bType.getElementType());
            emitAsyncCopyFire(stripPtr, bRowStride, nextSlotPtr, N, Npad,
                              bType.getElementType());
          }

          // MMA on current slot (data ready from previous barrier)
          for (int64_t tn = 0; tn < tilesN; ++tn) {
            Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
            Value matB = emitSGLoad(abLoadFn, curSlotPtr, Npad, Npad, bOff);

            for (int64_t tm = 0; tm < tilesM; ++tm) {
              matC_tiles[tm][tn] =
                  LLVM::CallOp::create(
                      rewriter, loc, abMmaFn,
                      ValueRange{matA_tiles[tm][tk], matB, matC_tiles[tm][tn]})
                      .getResult();
            }
          }

          // Wait for prefetch + barrier (ensures next slot is ready)
          if (tk + 1 < tilesK) {
            emitAsyncCopyWait();
            LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                                 ValueRange{fenceTG, execMod});
          }
        }
      } else {
        // Single-buffer path (original): scatter/async → barrier → MMA →
        // barrier
        for (int64_t tk = 0; tk < tilesK; ++tk) {
          int64_t rowStart = tk * 8;

          if (useAsyncB) {
            Value stripPtr = computeStripDevPtr(
                bPtrs, bOffsets, bBaseRow, bBaseCol, bRowStride, bColStride,
                rowStart, bType.getElementType());
            emitAsyncCopy(stripPtr, bRowStride, curPtrTGBatch, N, Npad,
                          bType.getElementType());
          } else {
            stripScatter(bBaseRow, bBaseCol, elemsB, bOffsets, Npad, rowStart,
                         bMixed, scatterBatchRound, bBatchWarpIdx,
                         bHasBatchWarps, abTgElemTy);
          }
          LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                               ValueRange{fenceTG, execMod});

          for (int64_t tn = 0; tn < tilesN; ++tn) {
            Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
            Value matB = emitSGLoad(abLoadFn, curPtrTGBatch, Npad, Npad, bOff);

            for (int64_t tm = 0; tm < tilesM; ++tm) {
              matC_tiles[tm][tn] =
                  LLVM::CallOp::create(
                      rewriter, loc, abMmaFn,
                      ValueRange{matA_tiles[tm][tk], matB, matC_tiles[tm][tn]})
                      .getResult();
            }
          }
          LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                               ValueRange{fenceTG, execMod});
        }
      }

      // Phase 4: Store C tiles -> TG (8-row strips), gather
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        int64_t rowStart = tm * 8;

        for (int64_t tn = 0; tn < tilesN; ++tn) {
          Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
          emitSGStore(storeFn, matC_tiles[tm][tn], curPtrTGBatch, Npad, Npad,
                      cOff);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});

        // Gather: each thread reads its C elements from TG.
        for (size_t i = 0; i < elemsC.size(); ++i) {
          int64_t elemBatch = (rowDim > 0) ? elemBatchIndex(cOffsets, i) : 0;

          // In sequential batch mode, skip elements not in current batch.
          if (batchRounds > 1 && rowDim > 0 && !cHasBatchWarps) {
            // No batch warps: compile-time batch IS actual batch.
            if (elemBatch != batchRound)
              continue;
          }

          int64_t rowOff = cOffsets[i][rowDim];
          int64_t colOff = cOffsets[i][colDim];
          Value actualRow = arith::AddIOp::create(
              rewriter, loc, cBaseRow,
              arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
          Value inStrip = arith::AndIOp::create(
              rewriter, loc,
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
              arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                    actualRow,
                                    arith::ConstantIntOp::create(
                                        rewriter, loc, rowStart + 8, 32)));

          // For C with batch warps in sequential mode, add runtime batch check.
          if (batchRounds > 1 && rowDim > 0 && cHasBatchWarps) {
            Value targetBatchWarp = arith::ConstantIntOp::create(
                rewriter, loc, batchRound - elemBatch, 32);
            Value batchMatch =
                arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      cBatchWarpIdx, targetBatchWarp);
            inStrip = arith::AndIOp::create(rewriter, loc, inStrip, batchMatch);
          }

          Value idx =
              stripFlatIdx(cBaseRow, cBaseCol, rowOff, colOff, Npad, rowStart);
          if (batchRounds > 1) {
            // Sequential mode: data is at TG base.
            Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, idx,
                                                    garbageIdx);
            Value val = gather1(ptrTG, safeIdx);
            if (val.getType() != outElemTy)
              val = fromF32(rewriter, loc, val, outElemTy);
            resultElems[i] = arith::SelectOp::create(rewriter, loc, inStrip,
                                                     val, resultElems[i]);
          } else {
            // Warp-distributed mode: add batch TG offset.
            Value batchOff = elemBatchTGOffset(cOffsets, i, cMixed);
            Value batchIdx =
                arith::AddIOp::create(rewriter, loc, idx, batchOff);
            Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip,
                                                    batchIdx, garbageIdx);
            Value val = gather1(ptrTG, safeIdx);
            if (val.getType() != outElemTy)
              val = fromF32(rewriter, loc, val, outElemTy);
            resultElems[i] = arith::SelectOp::create(rewriter, loc, inStrip,
                                                     val, resultElems[i]);
          }
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
      }
    } // end batchRound loop

    // ── Pack result ───────────────────────────────────────────────────
    auto outLLVMTy = getTypeConverter()->convertType(cType);
    if (!outLLVMTy)
      return failure();

    if (auto outStructTy = dyn_cast<LLVMStructType>(outLLVMTy)) {
      Value result = UndefOp::create(rewriter, loc, outStructTy);
      for (size_t i = 0; i < resultElems.size(); ++i)
        result = InsertValueOp::create(rewriter, loc, outStructTy, result,
                                       resultElems[i],
                                       ArrayRef<int64_t>{(int64_t)i});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, resultElems[0]);
    }
    return success();
  }
};

// ============================================================================
// DotOpAppleMmaConversion: AppleMmaEncoding on C, rank-2 only.
// Handles dots where AccelerateAppleMatmul has rewritten C to AppleMmaEncoding.
// Uses the simpler 2D-only path with static strip bucketing.
//
// OPTIMIZATION: When A/B operands come from tt.load (device memory), loads
// MMA tiles directly from device memory via p1f32 intrinsics, eliminating
// the TG scatter/gather bottleneck and ~12 barriers per dot iteration.
// Falls back to TG scatter path when device pointers are unavailable.
// ============================================================================
struct DotOpAppleMmaConversion : public ConvertOpToLLVMPattern<tt::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tt::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto cType = cast<RankedTensorType>(op.getC().getType());
    auto cMmaEnc = dyn_cast<AppleMmaEncodingAttr>(cType.getEncoding());
    // Only handle AppleMmaEncoding on C.
    if (!cMmaEnc)
      return failure();

    // Only support rank-2 tensors for MMA encoding path.
    unsigned rank = cType.getRank();
    if (rank != 2)
      return failure();

    auto aType = cast<RankedTensorType>(op.getA().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());

    int64_t M = cType.getShape()[0];
    int64_t N = cType.getShape()[1];
    int64_t K = aType.getShape()[1]; // A is [M, K]

    auto f32Ty = Float32Type::get(ctx);
    auto tgPtrTy = LLVMPointerType::get(ctx, 3);
    auto devPtrTy = LLVMPointerType::get(ctx, 1);
    auto matTy = getSimdgroupMatrixType(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);

    // ── Declare air intrinsics ────────────────────────────────────────

    auto laneIdFn =
        getOrInsertIntrinsic(rewriter, mod, "air.thread_index_in_simdgroup",
                             LLVMFunctionType::get(i32Ty, {}, false));

    auto voidTy = LLVMVoidType::get(ctx);
    auto barrTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
    auto tgBarrFn =
        getOrInsertIntrinsic(rewriter, mod, "air.threadgroup.barrier", barrTy);
    (void)getOrInsertIntrinsic(rewriter, mod, "air.simdgroup.barrier", barrTy);

    auto vec2i64Ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
    auto i1Ty = IntegerType::get(ctx, 1);
    bool canonSG = useCanonicalSimdgroupSig();
    // Build the simdgroup load/store arg-type lists for the selected target
    // ABI.
    auto sgLoadArgTys = [&](Type ptrTy) -> SmallVector<Type> {
      if (canonSG)
        return {ptrTy, i64Ty, vec2i64Ty, i1Ty};
      return {ptrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty};
    };
    auto sgStoreArgTys = [&](Type matVecTy, Type ptrTy) -> SmallVector<Type> {
      if (canonSG)
        return {matVecTy, ptrTy, i64Ty, vec2i64Ty, i1Ty};
      return {matVecTy, ptrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty};
    };
    // f32 TG load/store (C accumulator always f32)
    auto loadFn = getOrInsertIntrinsic(
        rewriter, mod, "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
        LLVMFunctionType::get(matTy, sgLoadArgTys(tgPtrTy), false));
    auto storeFn = getOrInsertIntrinsic(
        rewriter, mod, "air.simdgroup_matrix_8x8_store.v64f32.p3f32",
        LLVMFunctionType::get(voidTy, sgStoreArgTys(matTy, tgPtrTy), false));

    // Type-specific TG MMA load/multiply for A/B (bf16/f16 use native MMA)
    auto abTgMmaInfo = getMMAIntrinsicInfo(ctx, aType.getElementType());
    auto abTgLoadFn = getOrInsertIntrinsic(
        rewriter, mod, abTgMmaInfo.tgLoadName,
        LLVMFunctionType::get(abTgMmaInfo.matVecTy, sgLoadArgTys(tgPtrTy),
                              false));
    auto abTgMmaFn = getOrInsertIntrinsic(
        rewriter, mod, abTgMmaInfo.mmaName,
        LLVMFunctionType::get(
            matTy, {abTgMmaInfo.matVecTy, abTgMmaInfo.matVecTy, matTy}, false));
    // TG element type for A/B in TG fallback path
    Type abTgScatterTy = f32Ty;
    if (aType.getElementType().isF16())
      abTgScatterTy = Float16Type::get(ctx);
    else if (aType.getElementType().isBF16())
      abTgScatterTy = BFloat16Type::get(ctx);

    // simdgroup TG load/store emitters (compile-time-constant pitch). See the
    // identically-named helpers in DotOpBlockedConversion for the ABI details.
    auto emitSGLoad = [&](LLVMFuncOp fn, Value ptr, int64_t shapeDim,
                          int64_t pitch, Value off) -> Value {
      SmallVector<Value> args;
      if (canonSG)
        args = {ptr, makeI64(rewriter, loc, pitch), off,
                makeI1False(rewriter, loc)};
      else
        args = {ptr, makeI64Vec2(rewriter, loc, shapeDim, 8),
                makeI64Vec2(rewriter, loc, 1, pitch), off};
      return LLVM::CallOp::create(rewriter, loc, fn, args).getResult();
    };
    auto emitSGStore = [&](LLVMFuncOp fn, Value mat, Value ptr,
                           int64_t shapeDim, int64_t pitch, Value off) {
      SmallVector<Value> args;
      if (canonSG)
        args = {mat, ptr, makeI64(rewriter, loc, pitch), off,
                makeI1False(rewriter, loc)};
      else
        args = {mat, ptr, makeI64Vec2(rewriter, loc, shapeDim, 8),
                makeI64Vec2(rewriter, loc, 1, pitch), off};
      LLVM::CallOp::create(rewriter, loc, fn, args);
    };
    // Device simdgroup load emitter: strides are runtime Values. devStride is
    // the row pitch (canonical elements_per_row), devShape the 3-vector shape.
    auto emitDevSGLoad = [&](LLVMFuncOp fn, Value ptr, Value devShape,
                             Value devStride, Value off,
                             Value transposeFalse) -> Value {
      SmallVector<Value> args;
      if (canonSG)
        args = {ptr, devStride, off, transposeFalse};
      else
        args = {ptr, devShape, devStride, off};
      return LLVM::CallOp::create(rewriter, loc, fn, args).getResult();
    };

    // ── Device memory MMA intrinsics (type-specific) ─────────────────
    // For f16/bf16 data: load returns <64 x half/bfloat>, MMA takes
    // half/bfloat inputs with f32 accumulator.
    // For f32 data: load returns <64 x float>, MMA takes f32 inputs.
    auto aElemTy = aType.getElementType();
    auto f16Ty = Float16Type::get(ctx);
    auto bf16Ty = BFloat16Type::get(ctx);
    bool isF16Input = aElemTy.isF16();
    bool isBF16Input = aElemTy.isBF16();

    // Determine device load intrinsic based on element type
    Type devMatElemTy = f32Ty; // element type for device MMA matrix
    std::string devLoadName = "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
    std::string devMmaName = "air.simdgroup_matrix_8x8_multiply_accumulate."
                             "v64f32.v64f32.v64f32.v64f32";
    Type devGepElemTy = f32Ty; // GEP element type

    if (isF16Input) {
      devMatElemTy = f16Ty;
      devLoadName = "air.simdgroup_matrix_8x8_load.v64f16.p1f16";
      devMmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16."
                   "v64f16.v64f32";
      devGepElemTy = f16Ty;
    } else if (isBF16Input) {
      devMatElemTy = bf16Ty;
      devLoadName = "air.simdgroup_matrix_8x8_load.v64bf16.p1bf16";
      devMmaName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32."
                   "v64bf16.v64bf16.v64f32";
      devGepElemTy = bf16Ty;
    }

    auto devMatTy = LLVM::getVectorType(devMatElemTy, 64);
    auto devLoadFn = getOrInsertIntrinsic(
        rewriter, mod, devLoadName,
        LLVMFunctionType::get(devMatTy, sgLoadArgTys(devPtrTy), false));
    auto devMmaFn = getOrInsertIntrinsic(
        rewriter, mod, devMmaName,
        LLVMFunctionType::get(matTy, {devMatTy, devMatTy, matTy}, false));

    // ── Constants ────────────────────────────────────────────────────

    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

    // ── Thread identification ─────────────────────────────────────────

    Value laneId =
        LLVM::CallOp::create(rewriter, loc, laneIdFn, ValueRange{}).getResult();

    auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto tidFn = getOrInsertIntrinsic(
        rewriter, mod, "air.thread_position_in_threadgroup",
        LLVMFunctionType::get(arrI32x3Ty, {}, false));
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                               ArrayRef<int64_t>{0});
    Value warpId = divByConst(rewriter, loc, tid32, 32); // tid / 32

    // ── Get blocked encoding params for A, B ────────────────────────

    auto unpack = [&](Value v) -> SmallVector<Value> {
      SmallVector<Value> elems;
      if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
        for (unsigned i = 0; i < sTy.getBody().size(); ++i)
          elems.push_back(
              ExtractValueOp::create(rewriter, loc, sTy.getBody()[i], v,
                                     ArrayRef<int64_t>{(int64_t)i}));
      } else {
        elems = {v};
      }
      return elems;
    };

    auto resolveOperand = [&](Value tritonVal, Value adaptorVal,
                              RankedTensorType opTy)
        -> std::tuple<SmallVector<Value>, SmallVector<SmallVector<unsigned>>,
                      ttg::BlockedEncodingAttr> {
      if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>()) {
        Value mapped = rewriter.getRemappedValue(cvt.getSrc());
        if (mapped) {
          auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
          auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
          if (srcEnc) {
            auto offsets = emitOffsetForLayout(srcEnc, srcTy);
            return {unpack(mapped), offsets, srcEnc};
          }
        }
      }
      auto enc = opTy.getEncoding();
      if (auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(enc)) {
        auto parentEnc = dyn_cast<ttg::BlockedEncodingAttr>(dotEnc.getParent());
        if (parentEnc) {
          auto offsets = emitOffsetForLayout(enc, opTy);
          return {unpack(adaptorVal), offsets, parentEnc};
        }
      }
      if (auto blk = dyn_cast<ttg::BlockedEncodingAttr>(enc)) {
        auto offsets = emitOffsetForLayout(blk, opTy);
        return {unpack(adaptorVal), offsets, blk};
      }
      return {{}, {}, nullptr};
    };

    // ── Try to extract device pointers for direct MMA loads ──────────
    // Trace: dot.A → ConvertLayoutOp → LoadOp → pointer operand
    // Returns per-thread device pointers with the same encoding/offsets as the
    // values.
    auto resolveDevicePointers = [&](Value tritonVal) -> SmallVector<Value> {
      // Look through ConvertLayoutOp if present
      Value src = tritonVal;
      if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      // Check if source is a LoadOp
      auto loadOp = src.getDefiningOp<tt::LoadOp>();
      if (!loadOp)
        return {};
      // Get the pointer operand (tensor of device pointers)
      Value ptrTensor = loadOp.getPtr();
      Value mappedPtrs = rewriter.getRemappedValue(ptrTensor);
      if (!mappedPtrs)
        return {};
      return unpack(mappedPtrs);
    };

    auto [elemsA, aOffsets, aSrcEnc] =
        resolveOperand(op.getA(), adaptor.getA(), aType);
    auto [elemsB, bOffsets, bSrcEnc] =
        resolveOperand(op.getB(), adaptor.getB(), bType);
    auto elemsC = unpack(adaptor.getC());
    auto cOffsets = emitOffsetForLayout(cMmaEnc, cType);

    if (!aSrcEnc || !bSrcEnc)
      return failure();

    if ((int64_t)elemsA.size() != (int64_t)aOffsets.size() ||
        (int64_t)elemsB.size() != (int64_t)bOffsets.size() ||
        (int64_t)elemsC.size() != (int64_t)cOffsets.size())
      return failure();

    // Try to get device pointers for A and B
    auto aPtrs = resolveDevicePointers(op.getA());
    auto bPtrs = resolveDevicePointers(op.getB());
    bool useDeviceA = (aPtrs.size() == elemsA.size() && !aPtrs.empty());
    bool useDeviceB = (bPtrs.size() == elemsB.size() && !bPtrs.empty());

    // Env var TRITON_DEVICE_MMA=0 to force TG path (for debugging)
    {
      const char *envVar = std::getenv("TRITON_DEVICE_MMA");
      if (envVar && std::string(envVar) == "0") {
        useDeviceA = false;
        useDeviceB = false;
      }
    }

    // Device MMA loads only support float types (f32, f16, bf16).
    // For integer element types (e.g. int8), the MMA intrinsic reads f32
    // from device memory but the actual data is packed at 1 byte/element,
    // causing a 4x stride mismatch. Force TG scatter path for integers.
    if (isa<IntegerType>(aElemTy)) {
      useDeviceA = false;
      useDeviceB = false;
    }

    // ── Compute row stride from pointer differences ──────────────────
    // For device MMA loads, we need the row stride (leading dimension)
    // in elements. Strategy:
    //   1. Try to find two elements on THIS thread with same col, different
    //   row.
    //   2. If not (common: all elements on same row), use simd_shuffle to
    //      get the same-column element from the thread in the NEXT row.
    //      For encoding spt=[1,8], tpw=[16,2], order=[1,0]:
    //        lR = laneId / tpw[colDim] = laneId / 2
    //        lC = laneId % tpw[colDim] = laneId % 2
    //      Thread laneId and laneId+tpw[colDim] differ by 1 row.
    //      Shuffle offset = tpw[colDim] for row-adjacent lane.
    auto computeRowStride = [&](SmallVector<Value> &ptrs,
                                SmallVector<SmallVector<unsigned>> &offsets,
                                Type elemTy,
                                ttg::BlockedEncodingAttr enc) -> Value {
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

      // Strategy 1: same-thread pair with same col, different row
      for (size_t i = 0; i < offsets.size(); ++i) {
        for (size_t j = i + 1; j < offsets.size(); ++j) {
          if (offsets[i][1] == offsets[j][1] &&
              offsets[i][0] != offsets[j][0]) {
            int64_t rowDiff = (int64_t)offsets[j][0] - (int64_t)offsets[i][0];
            Value ptrI64_i =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[i]);
            Value ptrI64_j =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[j]);
            Value byteDiff =
                arith::SubIOp::create(rewriter, loc, ptrI64_j, ptrI64_i);
            Value elemSize = arith::ConstantIntOp::create(
                rewriter, loc, (int64_t)elemBytes, 64);
            Value elemDiff =
                arith::DivSIOp::create(rewriter, loc, byteDiff, elemSize);
            Value rowDiffVal =
                arith::ConstantIntOp::create(rewriter, loc, rowDiff, 64);
            return arith::DivSIOp::create(rewriter, loc, elemDiff, rowDiffVal);
          }
        }
      }

      // Strategy 2: simd_shuffle_xor to get pointer from adjacent-row thread.
      // For blocked encoding with order=[1,0] (col-fastest):
      //   XOR mask = threadsPerWarp[colDim] swaps row-adjacent threads
      // For order=[0,1] (row-fastest):
      //   XOR mask = 1 swaps row-adjacent threads
      auto order = enc.getOrder();
      auto tpw = enc.getThreadsPerWarp();
      auto spt = enc.getSizePerThread();
      unsigned encRank = spt.size();
      unsigned encColDim = encRank - 1;
      unsigned encRowDim = encRank - 2;
      bool colFastest = (order[0] == (unsigned)encColDim);

      // XOR mask that swaps row-adjacent threads
      int64_t xorMask = colFastest ? tpw[encColDim] : 1;
      int64_t rowDiff = spt[encRowDim];

      // Convert ptr to i64, shuffle_xor, compute stride.
      // Split i64 into two i32 for the shuffle (Metal doesn't support i64
      // shuffle).
      Value ptrI64 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[0]);
      auto i16Ty = IntegerType::get(ctx, 16);
      Value xorVal = arith::ConstantIntOp::create(rewriter, loc, xorMask, 16);

      // Split i64 into lo/hi i32, shuffle each, recombine
      Value lo = arith::TruncIOp::create(rewriter, loc, i32Ty, ptrI64);
      Value hi = arith::TruncIOp::create(
          rewriter, loc, i32Ty,
          arith::ShRUIOp::create(
              rewriter, loc, ptrI64,
              arith::ConstantIntOp::create(rewriter, loc, 32, 64)));

      auto shuffleFn32 = getOrInsertIntrinsic(
          rewriter, mod, "air.simd_shuffle_xor.s.i32",
          LLVMFunctionType::get(i32Ty, {i32Ty, i16Ty}, false));
      Value shufLo = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{lo, xorVal})
                         .getResult();
      Value shufHi = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{hi, xorVal})
                         .getResult();

      // Recombine: result = zext(shuf_lo) | (zext(shuf_hi) << 32)
      Value loExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufLo);
      Value hiExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufHi);
      Value hiShl = arith::ShLIOp::create(
          rewriter, loc, hiExt,
          arith::ConstantIntOp::create(rewriter, loc, 32, 64));
      Value otherPtrI64 = arith::OrIOp::create(rewriter, loc, loExt, hiShl);

      // Compute absolute stride: abs(other - this) / rowDiff
      Value byteDiff =
          arith::SubIOp::create(rewriter, loc, otherPtrI64, ptrI64);
      // Take absolute value: if negative, negate
      Value zero64 =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
      Value isNeg = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, byteDiff, zero64);
      Value negDiff = arith::SubIOp::create(rewriter, loc, zero64, byteDiff);
      Value absDiff =
          arith::SelectOp::create(rewriter, loc, isNeg, negDiff, byteDiff);

      Value elemSize =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
      Value elemDiff = arith::DivUIOp::create(rewriter, loc, absDiff, elemSize);
      if (rowDiff > 1) {
        Value rowDiffVal =
            arith::ConstantIntOp::create(rewriter, loc, rowDiff, 64);
        elemDiff = arith::DivUIOp::create(rewriter, loc, elemDiff, rowDiffVal);
      }
      return elemDiff;
    };

    // Compute column stride (distance between adjacent columns in same row).
    // Mirrors computeRowStride but looks for same-row-different-col pairs.
    auto computeColStride = [&](SmallVector<Value> &ptrs,
                                SmallVector<SmallVector<unsigned>> &offsets,
                                Type elemTy,
                                ttg::BlockedEncodingAttr enc) -> Value {
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

      // Strategy 1: same-thread pair with same row, different col
      for (size_t i = 0; i < offsets.size(); ++i) {
        for (size_t j = i + 1; j < offsets.size(); ++j) {
          if (offsets[i][0] == offsets[j][0] &&
              offsets[i][1] != offsets[j][1]) {
            int64_t colDiff = (int64_t)offsets[j][1] - (int64_t)offsets[i][1];
            Value ptrI64_i =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[i]);
            Value ptrI64_j =
                LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[j]);
            Value byteDiff =
                arith::SubIOp::create(rewriter, loc, ptrI64_j, ptrI64_i);
            Value elemSize = arith::ConstantIntOp::create(
                rewriter, loc, (int64_t)elemBytes, 64);
            Value elemDiff =
                arith::DivSIOp::create(rewriter, loc, byteDiff, elemSize);
            Value colDiffVal =
                arith::ConstantIntOp::create(rewriter, loc, colDiff, 64);
            return arith::DivSIOp::create(rewriter, loc, elemDiff, colDiffVal);
          }
        }
      }

      // Strategy 2: simd_shuffle_xor to get pointer from adjacent-col thread.
      auto order = enc.getOrder();
      auto tpw = enc.getThreadsPerWarp();
      auto spt = enc.getSizePerThread();
      unsigned encRank = spt.size();
      unsigned encColDim = encRank - 1;
      unsigned encRowDim = encRank - 2;
      bool colFastest = (order[0] == (unsigned)encColDim);

      // XOR mask that swaps col-adjacent threads (opposite of row)
      int64_t xorMask = colFastest ? 1 : tpw[encRowDim];
      int64_t colDiff = spt[encColDim];

      Value ptrI64 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptrs[0]);
      auto i16Ty = IntegerType::get(ctx, 16);
      Value xorVal = arith::ConstantIntOp::create(rewriter, loc, xorMask, 16);

      Value lo = arith::TruncIOp::create(rewriter, loc, i32Ty, ptrI64);
      Value hi = arith::TruncIOp::create(
          rewriter, loc, i32Ty,
          arith::ShRUIOp::create(
              rewriter, loc, ptrI64,
              arith::ConstantIntOp::create(rewriter, loc, 32, 64)));

      auto shuffleFn32 = getOrInsertIntrinsic(
          rewriter, mod, "air.simd_shuffle_xor.s.i32",
          LLVMFunctionType::get(i32Ty, {i32Ty, i16Ty}, false));
      Value shufLo = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{lo, xorVal})
                         .getResult();
      Value shufHi = LLVM::CallOp::create(rewriter, loc, shuffleFn32,
                                          ValueRange{hi, xorVal})
                         .getResult();

      Value loExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufLo);
      Value hiExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, shufHi);
      Value hiShl = arith::ShLIOp::create(
          rewriter, loc, hiExt,
          arith::ConstantIntOp::create(rewriter, loc, 32, 64));
      Value otherPtrI64 = arith::OrIOp::create(rewriter, loc, loExt, hiShl);

      Value byteDiff =
          arith::SubIOp::create(rewriter, loc, otherPtrI64, ptrI64);
      Value zero64 =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
      Value isNeg = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, byteDiff, zero64);
      Value negDiff = arith::SubIOp::create(rewriter, loc, zero64, byteDiff);
      Value absDiff =
          arith::SelectOp::create(rewriter, loc, isNeg, negDiff, byteDiff);

      Value elemSize =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
      Value elemDiff = arith::DivUIOp::create(rewriter, loc, absDiff, elemSize);
      if (colDiff > 1) {
        Value colDiffVal =
            arith::ConstantIntOp::create(rewriter, loc, colDiff, 64);
        elemDiff = arith::DivUIOp::create(rewriter, loc, elemDiff, colDiffVal);
      }
      return elemDiff;
    };

    Value aRowStride, bRowStride, aColStride, bColStride;
    if (useDeviceA) {
      aRowStride =
          computeRowStride(aPtrs, aOffsets, aType.getElementType(), aSrcEnc);
      aColStride =
          computeColStride(aPtrs, aOffsets, aType.getElementType(), aSrcEnc);
    }
    if (useDeviceB) {
      bRowStride =
          computeRowStride(bPtrs, bOffsets, bType.getElementType(), bSrcEnc);
      bColStride =
          computeColStride(bPtrs, bOffsets, bType.getElementType(), bSrcEnc);
    }

    // If stride computation failed, fall back to TG path
    if (useDeviceA && (!aRowStride || !aColStride))
      useDeviceA = false;
    if (useDeviceB && (!bRowStride || !bColStride))
      useDeviceB = false;

    // ── Compute device base pointer for each 8x8 MMA tile ───────────
    // tile_base = ptr[0] - (aOffsets[0][0] * rowStride + aOffsets[0][1]) *
    // elemSize Then offset by (tm*8 * rowStride + tk*8) * elemSize for each
    // tile.
    //
    // For the MMA load, we pass the pointer as float* regardless of the
    // actual element type (the intrinsic always loads as f32). But the
    // stride and offset must be in elements of the SOURCE type, so we
    // convert the base pointer to the proper byte address.
    // Compute tile base pointer for MMA load.
    // ALL threads must compute the SAME base pointer.
    // ptrs[0] points to the thread's own element at (baseRow+offsets[0][0],
    // baseCol+offsets[0][1]). Subtract this position to get the tile origin.
    //
    // tile_ptr = ptrs[0] + (tileRow - baseRow - offsets[0][0]) * rowStride
    //                    + (tileCol - baseCol - offsets[0][1])
    auto computeTileDevPtr = [&](SmallVector<Value> &ptrs,
                                 SmallVector<SmallVector<unsigned>> &offsets,
                                 Value rowStride, Value colStride,
                                 Value baseRow, Value baseCol, int64_t tileRow,
                                 int64_t tileCol) -> Value {
      Value refPtr = ptrs[0];
      int64_t refRowOff = offsets[0][0];
      int64_t refColOff = offsets[0][1];

      // rowDelta = tileRow - baseRow - refRowOff (runtime)
      Value baseRowExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseRow);
      Value rowDelta = arith::SubIOp::create(
          rewriter, loc,
          arith::ConstantIntOp::create(rewriter, loc,
                                       (int64_t)tileRow - refRowOff, 64),
          baseRowExt);

      // colDelta = tileCol - baseCol - refColOff (runtime)
      Value baseColExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseCol);
      Value colDelta = arith::SubIOp::create(
          rewriter, loc,
          arith::ConstantIntOp::create(rewriter, loc,
                                       (int64_t)tileCol - refColOff, 64),
          baseColExt);

      // elemOff = rowDelta * rowStride + colDelta * colStride
      Value elemOff = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(rewriter, loc, rowDelta, rowStride),
          arith::MulIOp::create(rewriter, loc, colDelta, colStride));

      Value tilePtr =
          LLVM::GEPOp::create(rewriter, loc, devPtrTy, devGepElemTy, refPtr,
                              ArrayRef<LLVM::GEPArg>{elemOff});
      return tilePtr;
    };

    // ── Make MMA stride vector for device loads ──────────────────────
    // stride = <colStride, rowStride>
    // For row-major data: colStride=1, rowStride=numCols
    // For col-major data: colStride=numRows, rowStride=1
    // Canonical (macOS<=15) device load uses a scalar elements_per_row (the row
    // stride); the 3-vector form uses the <colStride, rowStride> vector.
    auto makeDevMmaStride = [&](Value colStride, Value rowStride) -> Value {
      if (canonSG) {
        // Canonical (macOS<=15) simdgroup load takes a single scalar
        // elements_per_row pitch plus a transpose bool — it cannot express a
        // 2D <colStride,rowStride>.  The 3-vector form addresses element (i,j)
        // as base + i*rowStride + j*colStride.  The canonical form addresses
        // (i,j) as base + i*pitch + j  (transpose=false) or base + i + j*pitch
        // (transpose=true).  For a device MMA tile exactly one of the two
        // strides is unit (the operand is contiguous in one dim):
        //   row-major  (colStride==1): pitch=rowStride, transpose=false
        //   col-major  (rowStride==1): pitch=colStride, transpose=true
        // Pick the non-unit stride as the pitch.  The transpose flag is derived
        // at the call site from the same comparison.
        Value rowBig = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::uge, rowStride, colStride);
        return arith::SelectOp::create(rewriter, loc, rowBig, rowStride,
                                       colStride);
      }
      auto ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
      Value vec = UndefOp::create(rewriter, loc, ty);
      Value i0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value i1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      vec = InsertElementOp::create(rewriter, loc, ty, vec, colStride, i0);
      vec = InsertElementOp::create(rewriter, loc, ty, vec, rowStride, i1);
      return vec;
    };
    // Canonical transpose flag for a device MMA load: true when the operand is
    // column-major (rowStride < colStride, i.e. rows are the contiguous dim).
    // Returns i1 false for the 3-vector path (transpose is folded into the
    // stride vector there).
    auto makeDevMmaTranspose = [&](Value colStride, Value rowStride) -> Value {
      if (!canonSG)
        return makeI1False(rewriter, loc);
      return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                   rowStride, colStride);
    };

    // ── Compute runtime thread base position ──────────────────────────
    auto makeBase = [&](ttg::BlockedEncodingAttr enc, int64_t rows,
                        int64_t cols) -> std::pair<Value, Value> {
      auto spt = enc.getSizePerThread();
      auto tpw = enc.getThreadsPerWarp();
      auto wpc = enc.getWarpsPerCTA();
      auto order = enc.getOrder();

      unsigned encRank = spt.size();
      unsigned encRowDim = encRank - 2;
      unsigned encColDim = encRank - 1;

      int64_t sM = spt[encRowDim], sN = spt[encColDim];
      int64_t tM = tpw[encRowDim], tN = tpw[encColDim];
      int64_t wM = wpc[encRowDim], wN = wpc[encColDim];
      int64_t tileM = wM * tM * sM;
      int64_t tileN = wN * tN * sN;

      bool colFastest = (order[0] == (unsigned)encColDim);

      Value tMsM = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
      Value sM_val = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
      Value tNsN = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
      Value sN_val = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

      // Warp decomposition: Morton order when both dims are power-of-2,
      // otherwise linear div/mod.
      Value wR, wC;
      unsigned mortonBits = mortonBitsPerDim(wM, wN);
      if (mortonBits > 0) {
        if (colFastest) {
          wR = mortonDeinterleaveEven(rewriter, loc, warpId, mortonBits);
          wC = mortonDeinterleaveOdd(rewriter, loc, warpId, mortonBits);
        } else {
          wC = mortonDeinterleaveEven(rewriter, loc, warpId, mortonBits);
          wR = mortonDeinterleaveOdd(rewriter, loc, warpId, mortonBits);
        }
        if (wM < (1LL << mortonBits))
          wR = remByConst(rewriter, loc, wR, wM);
        if (wN < (1LL << mortonBits))
          wC = remByConst(rewriter, loc, wC, wN);
      } else if (colFastest) {
        wR = divByConst(rewriter, loc, warpId, wN);
        wC = remByConst(rewriter, loc, warpId, wN);
      } else {
        wR = remByConst(rewriter, loc, warpId, wM);
        wC = divByConst(rewriter, loc, warpId, wM);
      }
      // Lane decomposition: shift/mask when power-of-2
      Value lR, lC;
      if (colFastest) {
        lR = divByConst(rewriter, loc, laneId, tN);
        lC = remByConst(rewriter, loc, laneId, tN);
      } else {
        lR = remByConst(rewriter, loc, laneId, tM);
        lC = divByConst(rewriter, loc, laneId, tM);
      }

      Value baseRow = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wR, tMsM),
          arith::MulIOp::create(rewriter, loc, lR, sM_val));
      Value baseCol = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wC, tNsN),
          arith::MulIOp::create(rewriter, loc, lC, sN_val));

      if (tileM > rows)
        baseRow = remByConst(rewriter, loc, baseRow, rows);
      if (tileN > cols)
        baseCol = remByConst(rewriter, loc, baseCol, cols);

      return {baseRow, baseCol};
    };

    // MMA base: lane->(row,col) within 8x8 tile, warp->tile position.
    // Must use column-major warp ordering (warpId % wN → col, warpId / wN →
    // row) to match the LinearLayout in AppleMmaEncodingAttr::toLinearLayout(),
    // which the upstream convert_layout relies on for shared memory addressing.
    // Do NOT use Morton order here — Morton is only safe for internal TG
    // operations where scatter and gather are self-consistent.
    auto makeBaseMma = [&](AppleMmaEncodingAttr enc, int64_t rows,
                           int64_t cols) -> std::pair<Value, Value> {
      auto wpc = enc.getWarpsPerCTA();
      unsigned wN = wpc[1];

      Value c7 = arith::ConstantIntOp::create(rewriter, loc, 7, 32);
      Value c3 = arith::ConstantIntOp::create(rewriter, loc, 3, 32);
      Value laneCol = arith::AndIOp::create(rewriter, loc, laneId, c7);
      Value laneRow = arith::ShRUIOp::create(rewriter, loc, laneId, c3);

      Value c8 = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
      // Column-major warp tiling: matches toLinearLayout warpOrder={1,0}
      Value warpRow = divByConst(rewriter, loc, warpId, wN);
      Value warpCol = remByConst(rewriter, loc, warpId, wN);

      Value baseRow = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpRow, c8),
          laneRow);
      Value baseCol = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpCol, c8),
          laneCol);

      return {baseRow, baseCol};
    };

    auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
    auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
    auto [cBaseRow, cBaseCol] = makeBaseMma(cMmaEnc, M, N);

    // ── Create threadgroup global ─────────────────────────────────────
    // TG buffer needed for C scatter/load (always), and A/B scatter (TG path).
    unsigned id = getDotCounter(ctx)++;
    int64_t pad = tgPadForType(aElemTy);
    int64_t maxStrideMma = (useDeviceA && useDeviceB) ? N : std::max(K, N);
    int64_t unpaddedSizeMma = 8 * maxStrideMma + 1;
    bool canPadMma = (pad > 0) && ((unpaddedSizeMma + 8 * pad) * 4 <= 16384);
    int64_t Kpad = canPadMma ? K + pad : K;
    int64_t Npad = canPadMma ? N + pad : N;
    int64_t tgCStripSize = 8 * Npad;
    int64_t tgABStripSize = 8 * std::max(Kpad, Npad);
    int64_t tgStripSize =
        (useDeviceA && useDeviceB) ? tgCStripSize : tgABStripSize;
    int64_t tgSize = tgStripSize + 1;
    auto tgBuf = getOrCreateTGGlobal(
        rewriter, mod, ("__tg_dot_ab_" + llvm::Twine(id)).str(), tgSize);

    Value ptrTG =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgBuf.getName());

    // ── GEP helpers ───────────────────────────────────────────────────

    auto gather1 = [&](Value ptr, Value flatIdx64) -> Value {
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty, ptr,
                                      ArrayRef<LLVM::GEPArg>{flatIdx64});
      return LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult();
    };

    auto stripFlatIdx = [&](Value baseRow, Value baseCol, int64_t rowOff,
                            int64_t colOff, int64_t stride,
                            int64_t stripRowStart) -> Value {
      Value row32 =
          arith::AddIOp::create(rewriter, loc, baseRow,
                                arith::ConstantIntOp::create(
                                    rewriter, loc, rowOff - stripRowStart, 32));
      Value col32 = arith::AddIOp::create(
          rewriter, loc, baseCol,
          arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
      Value flat32 = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(
              rewriter, loc, row32,
              arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
          col32);
      return arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
    };

    int64_t tilesM = M / 8;
    int64_t tilesN = N / 8;
    int64_t tilesK = K / 8;

    // ── Static strip filtering ────────────────────────────────────────
    auto maxBaseRow = [](ttg::BlockedEncodingAttr enc) -> int64_t {
      auto spt = enc.getSizePerThread();
      auto tpw = enc.getThreadsPerWarp();
      auto wpc = enc.getWarpsPerCTA();
      unsigned encRank = spt.size();
      unsigned encRowDim = encRank - 2;
      return wpc[encRowDim] * tpw[encRowDim] * spt[encRowDim] - 1;
    };

    // For MMA encoding, max base row = warpsM * 8 - 1
    int64_t cMaxBase = cMmaEnc.getWarpsPerCTA()[0] * 8 - 1;

    auto bucketElements =
        [](SmallVector<SmallVector<unsigned>> &offsets, int64_t maxBase,
           int64_t numStrips,
           unsigned rowIdx) -> SmallVector<SmallVector<size_t>> {
      SmallVector<SmallVector<size_t>> buckets(numStrips);
      for (size_t i = 0; i < offsets.size(); ++i) {
        int64_t rowOff = offsets[i][rowIdx];
        for (int64_t s = 0; s < numStrips; ++s) {
          int64_t stripStart = s * 8;
          int64_t lo = stripStart - rowOff;
          int64_t hi = lo + 8;
          if (lo <= maxBase && hi > 0)
            buckets[s].push_back(i);
        }
      }
      return buckets;
    };

    auto cBuckets = bucketElements(cOffsets, cMaxBase, tilesM, 0);

    Value garbageIdx =
        arith::ConstantIntOp::create(rewriter, loc, tgStripSize, 64);

    auto filteredScatter = [&](Value ptr, Value garbIdx, Value baseRow,
                               Value baseCol, SmallVector<Value> &elems,
                               SmallVector<SmallVector<unsigned>> &offsets,
                               SmallVector<size_t> &bucket, int64_t stride,
                               int64_t rowStart, Type scatterTy) {
      for (size_t idx : bucket) {
        int64_t rowOff = offsets[idx][0];
        int64_t colOff = offsets[idx][1];
        Value actualRow = arith::AddIOp::create(
            rewriter, loc, baseRow,
            arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
        Value inStrip = arith::AndIOp::create(
            rewriter, loc,
            arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
            arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::ult, actualRow,
                arith::ConstantIntOp::create(rewriter, loc, rowStart + 8, 32)));
        Value sIdx =
            stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride, rowStart);
        Value safeIdx =
            arith::SelectOp::create(rewriter, loc, inStrip, sIdx, garbIdx);
        Value val = elems[idx];
        if (scatterTy == f32Ty) {
          if (val.getType() != f32Ty)
            val = toF32(rewriter, loc, val, f32Ty);
        } else {
          val = toMmaInputType(rewriter, loc, val, scatterTy);
        }
        Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, scatterTy, ptr,
                                        ArrayRef<LLVM::GEPArg>{safeIdx});
        LLVM::StoreOp::create(rewriter, loc, val, gep);
      }
    };

    // ── Phase 1: Load C tiles (filtered strip scatter via TG) ──────────
    SmallVector<SmallVector<Value>> matC_tiles(tilesM);
    for (int64_t tm = 0; tm < tilesM; ++tm) {
      matC_tiles[tm].resize(tilesN);
      int64_t rowStart = tm * 8;

      filteredScatter(ptrTG, garbageIdx, cBaseRow, cBaseCol, elemsC, cOffsets,
                      cBuckets[tm], Npad, rowStart, f32Ty);
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      for (int64_t tn = 0; tn < tilesN; ++tn) {
        Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
        matC_tiles[tm][tn] = emitSGLoad(loadFn, ptrTG, Npad, Npad, cOff);
      }
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});
    }

    // ── Phase 2: A/B loads + MMA ──────────────────────────────────
    if (useDeviceA && useDeviceB) {
      // ── DEVICE PATH: Direct MMA loads from device memory ──────────
      // No TG scatter, no barriers for A/B.
      // Prefetch: A tiles for tk+1 loaded before MMA of tk.
      Value aDevStride = makeDevMmaStride(aColStride, aRowStride);
      Value bDevStride = makeDevMmaStride(bColStride, bRowStride);
      Value mmaShape = makeI64Vec2(rewriter, loc, 8, 8);
      Value zeroOff = makeI64Vec2(rewriter, loc, 0, 0);
      // Per-operand canonical transpose flag (i1 false for the 3-vector path).
      Value aDevTranspose = makeDevMmaTranspose(aColStride, aRowStride);
      Value bDevTranspose = makeDevMmaTranspose(bColStride, bRowStride);

      // Prologue: load A tiles for tk=0
      SmallVector<Value> matA_cur(tilesM);
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        Value aTilePtr =
            computeTileDevPtr(aPtrs, aOffsets, aRowStride, aColStride, aBaseRow,
                              aBaseCol, tm * 8, 0);
        matA_cur[tm] = emitDevSGLoad(devLoadFn, aTilePtr, mmaShape, aDevStride,
                                     zeroOff, aDevTranspose);
      }

      for (int64_t tk = 0; tk < tilesK; ++tk) {
        // Prefetch A tiles for tk+1
        SmallVector<Value> matA_next(tilesM);
        if (tk + 1 < tilesK) {
          for (int64_t tm = 0; tm < tilesM; ++tm) {
            Value aTilePtr =
                computeTileDevPtr(aPtrs, aOffsets, aRowStride, aColStride,
                                  aBaseRow, aBaseCol, tm * 8, (tk + 1) * 8);
            matA_next[tm] = emitDevSGLoad(devLoadFn, aTilePtr, mmaShape,
                                          aDevStride, zeroOff, aDevTranspose);
          }
        }

        // Load B tiles and accumulate
        for (int64_t tn = 0; tn < tilesN; ++tn) {
          Value bTilePtr =
              computeTileDevPtr(bPtrs, bOffsets, bRowStride, bColStride,
                                bBaseRow, bBaseCol, tk * 8, tn * 8);
          Value matB = emitDevSGLoad(devLoadFn, bTilePtr, mmaShape, bDevStride,
                                     zeroOff, bDevTranspose);

          for (int64_t tm = 0; tm < tilesM; ++tm) {
            matC_tiles[tm][tn] =
                LLVM::CallOp::create(
                    rewriter, loc, devMmaFn,
                    ValueRange{matA_cur[tm], matB, matC_tiles[tm][tn]})
                    .getResult();
          }
        }

        if (tk + 1 < tilesK)
          matA_cur = matA_next;
      }
    } else {
      // ── TG PATH: Original scatter/load through threadgroup memory ──
      int64_t aMaxBase = maxBaseRow(aSrcEnc);
      int64_t bMaxBase = maxBaseRow(bSrcEnc);
      auto aBuckets = bucketElements(aOffsets, aMaxBase, tilesM, 0);
      auto bBuckets = bucketElements(bOffsets, bMaxBase, tilesK, 0);

      for (int64_t tk = 0; tk < tilesK; ++tk) {
        SmallVector<Value> matA_strip(tilesM);
        {
          for (int64_t tm = 0; tm < tilesM; ++tm) {
            int64_t rowStart = tm * 8;
            filteredScatter(ptrTG, garbageIdx, aBaseRow, aBaseCol, elemsA,
                            aOffsets, aBuckets[tm], Kpad, rowStart,
                            abTgScatterTy);
            LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                                 ValueRange{fenceTG, execMod});

            Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
            matA_strip[tm] = emitSGLoad(abTgLoadFn, ptrTG, Kpad, Kpad, aOff);
            LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                                 ValueRange{fenceTG, execMod});
          }
        }

        {
          int64_t rowStart = tk * 8;
          filteredScatter(ptrTG, garbageIdx, bBaseRow, bBaseCol, elemsB,
                          bOffsets, bBuckets[tk], Npad, rowStart,
                          abTgScatterTy);
          LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                               ValueRange{fenceTG, execMod});

          for (int64_t tn = 0; tn < tilesN; ++tn) {
            Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
            Value matB = emitSGLoad(abTgLoadFn, ptrTG, Npad, Npad, bOff);

            for (int64_t tm = 0; tm < tilesM; ++tm) {
              matC_tiles[tm][tn] =
                  LLVM::CallOp::create(
                      rewriter, loc, abTgMmaFn,
                      ValueRange{matA_strip[tm], matB, matC_tiles[tm][tn]})
                      .getResult();
            }
          }
          LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                               ValueRange{fenceTG, execMod});
        }
      }
    }

    // ── Phase 4: Extract C results from MMA tiles ────────────────────
    auto outElemTy = cType.getElementType();
    SmallVector<Value> resultElems(elemsC.size());
    for (size_t i = 0; i < elemsC.size(); ++i)
      resultElems[i] = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(outElemTy));

    if (useDeviceA && useDeviceB) {
      // ── DEVICE PATH: Shuffle-based MMA extract (no TG round-trip) ───
      //
      // The MMA hardware register layout differs from the logical layout
      // defined by toLinearLayout:
      //
      // PHYSICAL layout (extractelement indices 0, 1):
      //   phys_row = L[1] | (L[2]<<1) | (L[4]<<2)
      //   phys_col = (L[0]<<1) | (L[3]<<2) | R   (R = extract index 0 or 1)
      //
      // LOGICAL layout (toLinearLayout / cOffsets):
      //   log_row = L[3] | (L[4]<<1)   + (logReg ? 4 : 0)
      //   log_col = L[0] | (L[1]<<1) | (L[2]<<2)
      //
      // To extract the value at logical position (log_row, log_col) from
      // the MMA tile, we need to:
      //   1. Find source lane S that physically holds (log_row, log_col):
      //      S = log_col[1] | (log_row[0]<<1) | (log_row[1]<<2) |
      //      (log_col[2]<<3) | (log_row[2]<<4) In terms of lane bits:
      //        S = L[1] | (L[3]<<1) | (L[4]<<2) | (L[2]<<3)  (+ 16 for
      //        logReg=1)
      //   2. Physical register R = log_col[0] = L[0] = laneId & 1
      //   3. simd_shuffle(phys_reg_R, S) gives the value

      auto cWpc = cMmaEnc.getWarpsPerCTA();
      unsigned wN = cWpc[1];

      // Compute warpRow/warpCol for C tile mapping (runtime i32)
      Value cWarpRow = divByConst(rewriter, loc, warpId, wN);
      Value cWarpCol = remByConst(rewriter, loc, warpId, wN);

      // Declare shuffle intrinsic: air.simd_shuffle.f32(float, i16) -> float
      auto i16Ty = IntegerType::get(ctx, 16);
      auto shuffleFn = getOrInsertIntrinsic(
          rewriter, mod, "air.simd_shuffle.f32",
          LLVMFunctionType::get(f32Ty, {f32Ty, i16Ty}, false));

      // Compute shuffle source lane for logical reg 0:
      // S0 = L[1] | (L[3]<<1) | (L[4]<<2) | (L[2]<<3)
      // Bit extraction from laneId (i32):
      auto extractBit = [&](Value v, int bit) -> Value {
        Value shifted =
            (bit > 0) ? arith::ShRUIOp::create(rewriter, loc, v,
                                               arith::ConstantIntOp::create(
                                                   rewriter, loc, bit, 32))
                      : v;
        return arith::AndIOp::create(
            rewriter, loc, shifted,
            arith::ConstantIntOp::create(rewriter, loc, 1, 32));
      };

      // S0 = L[1] | (L[3]<<1) | (L[4]<<2) | (L[2]<<3)
      Value bit1 = extractBit(laneId, 1);
      Value bit2 = extractBit(laneId, 2);
      Value bit3 = extractBit(laneId, 3);
      Value bit4 = extractBit(laneId, 4);

      Value S0 = bit1; // L[1] at position 0
      S0 = arith::OrIOp::create(
          rewriter, loc, S0,
          arith::ShLIOp::create(
              rewriter, loc, bit3,
              arith::ConstantIntOp::create(rewriter, loc, 1,
                                           32))); // L[3] at position 1
      S0 = arith::OrIOp::create(
          rewriter, loc, S0,
          arith::ShLIOp::create(
              rewriter, loc, bit4,
              arith::ConstantIntOp::create(rewriter, loc, 2,
                                           32))); // L[4] at position 2
      S0 = arith::OrIOp::create(
          rewriter, loc, S0,
          arith::ShLIOp::create(
              rewriter, loc, bit2,
              arith::ConstantIntOp::create(rewriter, loc, 3,
                                           32))); // L[2] at position 3

      // S1 = S0 | 16 (for logical register 1, target_row has bit 2 set)
      Value S1 = arith::OrIOp::create(
          rewriter, loc, S0,
          arith::ConstantIntOp::create(rewriter, loc, 16, 32));

      Value S0_i16 = arith::TruncIOp::create(rewriter, loc, i16Ty, S0);
      Value S1_i16 = arith::TruncIOp::create(rewriter, loc, i16Ty, S1);

      // Physical register selector: R = laneId & 1
      Value physReg = arith::AndIOp::create(
          rewriter, loc, laneId,
          arith::ConstantIntOp::create(rewriter, loc, 1, 32));
      Value isOddCol = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, physReg,
          arith::ConstantIntOp::create(rewriter, loc, 0, 32));

      // Pre-extract physical registers and shuffle for ALL tiles
      // For each tile, compute 4 shuffled values:
      //   shufReg0_S0, shufReg1_S0 (for logical reg 0)
      //   shufReg0_S1, shufReg1_S1 (for logical reg 1)
      // Then select based on isOddCol:
      //   logReg0_val = isOddCol ? shufReg1_S0 : shufReg0_S0
      //   logReg1_val = isOddCol ? shufReg1_S1 : shufReg0_S1

      Value extractIdx0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value extractIdx1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);

      // Pre-compute shuffled logical values per tile: logReg0[flat],
      // logReg1[flat]
      SmallVector<Value> tileLogReg0(tilesM * tilesN);
      SmallVector<Value> tileLogReg1(tilesM * tilesN);
      for (int64_t tm = 0; tm < tilesM; ++tm)
        for (int64_t tn = 0; tn < tilesN; ++tn) {
          int64_t flat = tm * tilesN + tn;
          Value pr0 = LLVM::ExtractElementOp::create(
              rewriter, loc, f32Ty, matC_tiles[tm][tn], extractIdx0);
          Value pr1 = LLVM::ExtractElementOp::create(
              rewriter, loc, f32Ty, matC_tiles[tm][tn], extractIdx1);

          // Shuffle for logical reg 0 (S0)
          Value shR0_S0 = LLVM::CallOp::create(rewriter, loc, shuffleFn,
                                               ValueRange{pr0, S0_i16})
                              .getResult();
          Value shR1_S0 = LLVM::CallOp::create(rewriter, loc, shuffleFn,
                                               ValueRange{pr1, S0_i16})
                              .getResult();
          tileLogReg0[flat] = arith::SelectOp::create(rewriter, loc, isOddCol,
                                                      shR1_S0, shR0_S0);

          // Shuffle for logical reg 1 (S1)
          Value shR0_S1 = LLVM::CallOp::create(rewriter, loc, shuffleFn,
                                               ValueRange{pr0, S1_i16})
                              .getResult();
          Value shR1_S1 = LLVM::CallOp::create(rewriter, loc, shuffleFn,
                                               ValueRange{pr1, S1_i16})
                              .getResult();
          tileLogReg1[flat] = arith::SelectOp::create(rewriter, loc, isOddCol,
                                                      shR1_S1, shR0_S1);
        }

      for (size_t idx = 0; idx < cOffsets.size(); ++idx) {
        int64_t rowOff = cOffsets[idx][0];
        int64_t colOff = cOffsets[idx][1];
        int64_t localTm = rowOff / 8;
        int64_t localTn = colOff / 8;
        int64_t logRegIdx = (rowOff % 8 >= 4) ? 1 : 0;

        // Absolute tile index (runtime)
        // absoluteTm = warpRow + rowOff/8, absoluteTn = warpCol + colOff/8
        // This works because cBaseRow = warpRow*8 + laneRow (laneRow < 4),
        // so (cBaseRow + rowOff) / 8 = warpRow + rowOff/8 (since laneRow +
        // rowOff%8 < 8).
        Value absTm = arith::AddIOp::create(
            rewriter, loc, cWarpRow,
            arith::ConstantIntOp::create(rewriter, loc, localTm, 32));
        Value absTn = arith::AddIOp::create(
            rewriter, loc, cWarpCol,
            arith::ConstantIntOp::create(rewriter, loc, localTn, 32));
        Value flatTileIdx = arith::AddIOp::create(
            rewriter, loc,
            arith::MulIOp::create(
                rewriter, loc, absTm,
                arith::ConstantIntOp::create(rewriter, loc, tilesN, 32)),
            absTn);

        // Select chain: pick the correct tile's shuffled value
        auto &logRegVals = (logRegIdx == 0) ? tileLogReg0 : tileLogReg1;
        Value val = logRegVals[0]; // default
        for (int64_t t = (int64_t)logRegVals.size() - 1; t >= 0; --t) {
          Value match = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, flatTileIdx,
              arith::ConstantIntOp::create(rewriter, loc, t, 32));
          val =
              arith::SelectOp::create(rewriter, loc, match, logRegVals[t], val);
        }

        if (val.getType() != outElemTy)
          val = fromF32(rewriter, loc, val, outElemTy);
        resultElems[idx] = val;
      }
    } else {
      // ── TG PATH: Store C tiles -> TG, gather (original) ─────────
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        int64_t rowStart = tm * 8;

        for (int64_t tn = 0; tn < tilesN; ++tn) {
          Value cOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
          emitSGStore(storeFn, matC_tiles[tm][tn], ptrTG, Npad, Npad, cOff);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});

        for (size_t idx : cBuckets[tm]) {
          int64_t rowOff = cOffsets[idx][0];
          int64_t colOff = cOffsets[idx][1];
          Value actualRow = arith::AddIOp::create(
              rewriter, loc, cBaseRow,
              arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
          Value inStrip = arith::AndIOp::create(
              rewriter, loc,
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
              arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                    actualRow,
                                    arith::ConstantIntOp::create(
                                        rewriter, loc, rowStart + 8, 32)));
          Value sIdx =
              stripFlatIdx(cBaseRow, cBaseCol, rowOff, colOff, Npad, rowStart);
          Value safeIdx =
              arith::SelectOp::create(rewriter, loc, inStrip, sIdx, garbageIdx);
          Value val = gather1(ptrTG, safeIdx);
          if (val.getType() != outElemTy)
            val = fromF32(rewriter, loc, val, outElemTy);
          resultElems[idx] = arith::SelectOp::create(rewriter, loc, inStrip,
                                                     val, resultElems[idx]);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
      }
    }

    // ── Pack result ───────────────────────────────────────────────────
    auto outLLVMTy = getTypeConverter()->convertType(cType);
    if (!outLLVMTy)
      return failure();

    if (auto outStructTy = dyn_cast<LLVMStructType>(outLLVMTy)) {
      Value result = UndefOp::create(rewriter, loc, outStructTy);
      for (size_t i = 0; i < resultElems.size(); ++i)
        result = InsertValueOp::create(rewriter, loc, outStructTy, result,
                                       resultElems[i],
                                       ArrayRef<int64_t>{(int64_t)i});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, resultElems[0]);
    }
    return success();
  }
};

} // anonymous namespace

namespace mlir::triton::applegpu {

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit) {
  patterns.add<DotOpBlockedConversion>(typeConverter, benefit);
  patterns.add<DotOpAppleMmaConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::applegpu
