// DotOpBlockedConversion: lower tt.dot with blocked encoding on C
// (batch-aware).
#include "Dialect/TritonAppleGPU/IR/AppleMmaFragment.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "DotCommon.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "TritonAppleGPUTransforms/Passes.h"
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
using namespace mlir::triton::applegpu::dotcommon;

namespace {

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
// Alias a dot's TG scatter buffer into the module-wide global_smem arena
// (created by ConvertTritonAppleGPUToLLVM from the allocate-shared-memory
// pass) instead of allocating a fresh __tg_dot_ab global on top of it. The
// convert_layout scratch held in global_smem is live only before and after a
// dot (q/k operand converts feed the dot in registers; the output convert runs
// after), so the dot's transient scatter buffer can safely overlap it. Sharing
// the arena keeps the addrspace(3) footprint at max(convert, dot) instead of
// their sum, which is what overflowed the 32KB cap for batched int8 dot3d. If
// the dot needs more bytes than global_smem currently holds, grow it (and the
// ttg.shared attribute the convert budgeter reads). Returns a addrspace(3)
// pointer to the arena, or null if global_smem is unavailable.
static Value getOrGrowSharedArena(ConversionPatternRewriter &rewriter,
                                  Location loc, ModuleOp mod,
                                  int64_t neededBytes) {
  auto ctx = mod.getContext();
  auto g = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
  if (!g)
    return Value();
  auto arrTy = dyn_cast<LLVMArrayType>(g.getGlobalType());
  if (!arrTy)
    return Value();
  int64_t haveBytes = (int64_t)arrTy.getNumElements() *
                      (arrTy.getElementType().getIntOrFloatBitWidth() / 8);
  if (haveBytes < neededBytes) {
    auto i8Ty = IntegerType::get(ctx, 8);
    g.setGlobalTypeAttr(TypeAttr::get(LLVMArrayType::get(i8Ty, neededBytes)));
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared")) {
      if (attr.getValue().getZExtValue() < (uint64_t)neededBytes)
        mod->setAttr("ttg.shared",
                     IntegerAttr::get(IntegerType::get(ctx, 64), neededBytes));
    } else {
      mod->setAttr("ttg.shared",
                   IntegerAttr::get(IntegerType::get(ctx, 64), neededBytes));
    }
  }
  return LLVM::AddressOfOp::create(rewriter, loc, LLVMPointerType::get(ctx, 3),
                                   g.getName());
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
      // A masked load (tt.load %ptr, %mask, %other) zero-fills (or `other`-
      // fills) the out-of-bounds lanes in registers. The device-direct
      // simdgroup_matrix_8x8_load.p1 reads raw device memory and so CANNOT see
      // that masking - it would feed the MMA the unmasked device bytes for the
      // masked-out positions, corrupting the result (100% mismatch on masked
      // tl.dot, e.g. depthwise / grouped conv2d_backward). Decline the device
      // path so the operand goes through the TG scatter, which scatters the
      // already-masked register values and preserves the zero fill.
      if (loadOp.getMask())
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

    // Async-copy enable decision (consolidated). Default on; the device->TG
    // async copy is disabled when either of these hard correctness conditions
    // holds, falling back to the proven per-element scatter:
    //   - integer element type: async copy writes raw bytes to TG while the
    //     MMA loads read as f32, so the int8 byte width mismatch corrupts data.
    //   - matWarpsCEarly > 1 without a provably disjoint row partition of the
    //     8-row strip: identical whole-strip copies from multiple simdgroups
    //     write the same TG bytes, which races on Apple GPUs and corrupts
    //     data. With matWarpsCEarly in {2,4,8} each warp copies its own
    //     8/nWarps-row band (disjoint bytes), so the copies cannot race; the
    //     existing post-stage barrier publishes the strip across warps.
    // Whether a given operand actually uses async copy is gated further below
    // (batch==1, ptr/elem counts match, row-major encoding, computable
    // strides).
    int64_t dmaPartitionWarps =
        (matWarpsCEarly == 2 || matWarpsCEarly == 4 || matWarpsCEarly == 8)
            ? matWarpsCEarly
            : 1;
    int64_t dmaBandRows = 8 / dmaPartitionWarps;
    bool asyncCopyEnabled =
        !isa<IntegerType>(aElemTy) &&
        (matWarpsCEarly <= 1 || dmaPartitionWarps == matWarpsCEarly);

    bool useAsyncA = false, useAsyncB = false;
    Value aRowStride, bRowStride, aColStride, bColStride;
    auto gateAsyncOperand = [&](SmallVector<Value> &ptrs, int64_t elemCount,
                                SmallVector<SmallVector<unsigned>> &offsets,
                                Type elemTy, ttg::BlockedEncodingAttr srcEnc,
                                Value &rowStride, Value &colStride) -> bool {
      if (!asyncCopyEnabled || batchSize != 1 ||
          (int64_t)ptrs.size() != elemCount || ptrs.empty() ||
          !isRowMajorEncoding(srcEnc))
        return false;
      rowStride = computeRowStrideBlocked(ptrs, offsets, elemTy, srcEnc);
      colStride = computeColStrideBlocked(ptrs, offsets, elemTy, srcEnc);
      return rowStride != nullptr && colStride != nullptr;
    };
    useAsyncA = gateAsyncOperand(aPtrs, (int64_t)elemsA.size(), aOffsets,
                                 aType.getElementType(), aSrcEnc, aRowStride,
                                 aColStride);
    useAsyncB = gateAsyncOperand(bPtrs, (int64_t)elemsB.size(), bOffsets,
                                 bType.getElementType(), bSrcEnc, bRowStride,
                                 bColStride);

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
    // load+compute (one simdgroup).
    bool useDoubleBufB = useAsyncB && (tilesKEarly > 1) &&
                         (2 * bStripBytes <= 16384) && (batchSize == 1) &&
                         (matWarpsCEarly > 1);

    // Each batch slice needs its own TG region so MMA ops don't
    // cross-contaminate between warps assigned to different batches.
    // With double-buffering, Phase 3 needs 2 B strip slots.
    // The TG buffer must fit the max of: Phase 1 (A strip), Phase 2 (C strip),
    // Phase 3 (1 or 2 B strips), Phase 4 (C strip).
    int64_t phase3Strips = useDoubleBufB ? 2 : 1;
    int64_t tgSizeNeeded = std::max(tgStripSize, phase3Strips * 8 * Npad);
    int64_t tgSize = tgSizeNeeded * batchSize;
    // Prefer aliasing the scatter buffer into the shared global_smem arena so
    // the dot's transient TG footprint overlaps the convert_layout scratch
    // instead of stacking on top of it (the batched int8 dot3d 32KB overflow).
    // Restrict the alias to integer-element dots: those are the OOR cases, and
    // an int8 matmul kernel's only other threadgroup consumer is its own
    // #mma/#blocked convert (a SEPARATE __tg_cvt pool), so global_smem carries
    // only this dot's f32 GEPs and the Metal typed-pointer reader stays
    // consistent. Float dots can share global_smem with reductions/standard
    // converts that GEP it at a different element type; mixing element-typed
    // GEPs on one typed global trips the metallib reader, so they keep their
    // own typed __tg_dot_ab global.
    Value ptrTG;
    if (isa<IntegerType>(aElemTy))
      ptrTG = getOrGrowSharedArena(rewriter, loc, mod, tgSize * 4);
    if (!ptrTG) {
      auto tgBuf = getOrCreateTGGlobal(
          rewriter, mod, ("__tg_dot_ab_" + llvm::Twine(id)).str(), tgSize);
      ptrTG =
          LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgBuf.getName());
    }

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
      int64_t bandRows = 8;

      // Partitioned multi-warp DMA: warp w copies rows
      // [w*dmaBandRows, (w+1)*dmaBandRows) of the 8-row strip. Bands are
      // byte-disjoint, so concurrent per-simdgroup copies cannot overlap.
      if (dmaPartitionWarps > 1) {
        bandRows = dmaBandRows;
        Value warp64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, warpId);
        Value bandRow = arith::MulIOp::create(
            rewriter, loc, warp64,
            arith::ConstantIntOp::create(rewriter, loc, dmaBandRows, 64));
        Value srcOff =
            arith::MulIOp::create(rewriter, loc, bandRow, rowStrideElems);
        stripDevPtr =
            LLVM::GEPOp::create(rewriter, loc, devPtrTy, elemTy, stripDevPtr,
                                ArrayRef<LLVM::GEPArg>{srcOff});
        Value dstOff = arith::MulIOp::create(
            rewriter, loc, bandRow,
            arith::ConstantIntOp::create(rewriter, loc, tgPadStride, 64));
        tgDst = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, elemTy, tgDst,
                                    ArrayRef<LLVM::GEPArg>{dstOff});
      }

      Value sizeOf = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value alignOf = arith::ConstantIntOp::create(rewriter, loc, 1, 64);

      // Destination (TG): packed with padding
      Value dstStrideBytes = arith::ConstantIntOp::create(
          rewriter, loc, tgPadStride * elemBytes, 64);
      Value dstElemStride = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value dstTile = makeI64Vec2(rewriter, loc, tileWidthBytes, bandRows);

      // Source (device): stride in bytes
      Value elemBytesVal =
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)elemBytes, 64);
      Value srcStrideBytes =
          arith::MulIOp::create(rewriter, loc, rowStrideElems, elemBytesVal);
      Value srcElemStride = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value srcTile = makeI64Vec2(rewriter, loc, tileWidthBytes, bandRows);

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
    // stripBlock places this strip into its own 8-row TG block (block*8 rows),
    // so multiple strips can coexist in the buffer under one barrier pair.
    auto stripFlatIdx = [&](Value baseRow, Value baseCol, int64_t rowOff,
                            int64_t colOff, int64_t stride,
                            int64_t stripRowStart,
                            int64_t stripBlock = 0) -> Value {
      Value row32 = arith::AddIOp::create(
          rewriter, loc, baseRow,
          arith::ConstantIntOp::create(
              rewriter, loc, rowOff - stripRowStart + stripBlock * 8, 32));
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

    // Out-of-strip gather sink: a valid in-bounds slot (0). Gather results from
    // this slot are always discarded by the inStrip select, and scatter stores
    // are predicated so out-of-strip lanes never write here.
    Value garbageIdx = arith::ConstantIntOp::create(rewriter, loc, 0, 64);

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
      // The store predicate (strip-row check, plus optional batch-warp match)
      // depends only on (eb, rowOff), so all elements sharing that key share an
      // identical guard. Group the elements by (eb, rowOff) and emit a single
      // conditional block per group: one cond_br per distinct strip row instead
      // of per element, so the block count stays proportional to strip rows
      // rather than the full tile.
      std::map<std::pair<int64_t, int64_t>, SmallVector<size_t>> groups;
      SmallVector<std::pair<int64_t, int64_t>> groupOrder;
      for (size_t i = 0; i < elems.size(); ++i) {
        int64_t eb = (rowDim > 0) ? elemBatchIndex(offsets, i) : 0;
        if (curBatchRound >= 0 && rowDim > 0 && !opHasBatchWarps) {
          if (eb != curBatchRound)
            continue;
        }
        int64_t rowOff = offsets[i][rowDim];
        auto key = std::make_pair(eb, rowOff);
        if (groups.find(key) == groups.end())
          groupOrder.push_back(key);
        groups[key].push_back(i);
      }

      for (auto &key : groupOrder) {
        int64_t eb = key.first;
        int64_t rowOff = key.second;
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

        auto *curBlock = rewriter.getInsertionBlock();
        auto curPoint = rewriter.getInsertionPoint();
        auto *endBlock = curBlock->splitBlock(curPoint);
        auto *thenBlock = rewriter.createBlock(endBlock);
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, inStrip, thenBlock, endBlock);
        rewriter.setInsertionPointToEnd(thenBlock);

        for (size_t i : groups[key]) {
          int64_t colOff = offsets[i][colDim];
          Value idx =
              stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride, rowStart);
          Value val = (scatterTy == f32Ty)
                          ? toF32(rewriter, loc, elems[i], f32Ty)
                          : toMmaInputType(rewriter, loc, elems[i], scatterTy);
          Value storeIdx;
          if (curBatchRound >= 0) {
            storeIdx = idx;
          } else {
            Value batchOff = elemBatchTGOffset(offsets, i, mixed);
            storeIdx = arith::AddIOp::create(rewriter, loc, idx, batchOff);
          }
          Value gep =
              LLVM::GEPOp::create(rewriter, loc, tgPtrTy, scatterTy, ptrTG,
                                  ArrayRef<LLVM::GEPArg>{storeIdx});
          LLVM::StoreOp::create(rewriter, loc, val, gep);
        }
        LLVM::BrOp::create(rewriter, loc, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
      }
    };

    // ── Initialize result to zero ────────────────────────────────────
    auto outElemTy = cType.getElementType();
    SmallVector<Value> resultElems(cOffsets.size());
    for (size_t i = 0; i < cOffsets.size(); ++i)
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

        // Gather: each thread reads its C elements from TG, in the C
        // accumulator layout (cOffsets, cBaseRow/cBaseCol).
        for (size_t i = 0; i < cOffsets.size(); ++i) {
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

    Value result;
    if (auto outStructTy = dyn_cast<LLVMStructType>(outLLVMTy)) {
      result = UndefOp::create(rewriter, loc, outStructTy);
      for (size_t i = 0; i < resultElems.size(); ++i)
        result = InsertValueOp::create(rewriter, loc, outStructTy, result,
                                       resultElems[i],
                                       ArrayRef<int64_t>{(int64_t)i});
    } else {
      result = resultElems[0];
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // anonymous namespace

namespace mlir::triton::applegpu {

void populateDotOpBlockedPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit) {
  patterns.add<DotOpBlockedConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::applegpu
