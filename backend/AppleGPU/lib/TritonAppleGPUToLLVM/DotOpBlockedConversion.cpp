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

// Morton (Z-order) deinterleaving for warp grid mapping: even bits → row, odd
// bits → col, giving cache-local Z-curve traversal of the output tile grid.
// Even variant compacts bits 0,2,4,...; odd variant compacts bits 1,3,5,...
static Value mortonDeinterleaveEven(OpBuilder &b, Location loc, Value id,
                                    unsigned numBits) {
  Value result = arith::ConstantIntOp::create(b, loc, 0, 32);
  for (unsigned i = 0; i < numBits; ++i) {
    unsigned srcBit = i * 2;
    Value shifted =
        (srcBit > 0)
            ? arith::ShRUIOp::create(
                  b, loc, id, arith::ConstantIntOp::create(b, loc, srcBit, 32))
            : id;
    Value bit = arith::AndIOp::create(
        b, loc, shifted, arith::ConstantIntOp::create(b, loc, 1, 32));
    if (i > 0)
      bit = arith::ShLIOp::create(b, loc, bit,
                                  arith::ConstantIntOp::create(b, loc, i, 32));
    result = arith::OrIOp::create(b, loc, result, bit);
  }
  return result;
}

static Value mortonDeinterleaveOdd(OpBuilder &b, Location loc, Value id,
                                   unsigned numBits) {
  Value result = arith::ConstantIntOp::create(b, loc, 0, 32);
  for (unsigned i = 0; i < numBits; ++i) {
    unsigned srcBit = i * 2 + 1;
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

// Morton needs an equal-power-of-2 (square) warp grid: non-square grids don't
// deinterleave to a bijection (e.g. 1x4 yields only 2 distinct cols → warp
// collisions). Returns bits/dim (log2), or 0 if Morton isn't applicable.
static unsigned mortonBitsPerDim(int64_t wM, int64_t wN) {
  if (wM != wN)
    return 0;
  if (!isPowerOf2(wM))
    return 0;
  if (wM <= 1)
    return 0;
  return log2(wM);
}
// Alias a dot's TG scatter buffer into global_smem rather than stacking a fresh
// __tg_dot_ab global (which overflowed the 32KB cap for batched int8 dot3d).
// The convert_layout scratch there is dead during the dot, so overlap is safe.
// Grows global_smem (and the ttg.shared attr) if needed. Null if unavailable.
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
// DotOpBlockedConversion: blocked encoding on C, any rank (batch-aware).
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
    // simdgroup load/store arg-type lists for the selected target ABI.
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

    // simdgroup load/store emitters: pick the canonical (macOS<=15) or 3-vector
    // (macOS>=16) arg list. shapeDim = first lane of the 3-vector shape; pitch
    // = row pitch (3-vector stride lane1 / canonical elements_per_row).
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
      // The device-direct simdgroup load reads raw memory and can't see a
      // load's register-side mask fill, so decline the device path for masked
      // loads and route through TG scatter (else masked tl.dot miscompiles).
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

    // Compute row stride from pointer differences (shuffle-based, as in the MMA
    // conversion).
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

    // Async copy requires row-major layout: the DMA copies contiguous bytes per
    // row, which async_copy_2d can't do for column-major data. Heuristic:
    // col dim fastest-varying (order[0]==colDim) ⇒ likely row-major.
    auto isRowMajorEncoding = [&](ttg::BlockedEncodingAttr enc) -> bool {
      auto order = enc.getOrder();
      unsigned encColDim = enc.getSizePerThread().size() - 1;
      return order[0] == (unsigned)encColDim;
    };

    // Warps assigned to the MxN tile. >1 means multiple simdgroups share the
    // same TG strip, so concurrent async copies can race on the TG region.
    auto cWpcEarly = cEnc.getWarpsPerCTA();
    int64_t matWarpsCEarly = cWpcEarly[rowDim] * cWpcEarly[colDim];

    // Async copy is default-on but disabled (-> per-element scatter) for
    // correctness when either: integer element type (async writes raw bytes but
    // MMA loads read f32, corrupting int8); or matWarpsCEarly>1 without a
    // disjoint row partition (multi-simdgroup whole-strip copies race). For
    // matWarpsCEarly in {2,4,8} each warp copies its own band, so no race.
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

    // Runtime thread base position. For 3D+ tensors only the last two dims feed
    // the MMA row/col base; batch dims use compile-time offset matching.
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

      // Strip the batch warp component from warpId; the row/col decomposition
      // uses only the row/col warps.
      int64_t batchWarps = 1;
      for (unsigned d = 0; d < encRowDim; ++d)
        batchWarps *= wpc[d];
      Value matWarpId = warpId;
      if (batchWarps > 1) {
        int64_t matWarps = wM * wN;
        matWarpId = remByConst(rewriter, loc, warpId, matWarps);
      }

      int64_t batchLanes = 1;
      for (unsigned d = 0; d < encRowDim; ++d)
        batchLanes *= tpw[d];
      Value matLaneId = laneId;
      if (batchLanes > 1) {
        int64_t matLanes = tM * tN;
        matLaneId = remByConst(rewriter, loc, laneId, matLanes);
      }

      // order[0] is the fastest-varying dim index.
      bool colFastest = (order[0] == (unsigned)encColDim);

      Value tMsM = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
      Value sM_val = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
      Value tNsN = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
      Value sN_val = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

      // Warp decomposition: Morton Z-order when both dims are power-of-2, else
      // linear div/mod. colFastest ⇒ even bits=row, odd=col (swapped
      // otherwise).
      Value wR, wC;
      unsigned mortonBits = mortonBitsPerDim(wM, wN);
      if (mortonBits > 0) {
        if (colFastest) {
          wR = mortonDeinterleaveEven(rewriter, loc, matWarpId, mortonBits);
          wC = mortonDeinterleaveOdd(rewriter, loc, matWarpId, mortonBits);
        } else {
          wC = mortonDeinterleaveEven(rewriter, loc, matWarpId, mortonBits);
          wR = mortonDeinterleaveOdd(rewriter, loc, matWarpId, mortonBits);
        }
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

      // Wrap redundant threads (tileM > rows).
      if (tileM > rows)
        baseRow = remByConst(rewriter, loc, baseRow, rows);
      if (tileN > cols)
        baseCol = remByConst(rewriter, loc, baseCol, cols);

      return {baseRow, baseCol};
    };

    auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
    auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
    auto [cBaseRow, cBaseCol] = makeBase(cEnc, M, N);

    // DotOperandEncoding (Path 2): K is fully replicated per thread and the
    // offsets already span the full K range, so the K-dim base must be 0 or
    // base+offset overshoots K.
    Value zero32 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    if (aDotOpIdx == 0) {
      aBaseCol = zero32;
    }
    if (bDotOpIdx == 1) {
      bBaseRow = zero32;
    }

    unsigned id = getDotCounter(ctx)++;
    // Padded strides for bank-conflict avoidance, only when the padded buffer
    // fits in 16KB (headroom below the 32KB TG limit).
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

    // Double-buffer B strips in Phase 3 (overlap DMA with MMA) only when async
    // B is used, K>8, doubled buffer fits 16KB, and matWarpsCEarly>1 (a single
    // simdgroup can't overlap load+compute).
    int64_t bStripBytes = 8 * Npad * 4;
    int64_t tilesKEarly = K / 8;
    bool useDoubleBufB = useAsyncB && (tilesKEarly > 1) &&
                         (2 * bStripBytes <= 16384) && (batchSize == 1) &&
                         (matWarpsCEarly > 1);

    auto cWpc = cEnc.getWarpsPerCTA();
    int64_t matWarpsC = cWpc[rowDim] * cWpc[colDim];
    int64_t numTotalWarps = 1;
    for (auto w : cWpc)
      numTotalWarps *= w;
    int64_t numBatchWarps = numTotalWarps / matWarpsC;

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

    // Warp-distributed batches need one TG region per simultaneously resident
    // batch slice so MMA ops don't cross-contaminate. Sequential batch rounds
    // reuse the same region for one batch at a time.
    int64_t phase3Strips = useDoubleBufB ? 2 : 1;
    int64_t tgSizeNeeded = std::max(tgStripSize, phase3Strips * 8 * Npad);
    int64_t residentBatchSlices = (batchRounds > 1) ? 1 : batchSize;
    int64_t tgSize = tgSizeNeeded * residentBatchSlices;
    // Alias f32-typed scatter buffers into global_smem when available. This
    // fixes batched int8/f16 dot3d OORs where upstream convert_layout scratch
    // already reserves most/all of the 32KB budget and the dot scratch is live
    // in a disjoint phase. Non-f32 native f16/bf16 dots keep a typed global to
    // avoid mixed element-typed GEPs on the same TG object.
    Value ptrTG;
    if (abTgElemTy == f32Ty)
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

    // Device strip base pointer, uniform across threads. Subtracting this
    // thread's (row,col) offset from ptrs[0] cancels the per-thread terms,
    // leaving matrixBase + stripRowStart*rowStride.
    auto computeStripDevPtr = [&](SmallVector<Value> &ptrs,
                                  SmallVector<SmallVector<unsigned>> &offsets,
                                  Value baseRow, Value baseCol, Value rowStride,
                                  Value colStride, int64_t stripRowStart,
                                  Type elemTy) -> Value {
      int64_t refRowOff = offsets[0][rowDim];
      int64_t refColOff = offsets[0][colDim];

      Value baseRowExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseRow);
      Value baseColExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseCol);

      Value rowDelta = arith::AddIOp::create(
          rewriter, loc, baseRowExt,
          arith::ConstantIntOp::create(rewriter, loc, refRowOff - stripRowStart,
                                       64));
      Value colDelta = arith::AddIOp::create(
          rewriter, loc, baseColExt,
          arith::ConstantIntOp::create(rewriter, loc, refColOff, 64));
      Value elemOff = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(rewriter, loc, rowDelta, rowStride),
          arith::MulIOp::create(rewriter, loc, colDelta, colStride));
      Value negElemOff = arith::SubIOp::create(
          rewriter, loc,
          arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64), elemOff);
      return LLVM::GEPOp::create(rewriter, loc, devPtrTy, elemTy, ptrs[0],
                                 ArrayRef<LLVM::GEPArg>{negElemOff});
    };

    // Fire one 8-row strip async copy device→TG; caller must emitAsyncCopyWait.
    // tgPadStride = padded TG stride (stripCols + padding).
    auto emitAsyncCopyFire = [&](Value stripDevPtr, Value rowStrideElems,
                                 Value tgDst, int64_t stripCols,
                                 int64_t tgPadStride, Type elemTy) {
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      int64_t tileWidthBytes = stripCols * elemBytes;
      int64_t bandRows = 8;

      // Partitioned multi-warp DMA: warp w copies its own band of the 8-row
      // strip. Bands are byte-disjoint, so per-simdgroup copies can't race.
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

      Value dstStrideBytes = arith::ConstantIntOp::create(
          rewriter, loc, tgPadStride * elemBytes, 64);
      Value dstElemStride = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value dstTile = makeI64Vec2(rewriter, loc, tileWidthBytes, bandRows);

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

    // Runtime batch-offset pointer for SIMD matrix load/store. Each batch slice
    // gets its own tgStripSize TG region (no cross-warp contamination).
    // Per-operand batch warp index = warpId / (row*col warpsPerCTA): which
    // batch slice each warp handles.
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

    // Flat TG index within a strip. stripBlock places the strip into its own
    // 8-row TG block so multiple strips coexist under one barrier pair.
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

    // Whether an operand has a runtime batch component (batchWarpIdx
    // contributes). False ⇒ batch lives entirely in compile-time offsets.
    auto hasBatchWarps = [&](ttg::BlockedEncodingAttr enc) -> bool {
      if (rowDim == 0)
        return false;
      auto wpc = enc.getWarpsPerCTA();
      int64_t bw = 1;
      for (unsigned d = 0; d < rowDim; ++d)
        bw *= wpc[d];
      return bw > 1;
    };

    bool aHasBatchWarps = hasBatchWarps(aSrcEnc);
    bool bHasBatchWarps = hasBatchWarps(bSrcEnc);
    bool cHasBatchWarps = hasBatchWarps(cEnc);

    // Whether offsets carry mixed batch values (needed for TG offset routing in
    // warp-distributed mode).
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

    // TG batch offset for an element: mixed-offset operands (dot_op A/B) use
    // the compile-time offset as batch index; uniform ones (C blocked) use the
    // runtime batchTGOffset64 from batchWarpIdx.
    auto elemBatchTGOffset =
        [&](const SmallVector<SmallVector<unsigned>> &offsets, size_t i,
            bool mixed) -> Value {
      if (rowDim == 0)
        return arith::ConstantIntOp::create(rewriter, loc, (int64_t)0, 64);
      if (mixed) {
        int64_t batchOff = 0;
        int64_t stride = 1;
        for (int d = (int)rowDim - 1; d >= 0; --d) {
          batchOff += offsets[i][d] * stride;
          stride *= cType.getShape()[d];
        }
        return arith::ConstantIntOp::create(rewriter, loc,
                                            batchOff * tgStripSize, 64);
      }
      return batchTGOffset64;
    };

    // Flat batch index for element i from compile-time offsets.
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

    // Scatter elements into TG for an 8-row strip. scatterTy = f32 for C,
    // native for A/B. Sequential batch mode (curBatchRound>=0): only the
    // current batch's elements scatter to the TG base. Warp-distributed
    // (curBatchRound<0): all elements scatter, each to its elemBatchTGOffset
    // region.
    auto stripScatter = [&](Value baseRow, Value baseCol,
                            SmallVector<Value> &elems,
                            SmallVector<SmallVector<unsigned>> &offsets,
                            int64_t stride, int64_t rowStart, bool mixed,
                            int64_t curBatchRound, Value operandBatchWarpIdx,
                            bool opHasBatchWarps, Type scatterTy) {
      // The store predicate depends only on (eb, rowOff), so group elements by
      // that key and emit one cond_br per distinct strip row - block count
      // stays proportional to strip rows, not the full tile.
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

        // With batch warps in sequential mode: actual batch = elemBatchIndex +
        // batchWarpIdx, so match batchWarpIdx == curBatchRound -
        // elemBatchIndex.
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

    // MMA computation (batch-aware). numBatchWarps>=batchSize: each warp owns
    // one batch (single pass). numBatchWarps<batchSize: one batch per round,
    // all to the TG base, runtime batch filtering selecting the round's data.

    for (int64_t batchRound = 0; batchRound < batchRounds; ++batchRound) {
      Value curPtrTGBatch = ptrTGBatch;
      int64_t scatterBatchRound = -1; // -1 = warp-distributed (no filter)
      if (batchRounds > 1) {
        curPtrTGBatch = ptrTG;
        scatterBatchRound = batchRound;
      }

      // Phase 1: Load A tiles (8-row strips) - native type MMA
      SmallVector<SmallVector<Value>> matA_tiles(tilesM);
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        matA_tiles[tm].resize(tilesK);
        int64_t rowStart = tm * 8;

        if (useAsyncA) {
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

      // Phase 3: B strips + MMA. Double-buffered path overlaps DMA of B[k+1]
      // with MMA on B[k] across two alternating 8*Npad TG slots.
      if (useDoubleBufB) {
        int64_t slotOffset = 8 * Npad;
        Value ptrSlot0 = curPtrTGBatch;
        Value ptrSlot1 =
            LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty, curPtrTGBatch,
                                ArrayRef<LLVM::GEPArg>{(int64_t)slotOffset});

        // Prologue: load B[0] into slot 0.
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

          // Prefetch B[tk+1] into next slot (fire only).
          if (tk + 1 < tilesK) {
            int64_t nextRowStart = (tk + 1) * 8;
            Value stripPtr = computeStripDevPtr(
                bPtrs, bOffsets, bBaseRow, bBaseCol, bRowStride, bColStride,
                nextRowStart, bType.getElementType());
            emitAsyncCopyFire(stripPtr, bRowStride, nextSlotPtr, N, Npad,
                              bType.getElementType());
          }

          // MMA on current slot (ready from previous barrier).
          for (int64_t tn = 0; tn < tilesN; ++tn) {
            Value bOff = makeI64Vec2(rewriter, loc, tn * 8, 0);
            Value matB = emitSGLoad(abLoadFn, curSlotPtr, Npad, Npad, bOff);

            for (int64_t tm = 0; tm < tilesM; ++tm) {
              Value matA = matA_tiles[tm][tk];
              matC_tiles[tm][tn] =
                  LLVM::CallOp::create(
                      rewriter, loc, abMmaFn,
                      ValueRange{matA, matB, matC_tiles[tm][tn]})
                      .getResult();
            }
          }

          // Wait for prefetch + barrier so the next slot is ready.
          if (tk + 1 < tilesK) {
            emitAsyncCopyWait();
            LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                                 ValueRange{fenceTG, execMod});
          }
        }
      } else {
        // Single-buffer path: scatter/async → barrier → MMA → barrier.
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
              Value matA = matA_tiles[tm][tk];
              matC_tiles[tm][tn] =
                  LLVM::CallOp::create(
                      rewriter, loc, abMmaFn,
                      ValueRange{matA, matB, matC_tiles[tm][tn]})
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

        // Gather each thread's C elements from TG in the accumulator layout.
        for (size_t i = 0; i < cOffsets.size(); ++i) {
          int64_t elemBatch = (rowDim > 0) ? elemBatchIndex(cOffsets, i) : 0;

          // Sequential mode, no batch warps: compile-time batch IS actual
          // batch.
          if (batchRounds > 1 && rowDim > 0 && !cHasBatchWarps) {
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
            Value safeIdx = arith::SelectOp::create(rewriter, loc, inStrip, idx,
                                                    garbageIdx);
            Value val = gather1(ptrTG, safeIdx);
            if (val.getType() != outElemTy)
              val = fromF32(rewriter, loc, val, outElemTy);
            resultElems[i] = arith::SelectOp::create(rewriter, loc, inStrip,
                                                     val, resultElems[i]);
          } else {
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
