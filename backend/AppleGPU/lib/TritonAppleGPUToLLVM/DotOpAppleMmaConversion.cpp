// DotOpAppleMmaConversion: lower tt.dot with AppleMmaEncoding on C (rank-2).
#include "ConvertCommon.h"
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

struct OperandRouting {
  bool useDevice = false;
  bool useSmem = false;
  bool fast() const { return useDevice || useSmem; }
};

// DotOpAppleMmaConversion: AppleMmaEncoding on C, rank-2 only. Device-pointer
// A/B load MMA tiles directly via p1f32 intrinsics (skipping the TG
// scatter/gather barriers); otherwise falls back to the TG scatter path.
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
    if (!cMmaEnc)
      return failure();

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
    // Transposed TG simdgroup load from a COLUMN-MAJOR staging buffer: swaps
    // the 2D stride to {pitch,1} so the logical (k,n) tile is read transposed
    // in place, reaching the same packed MMA register a row-major B would.
    auto emitSGLoadT = [&](LLVMFuncOp fn, Value ptr, int64_t pitch,
                           Value off) -> Value {
      SmallVector<Value> args;
      if (canonSG)
        args = {ptr, makeI64(rewriter, loc, pitch), off,
                makeI1True(rewriter, loc)};
      else
        args = {ptr, makeI64Vec2(rewriter, loc, 8, pitch),
                makeI64Vec2(rewriter, loc, pitch, 1), off};
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

    // Device memory MMA intrinsics (type-specific). f16/bf16 load returns
    // <64 x half/bfloat> with an f32 accumulator; f32 loads <64 x float>.
    auto aElemTy = aType.getElementType();
    auto f16Ty = Float16Type::get(ctx);
    auto bf16Ty = BFloat16Type::get(ctx);
    bool isF16Input = aElemTy.isF16();
    bool isBF16Input = aElemTy.isBF16();

    // int8 has no f32-typed simdgroup load (the f32 load would misread
    // byte-packed ints), so the device path byte-loads, widens to f32, and
    // builds the matrix by hand (emitDevSGLoadInt8) before the f32 MMA.
    bool isIntInput = isa<IntegerType>(aElemTy);
    Type intGepElemTy = aElemTy;

    // Determine device load intrinsic based on element type
    Type devMatElemTy = f32Ty; // element type for device MMA matrix
    std::string devLoadName = "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
    std::string devMmaName = "air.simdgroup_matrix_8x8_multiply_accumulate."
                             "v64f32.v64f32.v64f32.v64f32";
    Type devGepElemTy = f32Ty; // GEP element type

    if (isIntInput) {
      devGepElemTy = intGepElemTy;
    } else if (isF16Input) {
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

    // Manual device-load + widen for int8 A/B tiles: each lane byte-loads its
    // two owned i8 elements under the physical AppleMma per-lane layout, widens
    // to f32, inserts at vector indices {0,1}. Per-lane mapping (lane T, reg
    // R):
    //   row = ((T>>1)&1) | (((T>>2)&1)<<1) | (((T>>4)&1)<<2)
    //   col = R | (((T>>0)&1)<<1) | (((T>>3)&1)<<2)
    auto emitDevSGLoadInt8 = [&](Value tilePtr, Value rowStride,
                                 Value colStride) -> Value {
      auto matIntTy = getSimdgroupMatrixType(ctx); // <64 x f32>
      auto initFn = getOrInsertIntrinsic(
          rewriter, mod, "air.simdgroup_matrix_8x8_init_filled.v64f32.f32",
          LLVMFunctionType::get(matIntTy, {f32Ty}, false));
      Value fz =
          arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(f32Ty));
      Value mat = LLVM::CallOp::create(rewriter, loc, initFn, ValueRange{fz})
                      .getResult();
      Value laneId64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, laneId);
      auto bit = [&](int64_t b) -> Value {
        Value sh = arith::ShRUIOp::create(
            rewriter, loc, laneId64,
            arith::ConstantIntOp::create(rewriter, loc, b, 64));
        return arith::AndIOp::create(
            rewriter, loc, sh,
            arith::ConstantIntOp::create(rewriter, loc, 1, 64));
      };
      Value l0 = bit(0), l1 = bit(1), l2 = bit(2), l3 = bit(3), l4 = bit(4);
      // row = L1 | (L2<<1) | (L4<<2)
      Value rowIdx = arith::OrIOp::create(
          rewriter, loc, l1,
          arith::OrIOp::create(
              rewriter, loc,
              arith::ShLIOp::create(
                  rewriter, loc, l2,
                  arith::ConstantIntOp::create(rewriter, loc, 1, 64)),
              arith::ShLIOp::create(
                  rewriter, loc, l4,
                  arith::ConstantIntOp::create(rewriter, loc, 2, 64))));
      // colBase = (L0<<1) | (L3<<2)   (col = colBase | R)
      Value colBase = arith::OrIOp::create(
          rewriter, loc,
          arith::ShLIOp::create(
              rewriter, loc, l0,
              arith::ConstantIntOp::create(rewriter, loc, 1, 64)),
          arith::ShLIOp::create(
              rewriter, loc, l3,
              arith::ConstantIntOp::create(rewriter, loc, 2, 64)));
      Value rowOff = arith::MulIOp::create(rewriter, loc, rowIdx, rowStride);
      for (int64_t r = 0; r < 2; ++r) {
        Value colIdx = arith::OrIOp::create(
            rewriter, loc, colBase,
            arith::ConstantIntOp::create(rewriter, loc, r, 64));
        Value elemOff = arith::AddIOp::create(
            rewriter, loc, rowOff,
            arith::MulIOp::create(rewriter, loc, colIdx, colStride));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, devPtrTy, devGepElemTy, tilePtr,
                                ArrayRef<LLVM::GEPArg>{elemOff});
        Value iVal =
            LLVM::LoadOp::create(rewriter, loc, aElemTy, elemPtr).getResult();
        Value fVal = toF32(rewriter, loc, iVal, f32Ty);
        Value vIdx = arith::ConstantIntOp::create(rewriter, loc, r, 32);
        mat = InsertElementOp::create(rewriter, loc, matIntTy, mat, fVal, vIdx);
      }
      return mat;
    };

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

    // Extract device pointers for direct MMA loads by tracing
    // dot.A -> ConvertLayoutOp -> LoadOp -> pointer operand. Returns per-thread
    // pointers with the same encoding/offsets as the values.
    auto resolveDevicePointers = [&](Value tritonVal) -> SmallVector<Value> {
      Value src = tritonVal;
      if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      auto loadOp = src.getDefiningOp<tt::LoadOp>();
      if (!loadOp)
        return {};
      // The device-direct SG load reads raw memory and can't see a load mask,
      // feeding the MMA unmasked bytes at masked-out positions; decline the
      // device path so the TG scatter preserves the register fill.
      if (loadOp.getMask())
        return {};
      Value ptrTensor = loadOp.getPtr();
      Value mappedPtrs = rewriter.getRemappedValue(ptrTensor);
      if (!mappedPtrs)
        return {};
      return unpack(mappedPtrs);
    };

    // Operand's tt.load pointer tensor; bails on masked loads (must not feed
    // the device-direct path).
    auto resolveLoadPtrTensor = [&](Value tritonVal) -> Value {
      Value src = tritonVal;
      if (auto cvt = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      auto loadOp = src.getDefiningOp<tt::LoadOp>();
      if (!loadOp || loadOp.getMask())
        return Value();
      return loadOp.getPtr();
    };

    auto remapStride = [&](Value stride, int64_t strideConst) -> Value {
      if (strideConst != INT64_MIN)
        return arith::ConstantIntOp::create(rewriter, loc, strideConst, 64);
      if (!stride)
        return Value();
      Value s = rewriter.getRemappedValue(stride);
      if (!s)
        return Value();
      if (auto it = dyn_cast<IntegerType>(s.getType())) {
        if (it.getWidth() < 64)
          s = arith::ExtSIOp::create(rewriter, loc, i64Ty, s);
        else if (it.getWidth() > 64)
          s = arith::TruncIOp::create(rewriter, loc, i64Ty, s);
      } else {
        return Value();
      }
      return s;
    };

    // (rowStride, colStride) from a proven 2D affine view; false -> caller
    // falls back to pointer-diff reconstruction.
    auto resolveAffinePtrInfo = [&](Value tritonVal, Value &rowStride,
                                    Value &colStride) -> bool {
      Value ptrTensor = resolveLoadPtrTensor(tritonVal);
      if (!ptrTensor)
        return false;
      AffineMmaPtrInfo info;
      if (!extractAffineMmaPtrInfo(ptrTensor, info))
        return false;
      Value rs = remapStride(info.rowStride, info.rowStrideConst);
      Value cs = remapStride(info.colStride, info.colStrideConst);
      if (!rs || !cs)
        return false;
      rowStride = rs;
      colStride = cs;
      return true;
    };

    // Pipelined-SMEM operands (num_stages>1): the operand comes from a
    // local_load of the staging buffer. Only a plain row-major strip
    // (maxPhase==1, order=[1,0], shape==operand shape) can be SG-loaded at
    // constant pitch. Returns {base ptr (as3), row pitch}; null -> registers.
    auto resolveSmemOperand = [&](Value tritonVal, RankedTensorType opTy,
                                  bool *colMajorOut =
                                      nullptr) -> std::pair<Value, int64_t> {
      Value src = tritonVal;
      if (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      auto localLoad = src.getDefiningOp<ttg::LocalLoadOp>();
      if (!localLoad)
        return {Value(), 0};
      auto memTy = dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
      if (!memTy || memTy.getRank() != 2)
        return {Value(), 0};
      if (memTy.getShape() != opTy.getShape() ||
          memTy.getElementType() != opTy.getElementType())
        return {Value(), 0};
      if (isa<IntegerType>(memTy.getElementType()))
        return {Value(), 0};
      auto shEnc =
          dyn_cast<ttg::SwizzledSharedEncodingAttr>(memTy.getEncoding());
      if (!shEnc || shEnc.getMaxPhase() != 1)
        return {Value(), 0};
      auto shOrder = shEnc.getOrder();
      if (shOrder.size() != 2)
        return {Value(), 0};
      bool colMajor = (shOrder[0] == 0);
      if (colMajor && !colMajorOut)
        return {Value(), 0};
      Value llStruct = rewriter.getRemappedValue(localLoad.getSrc());
      if (!llStruct)
        return {Value(), 0};
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, llStruct,
          getTypeConverter()->convertType(memTy.getElementType()), rewriter);
      if (colMajorOut)
        *colMajorOut = colMajor;
      int64_t pitch = colMajor ? memTy.getShape()[0] : memTy.getShape()[1];
      return {smemObj.getBase(), pitch};
    };

    auto [elemsA, aOffsets, aSrcEnc] =
        resolveOperand(op.getA(), adaptor.getA(), aType);
    auto [elemsB, bOffsets, bSrcEnc] =
        resolveOperand(op.getB(), adaptor.getB(), bType);
    auto elemsC = unpack(adaptor.getC());
    auto cOffsets = emitOffsetForLayout(cMmaEnc, cType);

    if (!aSrcEnc || !bSrcEnc)
      return failure();

    // Fragment ABI: C arrived as struct<(<64xf32> x F)>; the fragment vectors
    // seed matC_tiles and pack back directly, no scalar bridge.
    bool fragC = !elemsC.empty() && isa<VectorType>(elemsC.front().getType());
    SmallVector<Value> fragCIn;
    if (fragC)
      fragCIn = elemsC;

    if ((int64_t)elemsA.size() != (int64_t)aOffsets.size() ||
        (int64_t)elemsB.size() != (int64_t)bOffsets.size() ||
        (!fragC && (int64_t)elemsC.size() != (int64_t)cOffsets.size()))
      return failure();

    // Try to get device pointers for A and B
    auto aPtrs = resolveDevicePointers(op.getA());
    auto bPtrs = resolveDevicePointers(op.getB());
    OperandRouting aRoute, bRoute;
    aRoute.useDevice = (aPtrs.size() == elemsA.size() && !aPtrs.empty());
    bRoute.useDevice = (bPtrs.size() == elemsB.size() && !bPtrs.empty());

    // Row stride (leading dim) in elements for device MMA loads. Strategy 1:
    // two THIS-thread elements with same col, different row. Strategy 2 (all
    // elements same row): simd_shuffle the same-column pointer from the
    // row-adjacent lane.
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

      // Strategy 2: simd_shuffle_xor the pointer from the row-adjacent thread.
      // XOR mask that swaps row-adjacent threads: tpw[colDim] for col-fastest
      // (order=[1,0]), else 1.
      auto order = enc.getOrder();
      auto tpw = enc.getThreadsPerWarp();
      auto spt = enc.getSizePerThread();
      unsigned encRank = spt.size();
      unsigned encColDim = encRank - 1;
      unsigned encRowDim = encRank - 2;
      bool colFastest = (order[0] == (unsigned)encColDim);

      int64_t xorMask = colFastest ? tpw[encColDim] : 1;
      int64_t rowDiff = spt[encRowDim];

      // Split i64 into two i32 for the shuffle (Metal has no i64 shuffle).
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

      // Absolute stride: abs(other - this) / rowDiff.
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

      // XOR mask that swaps col-adjacent threads (opposite of row).
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
    if (aRoute.useDevice) {
      if (!resolveAffinePtrInfo(op.getA(), aRowStride, aColStride)) {
        aRowStride =
            computeRowStride(aPtrs, aOffsets, aType.getElementType(), aSrcEnc);
        aColStride =
            computeColStride(aPtrs, aOffsets, aType.getElementType(), aSrcEnc);
      }
    }
    if (bRoute.useDevice) {
      if (!resolveAffinePtrInfo(op.getB(), bRowStride, bColStride)) {
        bRowStride =
            computeRowStride(bPtrs, bOffsets, bType.getElementType(), bSrcEnc);
        bColStride =
            computeColStride(bPtrs, bOffsets, bType.getElementType(), bSrcEnc);
      }
    }

    // SMEM fallback for operands without device pointers: SG-load directly from
    // the staging buffer, keeping the dot register-resident (no operand
    // scatter, no per-strip barriers).
    Value aSmemBase, bSmemBase;
    int64_t aSmemPitch = 0, bSmemPitch = 0;
    bool bSmemColMajor = false;
    {
      if (!aRoute.useDevice)
        std::tie(aSmemBase, aSmemPitch) = resolveSmemOperand(op.getA(), aType);
      if (!bRoute.useDevice)
        std::tie(bSmemBase, bSmemPitch) =
            resolveSmemOperand(op.getB(), bType, &bSmemColMajor);
    }
    aRoute.useSmem = !aRoute.useDevice && static_cast<bool>(aSmemBase);
    bRoute.useSmem = !bRoute.useDevice && static_cast<bool>(bSmemBase);
    bool fastPath = aRoute.fast() && bRoute.fast();

    // Device base pointer for an 8x8 MMA tile. Subtracts this thread's own
    // element offset from ptrs[0] so every thread computes the SAME tile
    // origin. extraRow/extraCol are optional runtime addends for the per-warp
    // origin (warpRow*8/warpCol*8) so each warp loads only its owned tiles.
    auto computeTileDevPtr =
        [&](SmallVector<Value> &ptrs,
            SmallVector<SmallVector<unsigned>> &offsets, Value rowStride,
            Value colStride, Value baseRow, Value baseCol, int64_t tileRow,
            int64_t tileCol, Value extraRow = nullptr, Value extraCol = nullptr,
            Value refPtrOverride = nullptr) -> Value {
      Value refPtr = refPtrOverride ? refPtrOverride : ptrs[0];
      int64_t refRowOff = offsets[0][0];
      int64_t refColOff = offsets[0][1];

      // rowDelta = tileRow + extraRow - baseRow - refRowOff (runtime)
      Value baseRowExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseRow);
      Value rowDelta = arith::SubIOp::create(
          rewriter, loc,
          arith::ConstantIntOp::create(rewriter, loc,
                                       (int64_t)tileRow - refRowOff, 64),
          baseRowExt);
      if (extraRow) {
        Value e = arith::ExtUIOp::create(rewriter, loc, i64Ty, extraRow);
        rowDelta = arith::AddIOp::create(rewriter, loc, rowDelta, e);
      }

      // colDelta = tileCol + extraCol - baseCol - refColOff (runtime)
      Value baseColExt = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseCol);
      Value colDelta = arith::SubIOp::create(
          rewriter, loc,
          arith::ConstantIntOp::create(rewriter, loc,
                                       (int64_t)tileCol - refColOff, 64),
          baseColExt);
      if (extraCol) {
        Value e = arith::ExtUIOp::create(rewriter, loc, i64Ty, extraCol);
        colDelta = arith::AddIOp::create(rewriter, loc, colDelta, e);
      }

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

    // MMA stride for device loads. 3-vector form is <colStride, rowStride>; the
    // canonical (macOS<=15) form takes a single elements_per_row pitch + a
    // transpose bool. Exactly one stride is unit per MMA tile, so pick the
    // non-unit one as the pitch; transpose is derived at the call site.
    auto makeDevMmaStride = [&](Value colStride, Value rowStride) -> Value {
      if (canonSG) {
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
    // Canonical transpose flag: true when the operand is column-major
    // (rowStride < colStride). i1 false for the 3-vector path (transpose folded
    // into the stride vector there).
    auto makeDevMmaTranspose = [&](Value colStride, Value rowStride) -> Value {
      if (!canonSG)
        return makeI1False(rewriter, loc);
      return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                   rowStride, colStride);
    };

    // Canonical warp -> tile mapping (single source of truth). makeBase,
    // makeBaseMma, and the per-warp owned-tile loops MUST agree on tile
    // ownership or operand rows feed the wrong MMA. Linear column-major
    //   (warpRow, warpCol) = (w / wN, w % wN)
    // matching toLinearLayout() and emitOffsetForLayout; a Morton remap here
    // swaps warps for square pow2 warpsPerCTA and miscompiles.
    auto warpRowCol = [&](unsigned wN) -> std::pair<Value, Value> {
      Value warpRow = divByConst(rewriter, loc, warpId, wN);
      Value warpCol = remByConst(rewriter, loc, warpId, wN);
      return {warpRow, warpCol};
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

      // Warp decomposition: linear div/mod matching emitOffsetForLayout. The
      // A/B scatter round-trips through absolute threadgroup positions, so warp
      // order must be row/col-major (NOT Morton, which mis-maps square pow2
      // warpsPerCTA).
      Value wR, wC;
      if (colFastest) {
        std::tie(wR, wC) = warpRowCol(wN);
      } else {
        wR = remByConst(rewriter, loc, warpId, wM);
        wC = divByConst(rewriter, loc, warpId, wM);
      }
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

    // MMA base: lane->(row,col) within 8x8 tile, warp->tile position. Must use
    // column-major warp ordering to match toLinearLayout(); do NOT use Morton.
    auto makeBaseMma = [&](AppleMmaEncodingAttr enc, int64_t rows,
                           int64_t cols) -> std::pair<Value, Value> {
      auto wpc = enc.getWarpsPerCTA();
      unsigned wN = wpc[1];

      Value c7 = arith::ConstantIntOp::create(rewriter, loc, 7, 32);
      Value c3 = arith::ConstantIntOp::create(rewriter, loc, 3, 32);
      Value laneCol = arith::AndIOp::create(rewriter, loc, laneId, c7);
      Value laneRow = arith::ShRUIOp::create(rewriter, loc, laneId, c3);

      Value c8 = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
      // Shared with makeBase via warpRowCol so the two mappings can't diverge.
      auto [warpRow, warpCol] = warpRowCol(wN);

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

    auto aOffsetsL = aOffsets, bOffsetsL = bOffsets;
    Value aBaseRowL = aBaseRow, aBaseColL = aBaseCol;
    Value bBaseRowL = bBaseRow, bBaseColL = bBaseCol;

    // TG buffer for C scatter/load (always) and A/B scatter (TG path).
    unsigned id = getDotCounter(ctx)++;
    int64_t pad = tgPadForType(aElemTy);
    int64_t maxStrideMma = (fastPath) ? N : std::max(K, N);
    int64_t unpaddedSizeMma = 8 * maxStrideMma + 1;
    bool canPadMma = (pad > 0) && ((unpaddedSizeMma + 8 * pad) * 4 <= 16384);
    int64_t Kpad = canPadMma ? K + pad : K;
    int64_t Npad = canPadMma ? N + pad : N;
    int64_t tgCStripSize = 8 * Npad;
    int64_t tgABStripSize = 8 * std::max(Kpad, Npad);
    int64_t tgStripSize = (fastPath) ? tgCStripSize : tgABStripSize;
    // Multi-strip batching: give each 8-row strip its own TG block so all
    // strips scatter and read back under one barrier pair (trades TG memory for
    // fewer barriers). The batchStrips path keeps the whole C grid resident for
    // the per-warp owned-tile store-back; the per-strip path computes the full
    // grid. The compiler.py gatekeeper validates the real footprint vs 32KB.
    bool tgPath = !fastPath;
    int64_t maxStripsAB = std::max(M / 8, K / 8);
    int64_t batchedABSize = maxStripsAB * tgABStripSize;
    int64_t batchedCSize = (M / 8) * tgCStripSize;
    // Gate the resident-grid path on combined footprint vs 32KB. A sub-f32 dot
    // also keeps two half-typed convert_layout buffers resident (each ~the C
    // grid in operand bytes); the dot grid itself is f32 (4B/elem).
    int64_t residentGridBytes = std::max(batchedABSize, batchedCSize) * 4;
    unsigned aElemBytes = aElemTy.getIntOrFloatBitWidth() / 8;
    int64_t cvtBytes =
        (aElemBytes < 4) ? 2 * batchedCSize * (int64_t)aElemBytes : 0;
    bool batchStrips =
        tgPath && (residentGridBytes + cvtBytes <= kTGResidentBudgetBytes);
    int64_t tgSize = tgStripSize + 1;
    if (batchStrips)
      tgSize = std::max(batchedABSize, batchedCSize);
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

    // stripBlock places this strip into its own 8-row TG block (block*8 rows),
    // so multiple strips coexist under one barrier pair.
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

    // Reserved out-of-tile sink slot (last buffer element, tgSize-1).
    Value garbageIdx =
        arith::ConstantIntOp::create(rewriter, loc, tgSize - 1, 64);

    // Per-warp C-tile ownership: warp (warpRow,warpCol) owns the absolute tiles
    // (warpRow + k*warpsM, warpCol + j*warpsN), so each warp does only
    // ownM*ownN MMAs and the store-back writes each owned tile to its absolute
    // position (one writer per tile, no race).
    int64_t cWarpsM = cMmaEnc.getWarpsPerCTA()[0];
    int64_t cWarpsN = cMmaEnc.getWarpsPerCTA()[1];
    int64_t ownM = std::max<int64_t>(1, tilesM / cWarpsM);
    int64_t ownN = std::max<int64_t>(1, tilesN / cWarpsN);
    // Warp origin in elements + a base pre-offset to this warp's (0,0) owned
    // tile, for owned SG load/store (CONSTANT tile offsets only; a runtime SG
    // offset mis-lowers). Degenerates to the full grid when warpsM*warpsN == 1.
    auto [cWarpRowT, cWarpColT] = warpRowCol((unsigned)cWarpsN);
    Value c8own = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
    Value cWarpRowElem = arith::MulIOp::create(rewriter, loc, cWarpRowT, c8own);
    Value cWarpColElem = arith::MulIOp::create(rewriter, loc, cWarpColT, c8own);
    Value NpadOwn = arith::ConstantIntOp::create(rewriter, loc, Npad, 32);
    Value cWarpFlat = arith::AddIOp::create(
        rewriter, loc,
        arith::MulIOp::create(rewriter, loc, cWarpRowElem, NpadOwn),
        cWarpColElem);
    Value cWarpFlat64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, cWarpFlat);
    // Owned C base pointer (non-device TG path only; the device path builds its
    // own per-warp pointers inline).
    Value ptrTGcWarp = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty, ptrTG,
                                           ArrayRef<LLVM::GEPArg>{cWarpFlat64});
    // Owned A/B base pointers fold the warp origin into the TG base so each
    // warp SG-loads owned tiles at CONSTANT offsets (runtime SG offset
    // mis-lowers). A pre-offsets warpRow*8 rows; B a pure warpCol*8 column.
    Value KpadOwn = arith::ConstantIntOp::create(rewriter, loc, Kpad, 32);
    Value aWarpFlat64 = arith::ExtUIOp::create(
        rewriter, loc, i64Ty,
        arith::MulIOp::create(rewriter, loc, cWarpRowElem, KpadOwn));
    Value bWarpFlat64 =
        arith::ExtUIOp::create(rewriter, loc, i64Ty, cWarpColElem);
    Value ptrTGaWarp =
        LLVM::GEPOp::create(rewriter, loc, tgPtrTy, abTgScatterTy, ptrTG,
                            ArrayRef<LLVM::GEPArg>{aWarpFlat64});
    Value ptrTGbWarp =
        LLVM::GEPOp::create(rewriter, loc, tgPtrTy, abTgScatterTy, ptrTG,
                            ArrayRef<LLVM::GEPArg>{bWarpFlat64});

    // Direct scatter: store every operand element to its absolute TG offset
    // (baseRow+rowOff)*stride + (baseCol+colOff). No predicate, no garbage
    // sink; the matching SG-load reads each 8x8 tile from the same coordinates.
    auto directScatter = [&](Value ptr, Value baseRow, Value baseCol,
                             SmallVector<Value> &elems,
                             SmallVector<SmallVector<unsigned>> &offsets,
                             int64_t stride, Type scatterTy) {
      for (size_t idx = 0; idx < elems.size(); ++idx) {
        int64_t rowOff = offsets[idx][0];
        int64_t colOff = offsets[idx][1];
        Value row32 = arith::AddIOp::create(
            rewriter, loc, baseRow,
            arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
        Value col32 = arith::AddIOp::create(
            rewriter, loc, baseCol,
            arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
        Value flat32 = arith::AddIOp::create(
            rewriter, loc,
            arith::MulIOp::create(
                rewriter, loc, row32,
                arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
            col32);
        Value flat64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
        Value val = elems[idx];
        if (scatterTy == f32Ty) {
          if (val.getType() != f32Ty)
            val = toF32(rewriter, loc, val, f32Ty);
        } else {
          val = toMmaInputType(rewriter, loc, val, scatterTy);
        }
        Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, scatterTy, ptr,
                                        ArrayRef<LLVM::GEPArg>{flat64});
        LLVM::StoreOp::create(rewriter, loc, val, gep);
      }
    };

    auto filteredScatter = [&](Value ptr, Value garbIdx, Value baseRow,
                               Value baseCol, SmallVector<Value> &elems,
                               SmallVector<SmallVector<unsigned>> &offsets,
                               SmallVector<size_t> &bucket, int64_t stride,
                               int64_t rowStart, Type scatterTy,
                               int64_t stripBlock = 0) {
      // inStrip depends only on rowOff, so elements sharing a rowOff share one
      // predicate. Group the bucket by rowOff and guard each group with one
      // cond_br (block count proportional to strip rows, not to the full tile).
      std::map<int64_t, SmallVector<size_t>> byRow;
      SmallVector<int64_t> rowOrder;
      for (size_t idx : bucket) {
        int64_t rowOff = offsets[idx][0];
        if (byRow.find(rowOff) == byRow.end())
          rowOrder.push_back(rowOff);
        byRow[rowOff].push_back(idx);
      }
      for (int64_t rowOff : rowOrder) {
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
        auto *curBlock = rewriter.getInsertionBlock();
        auto insertPt = rewriter.getInsertionPoint();
        auto *thenBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));
        auto *afterBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(thenBlock)));
        afterBlock->getOperations().splice(afterBlock->begin(),
                                           curBlock->getOperations(), insertPt,
                                           curBlock->end());
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, inStrip, thenBlock, afterBlock);
        rewriter.setInsertionPointToStart(thenBlock);
        for (size_t idx : byRow[rowOff]) {
          int64_t colOff = offsets[idx][1];
          Value sIdx = stripFlatIdx(baseRow, baseCol, rowOff, colOff, stride,
                                    rowStart, stripBlock);
          Value val = elems[idx];
          if (scatterTy == f32Ty) {
            if (val.getType() != f32Ty)
              val = toF32(rewriter, loc, val, f32Ty);
          } else {
            val = toMmaInputType(rewriter, loc, val, scatterTy);
          }
          Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, scatterTy,
                                          ptr, ArrayRef<LLVM::GEPArg>{sIdx});
          LLVM::StoreOp::create(rewriter, loc, val, gep);
        }
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);
        rewriter.setInsertionPointToStart(afterBlock);
      }
    };
    (void)filteredScatter;

    // ── Phase 1: Load C tiles (filtered strip scatter via TG) ──────────
    SmallVector<SmallVector<Value>> matC_tiles(tilesM);
    for (int64_t tm = 0; tm < tilesM; ++tm)
      matC_tiles[tm].resize(tilesN);

    // Lane-local C bridge (physical AppleMma layout): each lane's #mma C
    // scalars sit at the simdgroup_matrix per-lane storage (vector indices
    // 0,1), so C-in is an insertelement and C-out the inverse extractelement at
    // vecIdx = colOff%2. No TG round-trip. owM/owN owned tile counts; wM/wN
    // warp grid.
    auto laneLocalCIn = [&](int64_t wM, int64_t wN, int64_t owM, int64_t owN) {
      auto matTyL = getSimdgroupMatrixType(ctx);
      if (fragC) {
        for (int64_t k = 0; k < owM; ++k)
          for (int64_t j = 0; j < owN; ++j) {
            int64_t fi = k * owN + j;
            matC_tiles[k][j] =
                (fi < (int64_t)fragCIn.size())
                    ? fragCIn[fi]
                    : getOrInsertSimdgroupInitFilled(rewriter, loc, mod);
          }
        return;
      }
      auto initFnL = getOrInsertIntrinsic(
          rewriter, mod, "air.simdgroup_matrix_8x8_init_filled.v64f32.f32",
          LLVMFunctionType::get(matTyL, {f32Ty}, false));
      Value fzL =
          arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(f32Ty));
      Value zMatL =
          LLVM::CallOp::create(rewriter, loc, initFnL, ValueRange{fzL})
              .getResult();
      for (int64_t k = 0; k < owM; ++k)
        for (int64_t j = 0; j < owN; ++j)
          matC_tiles[k][j] = zMatL;
      for (size_t idx = 0; idx < cOffsets.size(); ++idx) {
        int64_t rowOff = cOffsets[idx][0];
        int64_t colOff = cOffsets[idx][1];
        int64_t localTm = (rowOff / 8) / std::max<int64_t>(1, wM);
        int64_t localTn = (colOff / 8) / std::max<int64_t>(1, wN);
        if (localTm >= owM || localTn >= owN)
          continue;
        int64_t vecIdx = colOff % 2;
        Value v = elemsC[idx];
        if (v.getType() != f32Ty)
          v = toF32(rewriter, loc, v, f32Ty);
        Value vIdxC = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
        matC_tiles[localTm][localTn] = InsertElementOp::create(
            rewriter, loc, matTyL, matC_tiles[localTm][localTn], v, vIdxC);
      }
    };
    auto laneLocalCOut = [&](int64_t wM, int64_t wN, int64_t owM, int64_t owN,
                             Type outElemTy, SmallVector<Value> &resultElems) {
      for (size_t idx = 0; idx < cOffsets.size(); ++idx) {
        int64_t rowOff = cOffsets[idx][0];
        int64_t colOff = cOffsets[idx][1];
        int64_t localTm = (rowOff / 8) / std::max<int64_t>(1, wM);
        int64_t localTn = (colOff / 8) / std::max<int64_t>(1, wN);
        if (localTm >= owM || localTn >= owN)
          continue;
        int64_t vecIdx = colOff % 2;
        Value vIdxC = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
        Value val = ExtractElementOp::create(
            rewriter, loc, matC_tiles[localTm][localTn], vIdxC);
        if (val.getType() != outElemTy)
          val = fromF32(rewriter, loc, val, outElemTy);
        resultElems[idx] = val;
      }
    };

    // C accumulator-in must land in the SAME warp-owned slots the MMA and the
    // Phase-4 extract use, not the full grid, or a nonzero accumulator lands in
    // the wrong tile (invisible when C-in is zero).
    if (fastPath) {
      laneLocalCIn(cWarpsM, cWarpsN, ownM, ownN);
    } else if (batchStrips) {
      if (fragC) {
        laneLocalCIn(cWarpsM, cWarpsN, ownM, ownN);
      } else {
        auto matTyL = getSimdgroupMatrixType(ctx);
        auto initFnL = getOrInsertIntrinsic(
            rewriter, mod, "air.simdgroup_matrix_8x8_init_filled.v64f32.f32",
            LLVMFunctionType::get(matTyL, {f32Ty}, false));
        Value fzL = arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getZeroAttr(f32Ty));
        Value zMatL =
            LLVM::CallOp::create(rewriter, loc, initFnL, ValueRange{fzL})
                .getResult();
        for (int64_t k = 0; k < ownM; ++k)
          for (int64_t j = 0; j < ownN; ++j)
            matC_tiles[k][j] = zMatL;
        for (size_t idx = 0; idx < cOffsets.size(); ++idx) {
          int64_t rowOff = cOffsets[idx][0];
          int64_t colOff = cOffsets[idx][1];
          int64_t localTm = (rowOff / 8) / std::max<int64_t>(1, cWarpsM);
          int64_t localTn = (colOff / 8) / std::max<int64_t>(1, cWarpsN);
          if (localTm >= ownM || localTn >= ownN)
            continue;
          int64_t vecIdx = colOff % 2;
          Value v = elemsC[idx];
          if (v.getType() != f32Ty)
            v = toF32(rewriter, loc, v, f32Ty);
          Value vIdxC = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
          matC_tiles[localTm][localTn] = InsertElementOp::create(
              rewriter, loc, matTyL, matC_tiles[localTm][localTn], v, vIdxC);
        }
      }
    } else {
      laneLocalCIn(cWarpsM, cWarpsN, ownM, ownN);
    }

    // ── Phase 2: A/B loads + MMA ──────────────────────────────────
    if (fastPath) {
      // DEVICE PATH: direct MMA loads from device memory (no A/B scatter or
      // barriers). Per-warp tiling: each warp loads/accumulates only its owned
      // tiles, the device origin adding the runtime warp offset on top of the
      // compile-time step. A tiles for tk+1 prefetched before MMA of tk.
      Value aDevStride, bDevStride, aDevTranspose, bDevTranspose;
      Value aDevShape, bDevShape;
      auto makeDevShape = [&](Value rowStride) -> Value {
        auto ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
        Value vec = UndefOp::create(rewriter, loc, ty);
        Value i0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        Value i1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        Value eight = arith::ConstantIntOp::create(rewriter, loc, 8, 64);
        vec = InsertElementOp::create(rewriter, loc, ty, vec, rowStride, i0);
        vec = InsertElementOp::create(rewriter, loc, ty, vec, eight, i1);
        return vec;
      };
      if (aRoute.useDevice) {
        aDevStride = makeDevMmaStride(aColStride, aRowStride);
        aDevTranspose = makeDevMmaTranspose(aColStride, aRowStride);
        aDevShape = makeDevShape(aRowStride);
      }
      if (bRoute.useDevice) {
        bDevStride = makeDevMmaStride(bColStride, bRowStride);
        bDevTranspose = makeDevMmaTranspose(bColStride, bRowStride);
        bDevShape = makeDevShape(bRowStride);
      }
      Value zeroOff = makeI64Vec2(rewriter, loc, 0, 0);

      auto dWpc = cMmaEnc.getWarpsPerCTA();
      int64_t warpsM = dWpc[0];
      int64_t warpsN = dWpc[1];
      unsigned dwN = dWpc[1];
      int64_t ownM = std::max<int64_t>(1, tilesM / warpsM);
      int64_t ownN = std::max<int64_t>(1, tilesN / warpsN);

      // Runtime warp tile origin in ELEMENTS (warpRow*8, warpCol*8).
      Value dWarpRow = divByConst(rewriter, loc, warpId, dwN);
      Value dWarpCol = remByConst(rewriter, loc, warpId, dwN);
      Value c8i = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
      Value warpRowElem = arith::MulIOp::create(rewriter, loc, dWarpRow, c8i);
      Value warpColElem = arith::MulIOp::create(rewriter, loc, dWarpCol, c8i);

      // SMEM warp-base pointers: fold the warp origin into the base so SG loads
      // use CONSTANT tile offsets (a runtime SG offset mis-lowers). A's is a
      // row offset (warpRow*8 rows of pitch aSmemPitch); B's a pure column.
      Value ptrSmemAWarp, ptrSmemBWarp;
      if (aRoute.useSmem) {
        Value pitchA =
            arith::ConstantIntOp::create(rewriter, loc, aSmemPitch, 32);
        Value flat = arith::ExtUIOp::create(
            rewriter, loc, i64Ty,
            arith::MulIOp::create(rewriter, loc, warpRowElem, pitchA));
        ptrSmemAWarp =
            LLVM::GEPOp::create(rewriter, loc, tgPtrTy, abTgScatterTy,
                                aSmemBase, ArrayRef<LLVM::GEPArg>{flat});
      }
      if (bRoute.useSmem) {
        Value colUnit = warpColElem;
        if (bSmemColMajor) {
          Value pitchB =
              arith::ConstantIntOp::create(rewriter, loc, bSmemPitch, 32);
          colUnit = arith::MulIOp::create(rewriter, loc, warpColElem, pitchB);
        }
        Value flat = arith::ExtUIOp::create(rewriter, loc, i64Ty, colUnit);
        ptrSmemBWarp =
            LLVM::GEPOp::create(rewriter, loc, tgPtrTy, abTgScatterTy,
                                bSmemBase, ArrayRef<LLVM::GEPArg>{flat});
      }

      // Tile loaders dispatch per operand: SMEM strips load via the TG
      // simdgroup intrinsic at constant pitch, device operands via the
      // device-direct load; both yield the same v64 matrix type for one MMA fn.
      auto loadATile = [&](int64_t k, int64_t tk) -> Value {
        if (aRoute.useSmem) {
          Value aOff = makeI64Vec2(rewriter, loc, tk * 8, k * warpsM * 8);
          return emitSGLoad(abTgLoadFn, ptrSmemAWarp, aSmemPitch, aSmemPitch,
                            aOff);
        }
        Value aTilePtr = computeTileDevPtr(
            aPtrs, aOffsetsL, aRowStride, aColStride, aBaseRowL, aBaseColL,
            k * warpsM * 8, tk * 8, warpRowElem, nullptr);
        return isIntInput ? emitDevSGLoadInt8(aTilePtr, aRowStride, aColStride)
                          : emitDevSGLoad(devLoadFn, aTilePtr, aDevShape,
                                          aDevStride, zeroOff, aDevTranspose);
      };
      auto loadBTile = [&](int64_t tk, int64_t j) -> Value {
        if (bRoute.useSmem) {
          Value bOff = makeI64Vec2(rewriter, loc, j * warpsN * 8, tk * 8);
          if (bSmemColMajor)
            return emitSGLoadT(abTgLoadFn, ptrSmemBWarp, bSmemPitch, bOff);
          return emitSGLoad(abTgLoadFn, ptrSmemBWarp, bSmemPitch, bSmemPitch,
                            bOff);
        }
        Value bTilePtr = computeTileDevPtr(
            bPtrs, bOffsetsL, bRowStride, bColStride, bBaseRowL, bBaseColL,
            tk * 8, j * warpsN * 8, nullptr, warpColElem);
        return isIntInput ? emitDevSGLoadInt8(bTilePtr, bRowStride, bColStride)
                          : emitDevSGLoad(devLoadFn, bTilePtr, bDevShape,
                                          bDevStride, zeroOff, bDevTranspose);
      };

      if (aRoute.useSmem || bRoute.useSmem) {
        // SMEM-resident operands: SG-load every owned strip tile upfront, ONE
        // barrier, then the all-register MMA loop. The barrier is required: the
        // pipeliner prefetches num_stages-1 ahead, so without the fence the
        // next copy can clobber this slot mid-read (silent, K-count-dependent
        // corruption). Loading first keeps the MMA off the barrier's crit path.
        SmallVector<SmallVector<Value>> matA(ownM);
        for (int64_t k = 0; k < ownM; ++k) {
          matA[k].resize(tilesK);
          for (int64_t tk = 0; tk < tilesK; ++tk)
            matA[k][tk] = loadATile(k, tk);
        }
        SmallVector<SmallVector<Value>> matB(tilesK);
        for (int64_t tk = 0; tk < tilesK; ++tk) {
          matB[tk].resize(ownN);
          for (int64_t j = 0; j < ownN; ++j)
            matB[tk][j] = loadBTile(tk, j);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
        for (int64_t tk = 0; tk < tilesK; ++tk)
          for (int64_t j = 0; j < ownN; ++j)
            for (int64_t k = 0; k < ownM; ++k)
              matC_tiles[k][j] =
                  LLVM::CallOp::create(
                      rewriter, loc, devMmaFn,
                      ValueRange{matA[k][tk], matB[tk][j], matC_tiles[k][j]})
                      .getResult();
      } else {
        // Rolled K-loop: emit the K accumulation as a real LLVM-dialect loop
        // (accumulators carried as block args). A full unroll inflates register
        // pressure enough that the AGX driver caps maxThreadsPerThreadgroup
        // (~384 vs ~896); the bounded unroll factor (METAL_MMA_KUNROLL, default
        // 2) keeps latency hiding. Rolled only when tilesK % U == 0 and trip >=
        // 2; small/odd K falls back to straight-line unroll.
        int64_t kUnroll = 2;
        if (const char *e = ::getenv("METAL_MMA_KUNROLL")) {
          int64_t v = atoll(e);
          if (v >= 1)
            kUnroll = v;
        }
        // Clamp to a divisor of tilesK so the trip count is exact (no tail).
        while (kUnroll > 1 && (tilesK % kUnroll) != 0)
          --kUnroll;
        int64_t tripCount = tilesK / kUnroll;
        bool doRoll = (tilesK > kUnroll) && (tripCount >= 2);

        // Runtime-K tile loaders: the K offset rides a loop-carried A/B
        // reference pointer (aKRef/bKRef) advanced kUnroll*8 K-elements per
        // trip (ptr += const recurrence). tkElemOff is a fallback for callers
        // with no carried base.
        auto loadATileRT = [&](int64_t k, Value tkElemOff,
                               Value aKRef = nullptr) -> Value {
          Value aTilePtr = computeTileDevPtr(
              aPtrs, aOffsetsL, aRowStride, aColStride, aBaseRowL, aBaseColL,
              k * warpsM * 8, /*tileCol=*/0, warpRowElem,
              /*extraCol=*/aKRef ? nullptr : tkElemOff, aKRef);
          return isIntInput
                     ? emitDevSGLoadInt8(aTilePtr, aRowStride, aColStride)
                     : emitDevSGLoad(devLoadFn, aTilePtr, aDevShape, aDevStride,
                                     zeroOff, aDevTranspose);
        };
        auto loadBTileRT = [&](Value tkElemOff, int64_t j,
                               Value bKRef = nullptr) -> Value {
          Value bTilePtr = computeTileDevPtr(
              bPtrs, bOffsetsL, bRowStride, bColStride, bBaseRowL, bBaseColL,
              /*tileRow=*/0, j * warpsN * 8,
              /*extraRow=*/bKRef ? nullptr : tkElemOff, warpColElem, bKRef);
          return isIntInput
                     ? emitDevSGLoadInt8(bTilePtr, bRowStride, bColStride)
                     : emitDevSGLoad(devLoadFn, bTilePtr, bDevShape, bDevStride,
                                     zeroOff, bDevTranspose);
        };

        if (!doRoll) {
          // Straight-line unroll (small/odd K, rolling buys nothing).
          SmallVector<Value> matA_cur(ownM);
          for (int64_t k = 0; k < ownM; ++k)
            matA_cur[k] = loadATile(k, 0);

          for (int64_t tk = 0; tk < tilesK; ++tk) {
            SmallVector<Value> matA_next(ownM);
            if (tk + 1 < tilesK) {
              for (int64_t k = 0; k < ownM; ++k)
                matA_next[k] = loadATile(k, tk + 1);
            }
            for (int64_t j = 0; j < ownN; ++j) {
              Value matB = loadBTile(tk, j);
              for (int64_t k = 0; k < ownM; ++k) {
                matC_tiles[k][j] =
                    LLVM::CallOp::create(
                        rewriter, loc, devMmaFn,
                        ValueRange{matA_cur[k], matB, matC_tiles[k][j]})
                        .getResult();
              }
            }
            if (tk + 1 < tilesK)
              matA_cur = matA_next;
          }
        } else {
          // Real loop. Carried: the ownM*ownN accumulators flattened row-major
          // as [k*ownN + j]. IV is the K-step index tk (i32), stepping kUnroll.
          int64_t nAcc = ownM * ownN;
          auto accInit = [&](int64_t k, int64_t j) -> Value {
            return matC_tiles[k][j];
          };

          // Advance a device pointer by kElems K-elements (A's col axis /
          // B's row axis). Constant GEP, so the per-trip step is ptr += const.
          auto advanceK = [&](Value ptr, Value stride,
                              int64_t kElems) -> Value {
            Value off = arith::MulIOp::create(
                rewriter, loc,
                arith::ConstantIntOp::create(rewriter, loc, kElems, 64),
                stride);
            return LLVM::GEPOp::create(rewriter, loc, devPtrTy, devGepElemTy,
                                       ptr, ArrayRef<LLVM::GEPArg>{off});
          };
          bool carryKPtr = !isIntInput;
          Value aKRefInit = aPtrs[0], bKRefInit = bPtrs[0];

          Block *curBlock = rewriter.getInsertionBlock();
          Block::iterator curPoint = rewriter.getInsertionPoint();
          Block *exitBlock = curBlock->splitBlock(curPoint);

          // Header block args: tk (i32) + nAcc accumulators (matTy). The exit
          // block carries the SAME signature (cond_br false edge forwards
          // them).
          SmallVector<Type> hdrArgTys;
          SmallVector<Location> hdrArgLocs;
          hdrArgTys.push_back(i32Ty);
          hdrArgLocs.push_back(loc);
          for (int64_t a = 0; a < nAcc; ++a) {
            hdrArgTys.push_back(matTy);
            hdrArgLocs.push_back(loc);
          }
          // Loop-carried A/B reference pointers (strength-reduced K address).
          int64_t aKArgIdx = 1 + nAcc, bKArgIdx = 2 + nAcc;
          if (carryKPtr) {
            hdrArgTys.push_back(devPtrTy);
            hdrArgLocs.push_back(loc);
            hdrArgTys.push_back(devPtrTy);
            hdrArgLocs.push_back(loc);
          }
          for (size_t a = 0; a < hdrArgTys.size(); ++a)
            exitBlock->addArgument(hdrArgTys[a], hdrArgLocs[a]);
          Block *header =
              rewriter.createBlock(exitBlock, hdrArgTys, hdrArgLocs);
          Block *body = rewriter.createBlock(exitBlock);

          // Entry edge: jump into the header with tk=0 and the C-in
          // accumulators.
          rewriter.setInsertionPointToEnd(curBlock);
          SmallVector<Value> entryArgs;
          entryArgs.push_back(
              arith::ConstantIntOp::create(rewriter, loc, 0, 32));
          for (int64_t k = 0; k < ownM; ++k)
            for (int64_t j = 0; j < ownN; ++j)
              entryArgs.push_back(accInit(k, j));
          if (carryKPtr) {
            entryArgs.push_back(aKRefInit);
            entryArgs.push_back(bKRefInit);
          }
          LLVM::BrOp::create(rewriter, loc, entryArgs, header);

          // Header: while (tk < tilesK) -> body else -> exit.
          rewriter.setInsertionPointToEnd(header);
          Value tkIv = header->getArgument(0);
          Value tkLimit =
              arith::ConstantIntOp::create(rewriter, loc, tilesK, 32);
          Value cond = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::slt, tkIv, tkLimit);
          SmallVector<Value> exitFwd(header->getArguments().begin(),
                                     header->getArguments().end());
          LLVM::CondBrOp::create(rewriter, loc, cond, body, ValueRange{},
                                 exitBlock, exitFwd);

          // Body: unroll kUnroll sub-steps, threading the accumulators.
          rewriter.setInsertionPointToEnd(body);
          Value c8 = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
          // Working copy of the carried accumulators (indexed [k][j]).
          SmallVector<SmallVector<Value>> acc(ownM);
          for (int64_t k = 0; k < ownM; ++k) {
            acc[k].resize(ownN);
            for (int64_t j = 0; j < ownN; ++j)
              acc[k][j] = header->getArgument(1 + k * ownN + j);
          }
          Value aKRef = carryKPtr ? header->getArgument(aKArgIdx) : nullptr;
          Value bKRef = carryKPtr ? header->getArgument(bKArgIdx) : nullptr;
          for (int64_t u = 0; u < kUnroll; ++u) {
            // Sub-step: the carried pointers encode K=tk; advance by the
            // constant u*8 K-elements (folds into the load GEP, no runtime
            // mul).
            Value aSub = aKRef, bSub = bKRef;
            Value tkElemOff = nullptr;
            if (carryKPtr) {
              if (u > 0) {
                aSub = advanceK(aKRef, aColStride, u * 8);
                bSub = advanceK(bKRef, bRowStride, u * 8);
              }
            } else {
              Value subIdx =
                  (u == 0) ? tkIv
                           : arith::AddIOp::create(rewriter, loc, tkIv,
                                                   arith::ConstantIntOp::create(
                                                       rewriter, loc, u, 32));
              tkElemOff = arith::MulIOp::create(rewriter, loc, subIdx, c8);
            }
            SmallVector<Value> matA_cur(ownM);
            for (int64_t k = 0; k < ownM; ++k)
              matA_cur[k] = loadATileRT(k, tkElemOff, aSub);
            for (int64_t j = 0; j < ownN; ++j) {
              Value matB = loadBTileRT(tkElemOff, j, bSub);
              for (int64_t k = 0; k < ownM; ++k)
                acc[k][j] = LLVM::CallOp::create(
                                rewriter, loc, devMmaFn,
                                ValueRange{matA_cur[k], matB, acc[k][j]})
                                .getResult();
            }
          }
          // tk += kUnroll, branch back to header with updated accumulators.
          Value tkNext = arith::AddIOp::create(
              rewriter, loc, tkIv,
              arith::ConstantIntOp::create(rewriter, loc, kUnroll, 32));
          SmallVector<Value> backArgs;
          backArgs.push_back(tkNext);
          for (int64_t k = 0; k < ownM; ++k)
            for (int64_t j = 0; j < ownN; ++j)
              backArgs.push_back(acc[k][j]);
          if (carryKPtr) {
            backArgs.push_back(advanceK(aKRef, aColStride, kUnroll * 8));
            backArgs.push_back(advanceK(bKRef, bRowStride, kUnroll * 8));
          }
          auto latchBr = LLVM::BrOp::create(rewriter, loc, backArgs, header);
          Builder mdBuilder(ctx);
          auto unrollMD = LLVM::LoopUnrollAttr::get(
              ctx, mdBuilder.getBoolAttr(true), {}, {}, {}, {}, {}, {});
          auto loopMD =
              LLVM::LoopAnnotationAttr::get(ctx, {}, {}, {}, unrollMD, {}, {},
                                            {}, {}, {}, {}, {}, {}, {}, {}, {});
          latchBr.setLoopAnnotationAttr(loopMD);

          // Exit: harvest the final accumulators from the header-forwarded
          // args.
          rewriter.setInsertionPointToStart(exitBlock);
          for (int64_t k = 0; k < ownM; ++k)
            for (int64_t j = 0; j < ownN; ++j)
              matC_tiles[k][j] = exitBlock->getArgument(1 + k * ownN + j);
        }
      }
    } else if (batchStrips) {
      // TG PATH (batched): scatter all of A once, one barrier, SG-load every
      // (tm,tk) tile, one barrier; then the same for B fused with MMA.
      // Collapses the O(tilesK*tilesM) per-strip barrier pairs to 4 for the
      // whole dot.
      directScatter(ptrTG, aBaseRow, aBaseCol, elemsA, aOffsets, Kpad,
                    abTgScatterTy);
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      // Per-warp: every warp scattered the full A grid above, but loads only
      // its owned tile-rows via the warp-base pointer at constant offset.
      SmallVector<SmallVector<Value>> matA(ownM);
      for (int64_t k = 0; k < ownM; ++k) {
        matA[k].resize(tilesK);
        for (int64_t tk = 0; tk < tilesK; ++tk) {
          Value aOff = makeI64Vec2(rewriter, loc, tk * 8, k * cWarpsM * 8);
          matA[k][tk] = emitSGLoad(abTgLoadFn, ptrTGaWarp, Kpad, Kpad, aOff);
        }
      }
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      directScatter(ptrTG, bBaseRow, bBaseCol, elemsB, bOffsets, Npad,
                    abTgScatterTy);
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      // PER-WARP: each warp loads only its owned B tile-cols (absolute col
      // warpCol + j*warpsN) and accumulates only its owned C tiles.
      for (int64_t tk = 0; tk < tilesK; ++tk)
        for (int64_t j = 0; j < ownN; ++j) {
          Value bOff = makeI64Vec2(rewriter, loc, j * cWarpsN * 8, tk * 8);
          Value matB = emitSGLoad(abTgLoadFn, ptrTGbWarp, Npad, Npad, bOff);
          for (int64_t k = 0; k < ownM; ++k)
            matC_tiles[k][j] =
                LLVM::CallOp::create(
                    rewriter, loc, abTgMmaFn,
                    ValueRange{matA[k][tk], matB, matC_tiles[k][j]})
                    .getResult();
        }
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});
    } else {
      // TG PATH (per-strip): A's scatter stays CTA-uniform over every absolute
      // row strip (a row's source lanes may live in another C warp row); each
      // warp retains only its owned A rows and MMAs only its owned C tiles.
      int64_t aMaxBase = maxBaseRow(aSrcEnc);
      int64_t bMaxBase = maxBaseRow(bSrcEnc);
      auto aBuckets = bucketElements(aOffsets, aMaxBase, tilesM, 0);
      auto bBuckets = bucketElements(bOffsets, bMaxBase, tilesK, 0);

      SmallVector<SmallVector<Value>> matA(ownM);
      Value initA = LLVM::UndefOp::create(rewriter, loc, abTgMmaInfo.matVecTy);
      for (int64_t k = 0; k < ownM; ++k) {
        matA[k].resize(tilesK);
        for (int64_t tk = 0; tk < tilesK; ++tk)
          matA[k][tk] = initA;
      }
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        int64_t rowStart = tm * 8;
        filteredScatter(ptrTG, garbageIdx, aBaseRow, aBaseCol, elemsA, aOffsets,
                        aBuckets[tm], Kpad, rowStart, abTgScatterTy);
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
        int64_t localK = tm / std::max<int64_t>(1, cWarpsM);
        int64_t ownerWarpRow = tm % std::max<int64_t>(1, cWarpsM);
        Value ownsRow = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::eq, cWarpRowT,
            arith::ConstantIntOp::create(rewriter, loc, ownerWarpRow, 32));
        for (int64_t tk = 0; tk < tilesK; ++tk) {
          Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
          Value loaded = emitSGLoad(abTgLoadFn, ptrTG, Kpad, Kpad, aOff);
          if (localK < ownM)
            matA[localK][tk] = LLVM::SelectOp::create(rewriter, loc, ownsRow,
                                                      loaded, matA[localK][tk]);
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
      }

      for (int64_t tk = 0; tk < tilesK; ++tk) {
        int64_t rowStart = tk * 8;
        filteredScatter(ptrTG, garbageIdx, bBaseRow, bBaseCol, elemsB, bOffsets,
                        bBuckets[tk], Npad, rowStart, abTgScatterTy);
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});

        // Load only this warp's owned B tile-cols at a CONSTANT tile offset and
        // accumulate only the owned columns.
        for (int64_t j = 0; j < ownN; ++j) {
          Value bOff = makeI64Vec2(rewriter, loc, j * cWarpsN * 8, 0);
          Value matB = emitSGLoad(abTgLoadFn, ptrTGbWarp, Npad, Npad, bOff);
          for (int64_t k = 0; k < ownM; ++k)
            matC_tiles[k][j] =
                LLVM::CallOp::create(
                    rewriter, loc, abTgMmaFn,
                    ValueRange{matA[k][tk], matB, matC_tiles[k][j]})
                    .getResult();
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
      }
    }

    // ── Phase 4: Extract C results from MMA tiles ────────────────────
    auto outElemTy = cType.getElementType();

    // Fragment ABI: pack the owned matC_tiles vectors straight back into the
    // fragment struct, keeping the accumulator vectorized through the loop.
    if (fragC) {
      auto outStructTy =
          cast<LLVMStructType>(getTypeConverter()->convertType(cType));
      Value result = UndefOp::create(rewriter, loc, outStructTy);
      int64_t F = (int64_t)outStructTy.getBody().size();
      for (int64_t k = 0; k < ownM; ++k)
        for (int64_t j = 0; j < ownN; ++j) {
          int64_t fi = k * ownN + j;
          if (fi >= F)
            continue;
          Value frag = matC_tiles[k][j];
          result = InsertValueOp::create(rewriter, loc, outStructTy, result,
                                         frag, ArrayRef<int64_t>{fi});
        }
      rewriter.replaceOp(op, result);
      return success();
    }

    SmallVector<Value> resultElems(elemsC.size());
    for (size_t i = 0; i < elemsC.size(); ++i)
      resultElems[i] = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(outElemTy));

    // C-out is a lane-local extractelement (inverse of laneLocalCIn) for every
    // register-resident path; cOffsets are physical coords.
    if (fastPath) {
      laneLocalCOut(cWarpsM, cWarpsN, ownM, ownN, outElemTy, resultElems);
    } else if (batchStrips) {
      for (size_t idx = 0; idx < cOffsets.size(); ++idx) {
        int64_t rowOff = cOffsets[idx][0];
        int64_t colOff = cOffsets[idx][1];
        int64_t localTm = (rowOff / 8) / std::max<int64_t>(1, cWarpsM);
        int64_t localTn = (colOff / 8) / std::max<int64_t>(1, cWarpsN);
        if (localTm >= ownM || localTn >= ownN)
          continue;
        int64_t vecIdx = colOff % 2;
        Value vIdxC = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
        Value val = ExtractElementOp::create(
            rewriter, loc, matC_tiles[localTm][localTn], vIdxC);
        if (val.getType() != outElemTy)
          val = fromF32(rewriter, loc, val, outElemTy);
        resultElems[idx] = val;
      }
    } else {
      laneLocalCOut(cWarpsM, cWarpsN, ownM, ownN, outElemTy, resultElems);
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

// Fragment ABI: a splat constant feeding the dot-chain (zero accumulator init)
// lowers to a struct of F init-filled simdgroup fragments. Matches only when
// the type converter chose the fragment struct for this #mma tensor.
struct AppleMmaConstantConversion
    : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern<arith::ConstantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getType());
    if (!ty)
      return failure();
    auto enc = dyn_cast<AppleMmaEncodingAttr>(ty.getEncoding());
    if (!enc)
      return failure();
    auto outTy = getTypeConverter()->convertType(ty);
    auto structTy = dyn_cast_or_null<LLVMStructType>(outTy);
    if (!structTy || structTy.getBody().empty() ||
        !isa<VectorType>(structTy.getBody()[0]))
      return failure(); // not on the fragment path

    auto splat = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!splat || !splat.isSplat())
      return failure();
    auto fVal = dyn_cast<FloatAttr>(splat.getSplatValue<Attribute>());
    if (!fVal)
      return failure();

    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    Value frag = getOrInsertSimdgroupInitFilled(
        rewriter, loc, mod, (float)fVal.getValue().convertToDouble());
    Value result = UndefOp::create(rewriter, loc, structTy);
    for (size_t i = 0; i < structTy.getBody().size(); ++i)
      result = InsertValueOp::create(rewriter, loc, structTy, result, frag,
                                     ArrayRef<int64_t>{(int64_t)i});
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // anonymous namespace

namespace mlir::triton::applegpu {

void populateDotOpAppleMmaPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit) {
  patterns.add<DotOpAppleMmaConversion>(typeConverter, benefit);
  patterns.add<AppleMmaConstantConversion>(typeConverter,
                                           benefit.getBenefit() + 100);
}

} // namespace mlir::triton::applegpu
