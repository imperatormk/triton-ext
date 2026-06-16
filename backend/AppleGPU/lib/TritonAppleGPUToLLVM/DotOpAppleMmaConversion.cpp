// DotOpAppleMmaConversion: lower tt.dot with AppleMmaEncoding on C (rank-2).
#include "DotCommon.h"
#include "Dialect/TritonAppleGPU/IR/AppleMmaFragment.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
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

    // Integer (int8) inputs have no f32-typed simdgroup load: the
    // air.simdgroup_matrix_8x8_load intrinsic reads f32 from memory and would
    // misread 1-byte-packed integers (4x stride mismatch). For these, the
    // device path byte-loads each i8 element this lane owns, widens it to f32,
    // and builds the <64 x f32> matrix by hand (emitDevSGLoadInt8), then feeds
    // the unchanged f32 MMA -- numerically identical to the f32 TG scatter
    // path.
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

    // Manual device-load + widen for integer (int8) A/B tiles. The f32
    // air.simdgroup_matrix_8x8_load cannot read 1-byte-packed integers, so each
    // lane byte-loads exactly the two i8 elements it owns under the PHYSICAL
    // AppleMma per-lane layout (AppleMmaLayoutConversions.cpp), widens them to
    // f32, and inserts them at vector indices {0,1} to build the same <64 x
    // f32> matrix an f32 simdgroup load of the widened data would produce.
    // Physical per-lane mapping for lane T, register R in {0,1}:
    //   row = ((T>>1)&1) | (((T>>2)&1)<<1) | (((T>>4)&1)<<2)
    //   col = R | (((T>>0)&1)<<1) | (((T>>3)&1)<<2)
    // tilePtr is the i8 device pointer at the tile origin (row 0, col 0);
    // rowStride/colStride are runtime i64 element strides of the operand.
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
      // A masked load (tt.load %ptr, %mask, %other) zero-/other-fills the
      // out-of-bounds lanes in registers. The device-direct
      // simdgroup_matrix_8x8_load.p1 reads raw device memory and cannot see
      // that masking, so it would feed the MMA the unmasked device bytes for
      // the masked-out positions and corrupt the result (100% mismatch on
      // masked tl.dot, e.g. depthwise / grouped conv2d_backward). Decline the
      // device path so the operand goes through the TG scatter, which scatters
      // the already-masked register values and preserves the fill.
      if (loadOp.getMask())
        return {};
      // Get the pointer operand (tensor of device pointers)
      Value ptrTensor = loadOp.getPtr();
      Value mappedPtrs = rewriter.getRemappedValue(ptrTensor);
      if (!mappedPtrs)
        return {};
      return unpack(mappedPtrs);
    };

    // ── Pipelined-SMEM operands: SG-load straight from the staging buffer ──
    // With num_stages>1 the operand comes from ttg.local_load of the
    // pipeline's threadgroup staging buffer (resolveDevicePointers fails: the
    // tt.load is gone and the pipelined copies are masked). The operand bytes
    // already sit in TG memory as a plain row-major strip (maxPhase==1,
    // order=[1,0], shape==operand shape), which is exactly the layout
    // simdgroup_matrix_8x8_load reads at constant pitch, so emit SG loads
    // from the buffer the local_load reads. Returns {base ptr (addrspace 3),
    // row pitch elems}; null base means unprovable -> registers fallback.
    //
    // multiSlot: the strip is one rotating slot of a multi-stage pipeline
    // alloc (memdesc_index into a rank-3 local_alloc with >=2 stage slots).
    // Slots are disjoint and the index rotates s % stages, so the next
    // K-step's copy targets a different slot than this step's reads, and the
    // step in between always executes an async_wait barrier before its own
    // loads, which already separates these reads from the next write to the
    // SAME slot (two steps later). Single-slot staging (num_stages==2) keeps
    // the post-load barrier.
    auto isMultiSlotStaging = [](ttg::LocalLoadOp localLoad) -> bool {
      auto idxOp = localLoad.getSrc().getDefiningOp<ttg::MemDescIndexOp>();
      if (!idxOp)
        return false;
      auto alloc = idxOp.getSrc().getDefiningOp<ttg::LocalAllocOp>();
      if (!alloc || alloc.getSrc())
        return false;
      auto allocTy = cast<ttg::MemDescType>(alloc.getType());
      return allocTy.getRank() == 3 && allocTy.getShape()[0] >= 2;
    };
    auto resolveSmemOperand =
        [&](Value tritonVal,
            RankedTensorType opTy) -> std::tuple<Value, int64_t, bool> {
      Value src = tritonVal;
      if (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>())
        src = cvt.getSrc();
      auto localLoad = src.getDefiningOp<ttg::LocalLoadOp>();
      if (!localLoad)
        return {Value(), 0, false};
      auto memTy = dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
      if (!memTy || memTy.getRank() != 2)
        return {Value(), 0, false};
      if (memTy.getShape() != opTy.getShape() ||
          memTy.getElementType() != opTy.getElementType())
        return {Value(), 0, false};
      if (isa<IntegerType>(memTy.getElementType()))
        return {Value(), 0, false};
      auto shEnc =
          dyn_cast<ttg::SwizzledSharedEncodingAttr>(memTy.getEncoding());
      if (!shEnc || shEnc.getMaxPhase() != 1)
        return {Value(), 0, false};
      auto shOrder = shEnc.getOrder();
      if (shOrder.size() != 2 || shOrder[0] != 1)
        return {Value(), 0, false};
      Value llStruct = rewriter.getRemappedValue(localLoad.getSrc());
      if (!llStruct)
        return {Value(), 0, false};
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, llStruct,
          getTypeConverter()->convertType(memTy.getElementType()), rewriter);
      return {smemObj.getBase(), memTy.getShape()[1],
              isMultiSlotStaging(localLoad)};
    };

    auto [elemsA, aOffsets, aSrcEnc] =
        resolveOperand(op.getA(), adaptor.getA(), aType);
    auto [elemsB, bOffsets, bSrcEnc] =
        resolveOperand(op.getB(), adaptor.getB(), bType);
    auto elemsC = unpack(adaptor.getC());
    auto cOffsets = emitOffsetForLayout(cMmaEnc, cType);

    if (!aSrcEnc || !bSrcEnc)
      return failure();

    // Fragment ABI: C arrived as struct<(<64xf32> x F)> (the dot-chain
    // accumulator). The F fragment vectors seed matC_tiles directly and the
    // result packs back into the same struct — no scalar insertelement/
    // extractelement bridge on the register-resident paths.
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
    bool useDeviceA = (aPtrs.size() == elemsA.size() && !aPtrs.empty());
    bool useDeviceB = (bPtrs.size() == elemsB.size() && !bPtrs.empty());

    // Integer (int8) operands take the device path too: the f32
    // air.simdgroup_matrix_8x8_load cannot read 1-byte-packed integers, so the
    // device A/B loads are byte-loaded and widened to f32 by emitDevSGLoadInt8,
    // building the same <64 x f32> matrix the f32 MMA consumes. This drops the
    // large f32 TG operand buffer (which overflowed the 32KB budget for larger
    // K, e.g. K=128) entirely.

    // The device path keeps the per-warp C accumulator in registers
    // (laneLocalCIn / laneLocalCOut) and loads A and B straight from device
    // memory, so it never scatters operands through threadgroup memory. The TG
    // (batchStrips / per-strip) path coalesces its dot buffer with the
    // surrounding #mma<->#blocked convert_layout scratch and is preferred when
    // the whole operand grid fits the 32KB budget; for shapes whose resident
    // grid would overflow that budget (e.g. 128x128 single-CTA), the device
    // path is the only one that fits, so keep it there. This estimate mirrors
    // the batchedAB/batchedC sizing computed below.
    bool tgGridFitsBudget = false;
    {
      int64_t padEst = tgPadForType(aElemTy);
      int64_t maxStrideEst = std::max(K, N);
      int64_t unpaddedEst = 8 * maxStrideEst;
      bool canPadEst =
          (padEst > 0) && ((unpaddedEst + 8 * padEst) * 4 <= 16384);
      int64_t KpadEst = canPadEst ? K + padEst : K;
      int64_t NpadEst = canPadEst ? N + padEst : N;
      int64_t maxStripsEst = std::max(M / 8, K / 8);
      int64_t residentEst =
          std::max(maxStripsEst * 8 * std::max(KpadEst, NpadEst),
                   (M / 8) * 8 * NpadEst) *
          4;
      int64_t stagedBytes = 0;
      bool smemLive = true;
      if (auto a = mod->getAttrOfType<BoolAttr>("applegpu.smem_live"))
        smemLive = a.getValue();
      if (smemLive)
        if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
          stagedBytes = (int64_t)a.getValue().getZExtValue();
      tgGridFitsBudget = residentEst + stagedBytes <= kTGResidentBudgetBytes;
    }
    if (useDeviceA && useDeviceB && (M / 8) > 1 && tgGridFitsBudget) {
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

    // Pipelined-SMEM fallback for operands without device pointers: SG-load
    // directly from the staging buffer the local_load reads. Combined with the
    // device path this keeps the whole dot register-resident (no operand
    // scatter, no per-strip barriers). Gated on the SAME resident-grid budget
    // as the device path: when the padded TG grid fits, the batched TG path is
    // faster (the pipeline strips are unpadded, so SG loads from them
    // bank-conflict at small K pitches like 16/32 f32).
    Value aSmemBase, bSmemBase;
    int64_t aSmemPitch = 0, bSmemPitch = 0;
    bool aSmemMultiSlot = false, bSmemMultiSlot = false;
    if (!tgGridFitsBudget) {
      if (!useDeviceA)
        std::tie(aSmemBase, aSmemPitch, aSmemMultiSlot) =
            resolveSmemOperand(op.getA(), aType);
      if (!useDeviceB)
        std::tie(bSmemBase, bSmemPitch, bSmemMultiSlot) =
            resolveSmemOperand(op.getB(), bType);
    }
    bool useSmemA = !useDeviceA && static_cast<bool>(aSmemBase);
    bool useSmemB = !useDeviceB && static_cast<bool>(bSmemBase);
    bool fastA = useDeviceA || useSmemA;
    bool fastB = useDeviceB || useSmemB;
    bool fastPath = fastA && fastB;

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
    // extraRow/extraCol are optional runtime element addends (i32, default
    // null) folded into the tile address. The device-MMA path uses them to add
    // the per-warp MMA tile origin (warpRow*8 / warpCol*8) so each warp loads
    // only the operand tiles for the C tiles it actually owns.
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

    // ── Canonical warp -> tile mapping (SINGLE source of truth) ───────
    // The A/B operand scatter (makeBase, col-fastest branch), the C
    // accumulator (makeBaseMma), and the per-warp owned-tile ownership in the
    // TG/device MMA loops MUST all agree on which warp owns which 8x8 tile, or
    // operand rows are fed to the wrong MMA (see commit a72c17e: makeBase once
    // used Morton Z-order while makeBaseMma used linear, swapping warps 1<->2
    // for square pow2 warpsPerCTA and miscompiling nw=4). To make that class of
    // divergence impossible, BOTH call this one helper: warp w owns tile
    //   (warpRow, warpCol) = (w / wN, w % wN)
    // i.e. linear column-major warp order (warpOrder={1,0}), matching
    // AppleMmaEncodingAttr::toLinearLayout() and emitOffsetForLayout. Returns
    // the runtime (warpRow, warpCol) as i32. NOTE: makeBase's NON-col-fastest
    // branch deliberately uses the operand's transposed (row-major) warp order
    // for its own getOrder; that is operand-local and does not feed the MMA
    // tile ownership, so it is left as-is.
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

      // Warp decomposition: linear div/mod, matching the canonical blocked
      // layout that emitOffsetForLayout enumerates. The A/B scatter and the
      // simdgroup load round-trip through ABSOLUTE threadgroup positions (the
      // load reads tile (tk*8, tm*8) by absolute offset), so each logical
      // element must land at the absolute (row,col) that emitOffsetForLayout
      // assigns it. That layout uses row/col-major warp order (warpId / wN and
      // warpId % wN per the operand's getOrder), NOT Morton. A Morton warp
      // remap here disagrees with the absolute gather for any square
      // power-of-2 warpsPerCTA (e.g. [2,2]): warps 1 and 2 swap tiles, so the
      // MMA reads the wrong operand rows/cols and the whole result is wrong.
      // (The C accumulator in makeBaseMma already deliberately uses linear
      // order for this same reason.)
      Value wR, wC;
      if (colFastest) {
        // Canonical linear column-major order, shared with makeBaseMma.
        std::tie(wR, wC) = warpRowCol(wN);
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
      // Column-major warp tiling: matches toLinearLayout warpOrder={1,0}.
      // Shared with makeBase (col-fastest) via warpRowCol so the operand
      // scatter and this C mapping can never diverge again (commit a72c17e).
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

    // ── Create threadgroup global ─────────────────────────────────────
    // TG buffer needed for C scatter/load (always), and A/B scatter (TG path).
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
    // strips scatter and read back under one barrier pair instead of one pair
    // per strip. Trades threadgroup memory for fewer barriers. The batched
    // buffer coalesces with the layout-conversion scratch downstream, and the
    // compiler.py shared-memory gatekeeper measures that real post-coalesce
    // footprint and rejects any config that overflows 32KB, so this only needs
    // a sane upper bound to stop one dot claiming an absurd buffer.
    //
    // The batchStrips path keeps the whole C grid resident, which is exactly
    // what the per-warp owned-tile store-back (TASK #57) needs: each warp
    // writes only its owned tiles to their ABSOLUTE positions, then the
    // absolute-row gather reads each warp's elements back. The per-strip
    // (non-batched) path cannot do this (single-strip buffer, no resident
    // grid), so it still computes the full grid. To let the multi-warp configs
    // that miss the 16KB bound (e.g. BLOCK 64x64 num_warps=4, ~18KB) take the
    // per-warp batched path, the bound is 24KB: well under Metal's 32KB
    // threadgroup cap, and the real post-coalesce footprint is still validated
    // by the compiler.py gatekeeper. (24KB also bounds a single dot from
    // hogging the budget.)
    bool tgPath = !fastPath;
    int64_t maxStripsAB = std::max(M / 8, K / 8);
    // No garbage sink: the batchStrips and device paths scatter every element
    // to its absolute offset (directScatter) or keep C register-resident, so
    // the buffer holds only live data (no reserved out-of-strip slot).
    int64_t batchedABSize = maxStripsAB * tgABStripSize;
    int64_t batchedCSize = (M / 8) * tgCStripSize;
    // Gate the resident-grid path on its REAL combined footprint vs Metal's
    // 32KB threadgroup cap, not a bare grid bound: a sub-f32 dot also keeps two
    // half-typed convert_layout buffers (__tg_cvt_*, the #mma<->#blocked round
    // trip) resident, each about the C grid in operand bytes, and the launch
    // gatekeeper sums all of them. The dot grid itself is f32-typed (4B/elem).
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

    // Reserved out-of-tile sink slot (always the last buffer element). With
    // batched strips the buffer is larger, so the sink must be tgSize-1, not
    // tgStripSize (which now indexes live data).
    Value garbageIdx =
        arith::ConstantIntOp::create(rewriter, loc, tgSize - 1, 64);

    // ── Per-warp C-tile ownership (TASK #57) ──────────────────────────
    // Both the device path (above) and the TG path tile the output in a
    // warpsM x warpsN warp grid using the canonical linear mapping
    // (warpRowCol). Warp (warpRow, warpCol) owns the absolute C tiles
    //   (warpRow + k*warpsM, warpCol + j*warpsN),  k in [0,ownM), j in
    //   [0,ownN).
    // Previously EVERY warp computed the full tilesM x tilesN grid
    // (warpsM*warpsN x redundant MMA, the dominant nw=4 GEMM cost). Now each
    // warp computes only its owned ownM x ownN tiles, stored as
    // matC_tiles[k][j]. The TG store-back writes each owned tile to its
    // ABSOLUTE threadgroup position (so exactly one warp writes each tile, no
    // race) by folding the runtime warp origin into a base-pointer GEP, then
    // the existing absolute-row-filtered gather reads each warp's elements back
    // unchanged.
    int64_t cWarpsM = cMmaEnc.getWarpsPerCTA()[0];
    int64_t cWarpsN = cMmaEnc.getWarpsPerCTA()[1];
    int64_t ownM = std::max<int64_t>(1, tilesM / cWarpsM);
    int64_t ownN = std::max<int64_t>(1, tilesN / cWarpsN);
    // Per-warp ownership applies to the pure-TG BATCHED path only (the
    // batchStrips compute/store-back below): the device path has its own
    // per-warp lowering. The owned indexing degenerates to the full absolute
    // grid when warpsM*warpsN == 1 (warp origin 0, ownM=tilesM), so the
    // batchStrips path uses it unconditionally.
    // Runtime warp origin in elements (warpRow*8, warpCol*8) and a base pointer
    // pre-offset to this warp's (0,0) owned tile, used by the owned SG
    // load/store (constant tile offsets only; a runtime SG offset mis-lowers).
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
    // Owned base pointer for C tiles (only valid for the non-device TG path,
    // which round-trips C through the Npad-pitch TG buffer). The device path
    // builds its own per-warp pointers inline.
    Value ptrTGcWarp = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, f32Ty, ptrTG,
                                           ArrayRef<LLVM::GEPArg>{cWarpFlat64});
    // Owned A/B base pointers fold the runtime warp origin into the TG base so
    // each warp SG-loads only its owned operand tiles at CONSTANT tile offsets
    // (a runtime SG offset mis-lowers). A is scattered [strip=tile-row, Kpad]:
    // owned tile-rows are absolute {warpRow + k*warpsM}, so pre-offset by
    // warpRow*8 rows (flat warpRow*8 * Kpad). B is scattered [strip=tk, Npad]:
    // owned tile-cols are absolute {warpCol + j*warpsN}, a pure column offset
    // (flat warpCol*8). The GEP element type is the A/B scatter type so the
    // flat index is scaled in scatter elements.
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

    // Direct scatter: write every operand element to its absolute threadgroup
    // offset (baseRow+rowOff)*stride + (baseCol+colOff) with a plain
    // unconditional store. baseRow/baseCol are the runtime warp/lane origin and
    // rowOff/colOff are the compile-time layout offsets, so this offset is the
    // closed-form distributed->shared mapping (the invertAndCompose result for
    // the row-major TG strip): no predicate, no garbage sink, no per-strip
    // replication. The matching SG-load reads each 8x8 tile back from the same
    // absolute coordinates.
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
      // inStrip depends only on rowOff (actualRow = baseRow + rowOff with
      // baseRow/rowStart constant in this call), so all elements sharing a
      // rowOff share one predicate. Group the bucket by rowOff and guard each
      // group with a single conditional block: one cond_br per distinct row
      // instead of per element, which keeps the block count proportional to
      // strip rows rather than to the full tile.
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

    // LANE-LOCAL C bridge (physical AppleMma layout), shared by every MMA path.
    // With the physical toLinearLayout active, each lane's #mma C scalars sit
    // at exactly the simdgroup_matrix per-lane storage (vector indices 0,1), so
    // the C accumulator-in is a register pack (insertelement) and the C-out is
    // the inverse (extractelement) at vecIdx = colOff%2 (physical register bit
    // = col bit0). No TG round-trip, no cross-lane shuffle: same correct bridge
    // the batchStrips path uses, applied uniformly so every path matches the
    // global physical layout. owM/owN are the per-warp owned tile counts; wM/wN
    // the warp grid (single-warp grids degenerate to the full absolute grid).
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

    // DEVICE PATH per-warp C-accumulator-in: the device MMA (Phase 2) computes
    // and the shuffle extract (Phase 4) read ONLY this warp's owned tiles
    // matC_tiles[k][j] = absolute tile (warpRow+k*warpsM, warpCol+j*warpsN).
    // The C accumulator (e.g. add-matrix's convert_layout #blocked->#mma
    // feeding tt.dot's C operand) must therefore be gathered into the SAME
    // owned slots, not the full absolute grid, or a nonzero accumulator lands
    // in the wrong tile (invisible when C-in is zero; the add-matrix/rows/cols
    // wall).
    if (fastPath) {
      // LANE-LOCAL C-in (physical AppleMma layout). The device MMA (Phase 2)
      // and the Phase-4 extract read each warp's owned tiles matC_tiles[k][j];
      // the C accumulator-in must land in the SAME owned slots. Under the
      // physical toLinearLayout cOffsets are physical coords, so this is a
      // plain register pack (insertelement) with no TG round-trip, replacing
      // the old logical TG scatter/SG-gather (which double-permuted under the
      // physical layout).
      laneLocalCIn(cWarpsM, cWarpsN, ownM, ownN);
    } else if (batchStrips) {
      // ── LANE-LOCAL C-in (physical AppleMma layout) ───────────────────────
      // With the physical toLinearLayout active, each lane's #mma scalars sit
      // at exactly the simdgroup_matrix per-lane storage (vector indices 0,1),
      // so the C accumulator-in is a register pack (insertelement) with no TG
      // round-trip and no cross-lane shuffle.  vecIndex = colOff%2 (the
      // register bit = col bit0); owned local tile = (rowOff/8 - warpRowOff,
      // colOff/8 - warpColOff) but since cOffsets are warp-relative the local
      // tile is (rowOff/8, colOff/8) folded into ownership below.
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
        // cOffsets enumerate this lane's owned positions across the full tensor
        // tile grid; for per-warp ownership the entry's absolute tile maps to a
        // local owned slot. Single-warp grids (ownM=tilesM) map identically.
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
      // ── LANE-LOCAL C-in (physical AppleMma layout), per-strip path ──────
      // PER-WARP OWNED-COLUMN compute (Part 2): the per-strip path keeps A
      // full (every tile-row resident in registers) but loads only this warp's
      // owned B tile-cols, so matC_tiles is indexed [absolute tile-row tm]
      // [owned local tile-col j] over tilesM x ownN. Seed each entry with this
      // lane's C accumulator: the owning tile-ROW is runtime (cBaseRow), so a
      // runtime equality picks the matching compile-time row tm, while the
      // owned local column j = (colOff/8)/cWarpsN is compile-time. A register
      // pack (insertelement), no TG round-trip; the column select-chain that
      // the full-grid path used is gone (only the row compare remains).
      auto matTyP = getSimdgroupMatrixType(ctx);
      auto initFnP = getOrInsertIntrinsic(
          rewriter, mod, "air.simdgroup_matrix_8x8_init_filled.v64f32.f32",
          LLVMFunctionType::get(matTyP, {f32Ty}, false));
      Value fzP =
          arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(f32Ty));
      Value zMatP =
          LLVM::CallOp::create(rewriter, loc, initFnP, ValueRange{fzP})
              .getResult();
      for (int64_t tm = 0; tm < tilesM; ++tm)
        for (int64_t j = 0; j < ownN; ++j)
          matC_tiles[tm][j] = zMatP;
      Value c8p = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
      Value absTileRowP = arith::DivUIOp::create(rewriter, loc, cBaseRow, c8p);
      // Fragment C-in: the struct holds this warp's owned tiles (k in [0,ownM),
      // absolute row absTileRow + k*warpsM). Seed the absolute-indexed
      // matC_tiles[tm][j] by a runtime row-equality select so each fragment
      // lands in its true tile-row tm (the non-fragment path seeds per-element
      // below).
      if (fragC) {
        for (int64_t k = 0; k < ownM; ++k)
          for (int64_t j = 0; j < ownN; ++j) {
            int64_t fi = k * ownN + j;
            if (fi >= (int64_t)fragCIn.size())
              continue;
            Value wantTm = arith::AddIOp::create(
                rewriter, loc, absTileRowP,
                arith::ConstantIntOp::create(rewriter, loc, k * cWarpsM, 32));
            for (int64_t tm = 0; tm < tilesM; ++tm) {
              Value tmEq = arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::eq, wantTm,
                  arith::ConstantIntOp::create(rewriter, loc, tm, 32));
              matC_tiles[tm][j] = LLVM::SelectOp::create(
                  rewriter, loc, tmEq, fragCIn[fi], matC_tiles[tm][j]);
            }
          }
      }
      for (size_t idx = 0; !fragC && idx < cOffsets.size(); ++idx) {
        int64_t rowOff = cOffsets[idx][0];
        int64_t colOff = cOffsets[idx][1];
        int64_t localTn = (colOff / 8) / std::max<int64_t>(1, cWarpsN);
        if (localTn >= ownN)
          continue;
        Value wantTm = arith::AddIOp::create(
            rewriter, loc, absTileRowP,
            arith::ConstantIntOp::create(rewriter, loc, rowOff / 8, 32));
        int64_t vecIdx = colOff % 2;
        Value vIdxC = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
        Value v = elemsC[idx];
        if (v.getType() != f32Ty)
          v = toF32(rewriter, loc, v, f32Ty);
        for (int64_t tm = 0; tm < tilesM; ++tm) {
          Value tmEq = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, wantTm,
              arith::ConstantIntOp::create(rewriter, loc, tm, 32));
          Value cur = ExtractElementOp::create(rewriter, loc,
                                               matC_tiles[tm][localTn], vIdxC);
          Value nv = arith::SelectOp::create(rewriter, loc, tmEq, v, cur);
          matC_tiles[tm][localTn] = InsertElementOp::create(
              rewriter, loc, matTyP, matC_tiles[tm][localTn], nv, vIdxC);
        }
      }
    }

    // ── Phase 2: A/B loads + MMA ──────────────────────────────────
    if (fastPath) {
      // ── DEVICE PATH: Direct MMA loads from device memory ──────────
      // No TG scatter, no barriers for A/B.
      //
      // PER-WARP TILING: the MMA warps tile the output in a warpsM x warpsN
      // grid, so warp (warpRow,warpCol) owns the absolute C tiles
      //   (warpRow + k*warpsM, warpCol + j*warpsN),  k in [0,ownM), j in
      //   [0,ownN)
      // Earlier this loop computed the FULL tilesM x tilesN grid in every warp
      // and let the Phase-4 select-chain pick each lane's owned tile, doing
      // warpsM*warpsN x redundant MMA. Here each warp loads and accumulates
      // ONLY its owned tiles (matC_tiles[k][j] = owned tile (k,j)). The owned
      // tile's device origin adds the runtime warp offset (cWarpRow*8 /
      // cWarpCol*8) on top of the strided compile-time step (k*warpsM*8 /
      // j*warpsN*8). Prefetch: A tiles for tk+1 loaded before MMA of tk.
      Value aDevStride, bDevStride, aDevTranspose, bDevTranspose;
      if (useDeviceA) {
        aDevStride = makeDevMmaStride(aColStride, aRowStride);
        aDevTranspose = makeDevMmaTranspose(aColStride, aRowStride);
      }
      if (useDeviceB) {
        bDevStride = makeDevMmaStride(bColStride, bRowStride);
        bDevTranspose = makeDevMmaTranspose(bColStride, bRowStride);
      }
      Value mmaShape = makeI64Vec2(rewriter, loc, 8, 8);
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

      // SMEM warp-base pointers: fold the runtime warp origin into the base so
      // the SG loads use constant tile offsets (a runtime SG offset
      // mis-lowers). A's origin is a row offset (warpRow*8 rows of pitch
      // aSmemPitch); B's is a pure column offset (warpCol*8).
      Value ptrSmemAWarp, ptrSmemBWarp;
      if (useSmemA) {
        Value pitchA =
            arith::ConstantIntOp::create(rewriter, loc, aSmemPitch, 32);
        Value flat = arith::ExtUIOp::create(
            rewriter, loc, i64Ty,
            arith::MulIOp::create(rewriter, loc, warpRowElem, pitchA));
        ptrSmemAWarp =
            LLVM::GEPOp::create(rewriter, loc, tgPtrTy, abTgScatterTy,
                                aSmemBase, ArrayRef<LLVM::GEPArg>{flat});
      }
      if (useSmemB) {
        Value flat = arith::ExtUIOp::create(rewriter, loc, i64Ty, warpColElem);
        ptrSmemBWarp =
            LLVM::GEPOp::create(rewriter, loc, tgPtrTy, abTgScatterTy,
                                bSmemBase, ArrayRef<LLVM::GEPArg>{flat});
      }

      // Tile loaders dispatch per operand: pipelined SMEM strips load with the
      // TG simdgroup intrinsic at constant pitch; device operands keep the
      // device-direct load. Both produce the same v64 matrix type, so they
      // feed one MMA fn.
      auto loadATile = [&](int64_t k, int64_t tk) -> Value {
        if (useSmemA) {
          Value aOff = makeI64Vec2(rewriter, loc, tk * 8, k * warpsM * 8);
          return emitSGLoad(abTgLoadFn, ptrSmemAWarp, aSmemPitch, aSmemPitch,
                            aOff);
        }
        Value aTilePtr = computeTileDevPtr(
            aPtrs, aOffsets, aRowStride, aColStride, aBaseRow, aBaseCol,
            k * warpsM * 8, tk * 8, warpRowElem, nullptr);
        return isIntInput ? emitDevSGLoadInt8(aTilePtr, aRowStride, aColStride)
                          : emitDevSGLoad(devLoadFn, aTilePtr, mmaShape,
                                          aDevStride, zeroOff, aDevTranspose);
      };
      auto loadBTile = [&](int64_t tk, int64_t j) -> Value {
        if (useSmemB) {
          Value bOff = makeI64Vec2(rewriter, loc, j * warpsN * 8, tk * 8);
          return emitSGLoad(abTgLoadFn, ptrSmemBWarp, bSmemPitch, bSmemPitch,
                            bOff);
        }
        Value bTilePtr = computeTileDevPtr(
            bPtrs, bOffsets, bRowStride, bColStride, bBaseRow, bBaseCol, tk * 8,
            j * warpsN * 8, nullptr, warpColElem);
        return isIntInput ? emitDevSGLoadInt8(bTilePtr, bRowStride, bColStride)
                          : emitDevSGLoad(devLoadFn, bTilePtr, mmaShape,
                                          bDevStride, zeroOff, bDevTranspose);
      };

      if (useSmemA || useSmemB) {
        // SMEM-resident operands: SG-load EVERY owned strip tile upfront, then
        // one barrier, then the all-register MMA loop. The reads happen AFTER
        // the local_load the membar pass fenced, so without a barrier the next
        // pipeline copy can overwrite the staging strip mid-read (silent
        // corruption, K-iteration-count dependent). Loading first and fencing
        // once keeps the MMA off the barrier's critical path so the next
        // copies overlap the math.
        //
        // With multi-slot staging (num_stages>=3) the next K-step's copy
        // targets a DIFFERENT slot than these reads, and the step in between
        // executes an async_wait barrier before its own loads, which already
        // separates these reads from the next write to the same slot two
        // steps later. The post-load barrier is then redundant and elided;
        // single-slot staging keeps it.
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
        bool slotsRotate =
            (!useSmemA || aSmemMultiSlot) && (!useSmemB || bSmemMultiSlot);
        if (!slotsRotate)
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
        // ── ROLLED K-LOOP (task #86) ──────────────────────────────────────
        // Emit the K-step accumulation as a REAL LLVM-dialect loop with the
        // owned simdgroup-matrix accumulators carried as loop block arguments
        // (phi-style iter_args), instead of straight-line unrolling all tilesK
        // steps. The fully-unrolled form gave LLVM-O3 N identical copies of the
        // <64 x float> matrix tiles to scalarize/revectorize (extract-all/
        // insert-all churn), inflating register pressure so the AGX driver
        // capped maxThreadsPerThreadgroup (~384 vs ~896). With a real loop the
        // accumulator is a single persistent phi value O3 cannot amplify.
        //
        // A bounded compile-time unroll factor U (METAL_MMA_KUNROLL, default 2)
        // keeps latency hiding (several independent A/B loads + MMAs in flight
        // per trip) without the full explosion. The device-direct A/B loads
        // fold the runtime K offset (tk*8) into the tile pointer, so the loads
        // ride the runtime induction variable.
        //
        // Rolled only when tilesK is a multiple of U and the resulting trip
        // count is >= 2; otherwise (small/odd K) fall back to the original
        // straight-line unroll, where rolling buys nothing.
        int64_t kUnroll = 2;
        if (const char *e = ::getenv("METAL_MMA_KUNROLL")) {
          int64_t v = atoll(e);
          if (v >= 1)
            kUnroll = v;
        }
        // Clamp the unroll factor to a divisor of tilesK so the rolled trip
        // count is exact (no remainder tail to special-case).
        while (kUnroll > 1 && (tilesK % kUnroll) != 0)
          --kUnroll;
        int64_t tripCount = tilesK / kUnroll;
        bool doRoll = (tilesK > kUnroll) && (tripCount >= 2);

        // Runtime-K tile loaders: same as loadATile/loadBTile but the K offset
        // is a runtime Value (tk*8) folded into computeTileDevPtr's free extra
        // offset (A's column / B's row), so the load follows the loop IV.
        // The K offset rides a loop-carried A/B reference pointer (aKRef/bKRef)
        // advanced by kUnroll*8 K-elements per trip, so the per-trip device
        // address is a pointer-increment recurrence (ptr += const) instead of a
        // fresh mul(tk,stride)+gep rebuild. tkElemOff is kept only as a
        // fallback for callers that have no carried base (none on the rolled
        // path).
        auto loadATileRT = [&](int64_t k, Value tkElemOff,
                               Value aKRef = nullptr) -> Value {
          Value aTilePtr = computeTileDevPtr(
              aPtrs, aOffsets, aRowStride, aColStride, aBaseRow, aBaseCol,
              k * warpsM * 8, /*tileCol=*/0, warpRowElem,
              /*extraCol=*/aKRef ? nullptr : tkElemOff, aKRef);
          return isIntInput
                     ? emitDevSGLoadInt8(aTilePtr, aRowStride, aColStride)
                     : emitDevSGLoad(devLoadFn, aTilePtr, mmaShape, aDevStride,
                                     zeroOff, aDevTranspose);
        };
        auto loadBTileRT = [&](Value tkElemOff, int64_t j,
                               Value bKRef = nullptr) -> Value {
          Value bTilePtr = computeTileDevPtr(
              bPtrs, bOffsets, bRowStride, bColStride, bBaseRow, bBaseCol,
              /*tileRow=*/0, j * warpsN * 8,
              /*extraRow=*/bKRef ? nullptr : tkElemOff, warpColElem, bKRef);
          return isIntInput
                     ? emitDevSGLoadInt8(bTilePtr, bRowStride, bColStride)
                     : emitDevSGLoad(devLoadFn, bTilePtr, mmaShape, bDevStride,
                                     zeroOff, bDevTranspose);
        };

        if (!doRoll) {
          // Original straight-line unroll (small/odd K — rolling buys nothing).
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
          // Real loop. Carried values: the ownM*ownN accumulator matrices,
          // flattened row-major as [k*ownN + j]. The induction variable is the
          // K-step index tk (i32), stepping by kUnroll each trip.
          int64_t nAcc = ownM * ownN;
          auto accInit = [&](int64_t k, int64_t j) -> Value {
            return matC_tiles[k][j];
          };

          // Advance a device pointer by `kElems` K-elements: along A's column
          // axis (extraCol -> colStride) and B's row axis (extraRow ->
          // rowStride). A loop-invariant constant GEP, so the per-trip pointer
          // step is a single ptr += const recurrence (no mul(tk,stride)
          // rebuild).
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
          // block carries the SAME signature (the cond_br false edge forwards
          // tk + the final accumulators), so add matching args to it.
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
            // sub-step within the trip: the carried A/B pointers already encode
            // K=tk; advance by the compile-time-constant u*8 K-elements for the
            // sub-step (folds into the load GEP, no runtime mul).
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
          LLVM::BrOp::create(rewriter, loc, backArgs, header);

          // Exit: harvest the final accumulators from the header-forwarded
          // args.
          rewriter.setInsertionPointToStart(exitBlock);
          for (int64_t k = 0; k < ownM; ++k)
            for (int64_t j = 0; j < ownN; ++j)
              matC_tiles[k][j] = exitBlock->getArgument(1 + k * ownN + j);
        }
      }
    } else if (batchStrips) {
      // ── TG PATH (batched): scatter the whole A operand into per-strip TG
      // blocks once, one barrier, SG-load every (tm,tk) tile into registers,
      // one barrier; then the same for B, fused with MMA. This collapses the
      // per-strip / per-k barrier pairs (O(tilesK*tilesM)) down to 4 barriers
      // for the entire dot, which is the dominant per-K-step cost. ──
      // A: scatter every element to its absolute (row, col) once, one barrier.
      directScatter(ptrTG, aBaseRow, aBaseCol, elemsA, aOffsets, Kpad,
                    abTgScatterTy);
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      // PER-WARP (TASK #57): load only this warp's owned A tile-rows
      // matA[k] = absolute tile-row (warpRow + k*warpsM), via the warp-base A
      // pointer (warpRow*8 rows folded in) at constant offset (tk*8,
      // k*warpsM*8). Every warp still scattered the full A grid above, so all
      // strips are resident; only the load/MMA is now per-warp.
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

      // B: scatter every element to its absolute (row, col) once, one barrier.
      directScatter(ptrTG, bBaseRow, bBaseCol, elemsB, bOffsets, Npad,
                    abTgScatterTy);
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      // PER-WARP: each warp loads only its owned B tile-cols (absolute col
      // warpCol + j*warpsN) via the warp-base B pointer (warpCol*8 cols folded
      // in) and accumulates only its owned C tiles matC_tiles[k][j].
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
      // ── TG PATH (per-strip): scatter/load through threadgroup memory ──
      // PER-WARP OWNED-COLUMN COMPUTE (Part 2): a warp owns the absolute C
      // tile-cols (cWarpCol + j*cWarpsN), j in [0,ownN), so only those output
      // tile-cols survive the convert back to #blocked. The single B strip is
      // fully resident (8 x Npad, all N columns), so each warp SG-loads only
      // its owned tile-cols via the warp-base B pointer (ptrTGbWarp folds the
      // runtime cWarpCol*8 column origin in) at CONSTANT offsets, exactly as
      // the batchStrips path does, with no matrix-domain select (which the AIR
      // lowering rejects). A stays full per tile-row (every A tile-row is
      // resident in registers across all warps), so matC_tiles is indexed
      // [absolute tile-row tm][owned local tile-col j] over tilesM x ownN. The
      // MMA count drops by cWarpsN and the Phase-4 C-out column select-chain is
      // gone (only the runtime tile-row compare remains).
      int64_t aMaxBase = maxBaseRow(aSrcEnc);
      int64_t bMaxBase = maxBaseRow(bSrcEnc);
      auto aBuckets = bucketElements(aOffsets, aMaxBase, tilesM, 0);
      auto bBuckets = bucketElements(bOffsets, bMaxBase, tilesK, 0);

      // Each A strip is an 8 x Kpad block, so a single scatter of strip tm
      // holds every tk tile of that strip; load all of them into registers
      // before the next strip overwrites the shared TG buffer. This replaces
      // the old scatter-A-per-tk loop (which re-scattered the whole A operand
      // tilesK times because the B scatter clobbered the buffer each step),
      // collapsing O(tilesK*tilesM) A scatters/barriers down to O(tilesM).
      SmallVector<SmallVector<Value>> matA(tilesM);
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        matA[tm].resize(tilesK);
        int64_t rowStart = tm * 8;
        filteredScatter(ptrTG, garbageIdx, aBaseRow, aBaseCol, elemsA, aOffsets,
                        aBuckets[tm], Kpad, rowStart, abTgScatterTy);
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
        for (int64_t tk = 0; tk < tilesK; ++tk) {
          Value aOff = makeI64Vec2(rewriter, loc, tk * 8, 0);
          matA[tm][tk] = emitSGLoad(abTgLoadFn, ptrTG, Kpad, Kpad, aOff);
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

        // Load only this warp's owned B tile-cols (absolute col cWarpCol +
        // j*cWarpsN) via the warp-base B pointer at a CONSTANT tile offset, and
        // accumulate only the owned columns matC_tiles[tm][j].
        for (int64_t j = 0; j < ownN; ++j) {
          Value bOff = makeI64Vec2(rewriter, loc, j * cWarpsN * 8, 0);
          Value matB = emitSGLoad(abTgLoadFn, ptrTGbWarp, Npad, Npad, bOff);
          for (int64_t tm = 0; tm < tilesM; ++tm)
            matC_tiles[tm][j] =
                LLVM::CallOp::create(
                    rewriter, loc, abTgMmaFn,
                    ValueRange{matA[tm][tk], matB, matC_tiles[tm][j]})
                    .getResult();
        }
        LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                             ValueRange{fenceTG, execMod});
      }
    }

    // ── Phase 4: Extract C results from MMA tiles ────────────────────
    auto outElemTy = cType.getElementType();

    // Fragment ABI: pack the owned matC_tiles vectors straight back into the
    // fragment struct (no scalar extractelement bridge), keeping the
    // accumulator vectorized through the loop carry.
    if (fragC) {
      auto outStructTy =
          cast<LLVMStructType>(getTypeConverter()->convertType(cType));
      Value result = UndefOp::create(rewriter, loc, outStructTy);
      int64_t F = (int64_t)outStructTy.getBody().size();
      // fastPath / batchStrips index matC_tiles warp-locally ([0,ownM)), so the
      // owned tile (k,j) is matC_tiles[k][j] directly. The per-strip path
      // instead computes the FULL absolute tile-ROW grid (matC_tiles indexed
      // [0,tilesM)), with this warp's owned rows selected at runtime by
      // cBaseRow — mirror the non-fragment per-strip C-out's runtime
      // row-equality select so warp r packs ITS rows (warpRow + k*warpsM), not
      // absolute rows 0..ownM.
      bool perStrip = !fastPath && !batchStrips;
      Value absTileRow;
      if (perStrip) {
        Value c8 = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
        absTileRow = arith::DivUIOp::create(rewriter, loc, cBaseRow, c8);
      }
      for (int64_t k = 0; k < ownM; ++k)
        for (int64_t j = 0; j < ownN; ++j) {
          int64_t fi = k * ownN + j;
          if (fi >= F)
            continue;
          Value frag;
          if (perStrip) {
            Value wantTm = arith::AddIOp::create(
                rewriter, loc, absTileRow,
                arith::ConstantIntOp::create(rewriter, loc, k * cWarpsM, 32));
            frag = getOrInsertSimdgroupInitFilled(rewriter, loc, mod);
            for (int64_t tm = 0; tm < tilesM; ++tm) {
              Value tmEq = arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::eq, wantTm,
                  arith::ConstantIntOp::create(rewriter, loc, tm, 32));
              frag = LLVM::SelectOp::create(rewriter, loc, tmEq,
                                            matC_tiles[tm][j], frag);
            }
          } else {
            frag = matC_tiles[k][j];
          }
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

    // Under the physical AppleMma toLinearLayout the #mma C scalars sit at the
    // simdgroup_matrix per-lane storage (vector indices 0,1), so the C-out is a
    // lane-local extractelement for every register-resident path (device and
    // batchStrips). The shared/per-strip TG-store paths handle the bridge
    // below.
    if (fastPath) {
      // ── LANE-LOCAL C-out (physical AppleMma layout) ───
      // Inverse of the lane-local device C-in: each lane's owned tile holds its
      // two physical C scalars at vector indices {0,1}, indexed by colOff%2.
      // Under the physical toLinearLayout cOffsets are physical coords, so this
      // is a direct extractelement, replacing the old logical->physical shuffle
      // network (which double-permuted now that the layout is already
      // physical).
      laneLocalCOut(cWarpsM, cWarpsN, ownM, ownN, outElemTy, resultElems);
    } else if (batchStrips) {
      // ── LANE-LOCAL C-out (physical AppleMma layout) ──────────────────────
      // Inverse of the lane-local C-in: extractelement at the same vecIdx, no
      // TG round-trip, no shuffle.
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
      // ── LANE-LOCAL C-out (physical AppleMma layout), per-strip path ─────
      // Inverse of the per-strip owned C-in: matC_tiles is indexed [absolute
      // tile-row tm][owned local tile-col j] (the per-warp owned-column compute
      // in Phase 2). The owning tile-ROW is runtime (cBaseRow), so select the
      // matching compile-time row tm by a runtime equality chain over tilesM,
      // then extractelement at the owned local column and physical vecIdx. The
      // O(tilesN) column scan the full-grid path used is gone; only the
      // O(tilesM) row compare remains.
      Value c8 = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
      Value absTileRow = arith::DivUIOp::create(rewriter, loc, cBaseRow, c8);
      for (size_t idx = 0; idx < cOffsets.size(); ++idx) {
        int64_t rowOff = cOffsets[idx][0];
        int64_t colOff = cOffsets[idx][1];
        int64_t localTn = (colOff / 8) / std::max<int64_t>(1, cWarpsN);
        if (localTn >= ownN)
          continue;
        Value wantTm = arith::AddIOp::create(
            rewriter, loc, absTileRow,
            arith::ConstantIntOp::create(rewriter, loc, rowOff / 8, 32));
        int64_t vecIdx = colOff % 2;
        Value vIdxC = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
        Value sel = arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getZeroAttr(f32Ty));
        for (int64_t tm = 0; tm < tilesM; ++tm) {
          Value tmEq = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, wantTm,
              arith::ConstantIntOp::create(rewriter, loc, tm, 32));
          Value v = ExtractElementOp::create(rewriter, loc,
                                             matC_tiles[tm][localTn], vIdxC);
          sel = arith::SelectOp::create(rewriter, loc, tmEq, v, sel);
        }
        Value val = sel;
        if (val.getType() != outElemTy)
          val = fromF32(rewriter, loc, val, outElemTy);
        resultElems[idx] = val;
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

// Fragment ABI: a splat constant feeding the dot-chain (the zero accumulator
// init) lowers to a struct of F init-filled simdgroup fragments. Matches only
// when the type converter has chosen the fragment struct for this #mma tensor;
// otherwise defers to the generic flat constant lowering.
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
