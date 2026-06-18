#include "ConvertCommon.h"
#include "Dialect/TritonAppleGPU/IR/AppleMmaFragment.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "TritonAppleGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::applegpu {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

namespace {

// ConvertLayoutOp: identity pass-through for DotOperandEncoding targets,
// TG scatter/gather redistribution for blocked->blocked.
struct ConvertLayoutOpAppleConversion
    : public mlir::ConvertOpToLLVMPattern<ttg::ConvertLayoutOp> {
  using mlir::ConvertOpToLLVMPattern<
      ttg::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  static unsigned &getCounter(MLIRContext *ctx) {
    static llvm::DenseMap<MLIRContext *, unsigned> counters;
    return counters[ctx];
  }

  // Stable pool key for a shared convert TG global. Conversions with the same
  // TG element type alias one buffer (their live ranges are barrier-disjoint),
  // so the key only needs to distinguish element widths and pointer storage.
  static std::string getCvtPoolKey(Type tgElemTy) {
    if (isa<LLVMPointerType>(tgElemTy))
      return "p";
    return ("i" + llvm::Twine(tgElemTy.getIntOrFloatBitWidth())).str();
  }

  LogicalResult
  matchAndRewrite(ttg::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getResult().getType());

    // DotOperandEncoding target: identity pass-through from blocked source (the
    // dot lowering peels the cvt). NOT for an MMA source - its per-thread
    // element count differs, so the shared LinearLayout cvt must handle it.
    if (isa<ttg::DotOperandEncodingAttr>(dstTy.getEncoding()) &&
        isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding())) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    // A/B operands of an AppleMma dot need no lowering: the dot pattern reads
    // them from the cvt SOURCE, so the converted struct is dead. Lowering it
    // anyway burns a __tg_cvt strip that pushes pipelined kernels over the
    // 32KB TG cap.
    if (!op->use_empty() &&
        isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding())) {
      bool onlyMmaABUses = true;
      for (OpOperand &use : op->getUses()) {
        auto dot = dyn_cast<triton::DotOp>(use.getOwner());
        if (!dot || use.getOperandNumber() >= 2 ||
            !isa<AppleMmaEncodingAttr>(
                cast<RankedTensorType>(dot.getType()).getEncoding())) {
          onlyMmaABUses = false;
          break;
        }
      }
      if (onlyMmaABUses) {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
    }

    // 1-D #ttg.slice -> #ttg.slice (reduce-result restripe) routes through the
    // LL-indexed TG scatter/gather: the upstream smem path miscompiles it
    // (store collapses the reduce dim, gather does not -> stale-slot reads,
    // -inf reduce-init leaks into the result).
    if (srcTy.getShape().size() == 1 &&
        isa<ttg::SliceEncodingAttr>(srcTy.getEncoding()) &&
        isa<ttg::SliceEncodingAttr>(dstTy.getEncoding()) &&
        !isa<triton::PointerType>(srcTy.getElementType()) &&
        cvtNeedsSharedMemory(srcTy, dstTy) &&
        srcTy.getEncoding() != dstTy.getEncoding()) {
      auto loc = op.getLoc();
      auto *ctx = op.getContext();
      auto mod = op->getParentOfType<ModuleOp>();
      return convertLayout1DGeneric(op, adaptor, rewriter, srcTy.getEncoding(),
                                    dstTy.getEncoding(), srcTy, dstTy, loc, ctx,
                                    mod);
    }

    // blocked→blocked redistribution via TG scatter/gather, plus the
    // #mma (AppleMma) → #blocked C-output conversion (see mma branch below).
    auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    auto srcMmaEnc = dyn_cast<AppleMmaEncodingAttr>(srcTy.getEncoding());
    auto dstEnc = dyn_cast<ttg::BlockedEncodingAttr>(dstTy.getEncoding());
    if (!dstEnc || (!srcEnc && !srcMmaEnc))
      return failure();

    // Within-simdgroup (or pure register-shuffle) converts need no shared
    // memory: defer to the upstream simd_shuffle pattern. Exceptions stay on
    // the in-tree TG path below:
    //   - !tt.ptr elements: the shuffle pattern can't move them.
    //   - #mma sources: the shuffle pattern's ColumnAction asserts on the
    //     fragment struct element count (LinearLayout.cpp ColumnAction::apply);
    //     the TG-scatter slot map is correct (just slower).
    if (!srcMmaEnc && !isa<triton::PointerType>(srcTy.getElementType()) &&
        !cvtNeedsSharedMemory(srcTy, dstTy))
      return failure();

    if (srcEnc && srcEnc == dstEnc) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();
    auto shape = srcTy.getShape();

    if (shape.size() == 1) {
      return convertLayout1D(op, adaptor, rewriter, srcEnc, dstEnc, srcTy,
                             dstTy, loc, ctx, mod);
    }

    // #mma -> W-blocked store epilogue: every C element stays in its own
    // simdgroup, so lower as a register-only air.simd_shuffle restripe
    // (oracle-validated in tools/fragment-oracle). Any unexpected shape falls
    // through to the always-correct TG path below.
    if (srcMmaEnc && !cvtNeedsSharedMemory(srcTy, dstTy)) {
      if (succeeded(convertMmaToWShuffle(op, adaptor, rewriter, srcMmaEnc,
                                         dstEnc, srcTy, dstTy, loc, ctx, mod)))
        return success();
    }

    // ND blocked->blocked on the trailing two dims; rank>2 requires leading
    // dims size 1 (single rows x cols tile). Upstream
    // transferWithinBlockSwizzling miscompiles fp16/bf16 replicated layouts
    // here (stores <2 x half> at a 2-byte stride, reads back at 4-byte,
    // corrupting the row+16 slot); our fully-scalar (row,col) scatter/gather is
    // correct regardless of replication.
    unsigned rank = shape.size();
    if (rank < 2)
      return failure();
    unsigned rd = rank - 2;
    unsigned cd = rank - 1;
    for (unsigned d = 0; d < rd; ++d)
      if (shape[d] != 1)
        return failure();

    int64_t rows = shape[rd], cols = shape[cd];
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto tgPtrTy = LLVMPointerType::get(ctx, 3);

    // For pointer elements, use i64 in TG (Metal can't store ptrs in TG)
    bool isPointerElem = isa<LLVMPointerType>(elemTy);
    Type tgElemTy = isPointerElem ? i64Ty : elemTy;

    auto laneIdFnTy = LLVMFunctionType::get(i32Ty, {}, false);
    LLVMFuncOp laneIdFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn =
              mod.lookupSymbol<LLVMFuncOp>("air.thread_index_in_simdgroup"))
        laneIdFn = fn;
      else
        laneIdFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.thread_index_in_simdgroup",
                                      laneIdFnTy, Linkage::External);
    }

    auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    LLVMFuncOp tidFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>(
              "air.thread_position_in_threadgroup"))
        tidFn = fn;
      else
        tidFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                   "air.thread_position_in_threadgroup",
                                   tidFnTy, Linkage::External);
    }

    auto barrFnTy =
        LLVMFunctionType::get(LLVMVoidType::get(ctx), {i32Ty, i32Ty}, false);
    LLVMFuncOp tgBarrFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.threadgroup.barrier"))
        tgBarrFn = fn;
      else
        tgBarrFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.threadgroup.barrier", barrFnTy,
                                      Linkage::External);
    }

    Value laneId =
        LLVM::CallOp::create(rewriter, loc, laneIdFn, ValueRange{}).getResult();
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                               ArrayRef<int64_t>{0});
    Value c32 = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
    Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

    // Strip height = largest multiple of 8 fitting the 32KB TG budget (fewer
    // strips = fewer barriers), minus live global_smem and MMA dot TG buffers.
    constexpr int64_t tgBudgetBytes = 32 * 1024;
    int64_t smemBytes = 0;
    bool smemLive = true;
    if (auto attr = mod->getAttrOfType<BoolAttr>("applegpu.smem_live"))
      smemLive = attr.getValue();
    if (smemLive)
      if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
        smemBytes = attr.getValue().getZExtValue();
    int64_t mmaBytes = 0;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.mma_shared"))
      mmaBytes = attr.getValue().getZExtValue();
    int64_t availBytes = tgBudgetBytes - smemBytes - mmaBytes;
    int64_t elemBytes = isPointerElem ? 8 : elemTy.getIntOrFloatBitWidth() / 8;
    int64_t capRows = (availBytes / elemBytes) / cols;
    int64_t maxStripRows;
    if (capRows >= 8)
      maxStripRows = capRows - (capRows % 8); // round down to 8
    else
      maxStripRows = std::max<int64_t>(capRows, 1); // budget below 8 rows
    int64_t stripRows = std::min(maxStripRows, rows);
    int64_t tgStripSize = stripRows * cols;
    int64_t tgSize = tgStripSize;
    // Pool all 2D convert scatter/gathers into ONE shared TG global per
    // element-type key, sized to the running max. Each convert is fully fenced
    // (scatter, barrier, gather, barrier), so live ranges are disjoint and can
    // alias one buffer - keeping the addrspace(3) footprint at a single tile
    // (what kept the fused fp16 epilogue's 3x 64x64 buffers under the 32KB
    // cap).
    std::string tgName =
        ("__tg_cvt_" + llvm::Twine(getCvtPoolKey(tgElemTy))).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto existing = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      if (!existing) {
        auto arrTy = LLVMArrayType::get(tgElemTy, tgSize);
        LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                               Linkage::Internal, tgName, Attribute(),
                               isPointerElem ? 8 : 4, 3u);
      } else if (auto exAT = dyn_cast<LLVMArrayType>(existing.getGlobalType());
                 exAT && (int64_t)exAT.getNumElements() < tgSize) {
        existing.setGlobalTypeAttr(
            TypeAttr::get(LLVMArrayType::get(tgElemTy, tgSize)));
      }
    }
    auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
    Value tgPtr =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

    // Helper: compute base (row, col) with wrap + in-bounds predicate
    // Returns {bR, bC, pred} where pred is true if this thread owns valid data
    auto makeBase =
        [&](ttg::BlockedEncodingAttr enc) -> std::tuple<Value, Value, Value> {
      auto spt = enc.getSizePerThread();
      auto tpw = enc.getThreadsPerWarp();
      auto wpc = enc.getWarpsPerCTA();
      int64_t sM = spt[rd], sN = spt[cd];
      int64_t tM = tpw[rd], tN = tpw[cd];
      int64_t wM = wpc[rd], wN = wpc[cd];
      int64_t tileM = wM * tM * sM;
      int64_t tileN = wN * tN * sN;

      Value wN_v = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
      Value tN_v = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
      Value tMsM = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
      Value sM_v = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
      Value tNsN = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
      Value sN_v = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

      // order[0] is the fastest-changing dimension; faster dim uses mod,
      // slower uses div (warp and lane decompositions both follow this).
      auto order = enc.getOrder();
      bool colFastest = (order[0] == cd);

      Value wR, wC;
      if (colFastest) {
        wR = arith::DivUIOp::create(rewriter, loc, warpId, wN_v);
        wC = arith::RemUIOp::create(rewriter, loc, warpId, wN_v);
      } else {
        Value wM_v = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
        wR = arith::RemUIOp::create(rewriter, loc, warpId, wM_v);
        wC = arith::DivUIOp::create(rewriter, loc, warpId, wM_v);
      }
      Value lR, lC;
      if (colFastest) {
        lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_v);
        lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_v);
      } else {
        Value tM_v = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
        lR = arith::RemUIOp::create(rewriter, loc, laneId, tM_v);
        lC = arith::DivUIOp::create(rewriter, loc, laneId, tM_v);
      }

      Value bR = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wR, tMsM),
          arith::MulIOp::create(rewriter, loc, lR, sM_v));
      Value bC = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, wC, tNsN),
          arith::MulIOp::create(rewriter, loc, lC, sN_v));

      // In-bounds predicate must be computed before wrapping.
      Value pred;
      auto i1Ty = IntegerType::get(ctx, 1);
      if (tileM > rows || tileN > cols) {
        Value trueVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
        pred = trueVal;
        if (tileM > rows) {
          // Check: bR + max_sM_offset < rows (i.e. bR < rows since sM offsets
          // are 0-based)
          Value rowsV = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
          Value inR = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::ult, bR, rowsV);
          pred = arith::AndIOp::create(rewriter, loc, pred, inR);
          bR = arith::RemUIOp::create(rewriter, loc, bR, rowsV);
        }
        if (tileN > cols) {
          Value colsV = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
          Value inC = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::ult, bC, colsV);
          pred = arith::AndIOp::create(rewriter, loc, pred, inC);
          bC = arith::RemUIOp::create(rewriter, loc, bC, colsV);
        }
      } else {
        pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
      }
      return {bR, bC, pred};
    };

    // Per-lane base (row,col) for an AppleMma source. emitOffsetForLayout fixes
    // lane=0/warp=0, so the lane+warp contribution is added here. Physical
    // per-lane storage (column-major warp tiling, 8x8 tile step per warp):
    //   phys_row = L1 | (L2<<1) | (L4<<2)
    //   phys_col = (L0<<1) | (L3<<2)        (register supplies col bit 0)
    // Enumerated offsets cover exactly the in-bounds MxN positions ->
    // pred=true.
    auto makeBaseMma =
        [&](AppleMmaEncodingAttr enc) -> std::tuple<Value, Value, Value> {
      auto wpc = enc.getWarpsPerCTA();
      int64_t wN = wpc[cd];
      auto bit = [&](Value v, int64_t shift) -> Value {
        Value s = arith::ShRUIOp::create(
            rewriter, loc, v,
            arith::ConstantIntOp::create(rewriter, loc, shift, 32));
        return arith::AndIOp::create(
            rewriter, loc, s,
            arith::ConstantIntOp::create(rewriter, loc, 1, 32));
      };
      auto shl = [&](Value v, int64_t shift) -> Value {
        return arith::ShLIOp::create(
            rewriter, loc, v,
            arith::ConstantIntOp::create(rewriter, loc, shift, 32));
      };
      Value physRow = arith::OrIOp::create(
          rewriter, loc,
          arith::OrIOp::create(rewriter, loc, bit(laneId, 1),
                               shl(bit(laneId, 2), 1)),
          shl(bit(laneId, 4), 2));
      Value physCol = arith::OrIOp::create(
          rewriter, loc, shl(bit(laneId, 0), 1), shl(bit(laneId, 3), 2));
      Value c8 = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
      Value wN_v = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
      Value warpRow = arith::DivUIOp::create(rewriter, loc, warpId, wN_v);
      Value warpCol = arith::RemUIOp::create(rewriter, loc, warpId, wN_v);
      Value bR = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpRow, c8),
          physRow);
      Value bC = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpCol, c8),
          physCol);
      Value truePred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
      return {bR, bC, truePred};
    };

    // LinearLayout-based per-register offsets (blocked or AppleMma source).
    Attribute srcEncAttr = srcMmaEnc ? Attribute(srcMmaEnc) : Attribute(srcEnc);
    auto srcOffsets = emitOffsetForLayout(srcEncAttr, srcTy);
    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);

    SmallVector<std::pair<int64_t, int64_t>> srcCoords, dstCoords;
    for (auto &off : srcOffsets)
      srcCoords.push_back({off[rd], off[cd]});
    for (auto &off : dstOffsets)
      dstCoords.push_back({off[rd], off[cd]});

    Value src = adaptor.getSrc();
    SmallVector<Value> srcElems;
    auto sStructTy = dyn_cast<LLVMStructType>(src.getType());
    bool srcFragment = srcMmaEnc && sStructTy && !sStructTy.getBody().empty() &&
                       isa<VectorType>(sStructTy.getBody()[0]);
    if (srcFragment) {
      // #mma fragment struct -> per-element scalars via the (fragIdx, vecIdx)
      // slot map, so the TG-scatter epilogue sees flat scalars.
      auto info = applegpu::getAppleMmaFragmentInfo(srcTy, srcMmaEnc);
      auto fragElemTy =
          cast<VectorType>(sStructTy.getBody()[0]).getElementType();
      // Fragment carries f32 even for an f16/bf16 accumulator; narrow each
      // extracted SCALAR (scalar fptrunc compiles correctly on AGX, unlike a
      // vector bf16 round-trip on the simdgroup register).
      Type dstElemTy = getTypeConverter()->convertType(dstTy.getElementType());
      bool narrowScalar = fragElemTy != dstElemTy &&
                          isa<FloatType>(fragElemTy) &&
                          isa<FloatType>(dstElemTy) &&
                          dstElemTy.getIntOrFloatBitWidth() <
                              fragElemTy.getIntOrFloatBitWidth();
      SmallVector<Value> frags;
      for (unsigned i = 0; i < sStructTy.getBody().size(); ++i)
        frags.push_back(ExtractValueOp::create(rewriter, loc,
                                               sStructTy.getBody()[i], src,
                                               ArrayRef<int64_t>{(int64_t)i}));
      srcElems.resize(srcCoords.size());
      for (size_t i = 0; i < srcCoords.size(); ++i) {
        int64_t fragIdx, vecIdx;
        applegpu::appleMmaFragmentSlot(srcCoords[i].first, srcCoords[i].second,
                                       info, fragIdx, vecIdx);
        Value frag = (fragIdx < (int64_t)frags.size()) ? frags[fragIdx]
                     : frags.empty()                   ? Value()
                                                       : frags[0];
        Value vIdx = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
        Value sc = LLVM::ExtractElementOp::create(rewriter, loc, fragElemTy,
                                                  frag, vIdx);
        if (narrowScalar)
          sc = arith::TruncFOp::create(rewriter, loc, dstElemTy, sc);
        srcElems[i] = sc;
      }
    } else if (sStructTy) {
      for (unsigned i = 0; i < sStructTy.getBody().size(); ++i)
        srcElems.push_back(
            ExtractValueOp::create(rewriter, loc, sStructTy.getBody()[i], src,
                                   ArrayRef<int64_t>{(int64_t)i}));
    } else {
      srcElems = {src};
    }

    if (srcElems.size() != srcCoords.size())
      return failure();

    auto [srcBaseRow, srcBaseCol, srcPred] =
        srcMmaEnc ? makeBaseMma(srcMmaEnc) : makeBase(srcEnc);
    auto [dstBaseRow, dstBaseCol, dstPred] = makeBase(dstEnc);

    // Flat index into the strip; row offset is relative to strip start.
    auto stripFlatIdx = [&](Value bR, Value bC, int64_t rOff, int64_t cOff,
                            int64_t stripRowStart) -> Value {
      Value r =
          arith::AddIOp::create(rewriter, loc, bR,
                                arith::ConstantIntOp::create(
                                    rewriter, loc, rOff - stripRowStart, 32));
      Value c = arith::AddIOp::create(
          rewriter, loc, bC,
          arith::ConstantIntOp::create(rewriter, loc, cOff, 32));
      Value f = arith::AddIOp::create(
          rewriter, loc,
          arith::MulIOp::create(
              rewriter, loc, r,
              arith::ConstantIntOp::create(rewriter, loc, cols, 32)),
          c);
      return arith::ExtUIOp::create(rewriter, loc, i64Ty, f);
    };

    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

    SmallVector<Value> dstElems(dstCoords.size());
    Value zeroElem;
    if (isPointerElem) {
      // The masked-out fallback for a pointer lane must NOT be a null device
      // pointer: the AIR materializer refuses to build a PSO that can store
      // through a constant-null device ptr ("Failed to materializeAll") even
      // when predicated off. Reuse a real input pointer (srcElems[0]) as the
      // neutral element - valid, never dereferenced.
      if (!srcElems.empty()) {
        zeroElem = srcElems[0];
      } else {
        Value zeroInt = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
        zeroElem = LLVM::IntToPtrOp::create(rewriter, loc, elemTy, zeroInt);
      }
    } else {
      zeroElem = arith::ConstantOp::create(rewriter, loc, elemTy,
                                           rewriter.getZeroAttr(elemTy));
    }
    for (size_t i = 0; i < dstElems.size(); ++i)
      dstElems[i] = zeroElem;

    // Tiled scatter/gather: process stripRows rows at a time.
    // When rows*cols fits in 32KB, stripRows==rows → single pass (no overhead).
    // Otherwise tiles down to fit, adding 2 barriers per extra strip.
    int64_t numStrips = (rows + stripRows - 1) / stripRows;
    for (int64_t strip = 0; strip < numStrips; ++strip) {
      int64_t rowStart = strip * stripRows;
      int64_t rowEnd = std::min(rowStart + stripRows, rows);

      // Scatter this strip via predicated stores. pred = srcPred &&
      // inStrip(rOff) is shared per row, so group by rOff and emit one
      // conditional block per distinct row (block count proportional to strip
      // rows, not the full tile).
      std::map<int64_t, SmallVector<size_t>> srcByRow;
      SmallVector<int64_t> srcRowOrder;
      for (size_t i = 0; i < srcElems.size(); ++i) {
        int64_t rOff = srcCoords[i].first;
        if (srcByRow.find(rOff) == srcByRow.end())
          srcRowOrder.push_back(rOff);
        srcByRow[rOff].push_back(i);
      }
      bool singleStripSrc = (numStrips == 1);
      for (int64_t rOff : srcRowOrder) {
        // Single strip => every row in-strip; pred collapses to srcPred.
        Value pred;
        if (singleStripSrc) {
          pred = srcPred;
        } else {
          Value actualRow = arith::AddIOp::create(
              rewriter, loc, srcBaseRow,
              arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
          Value inStrip = arith::AndIOp::create(
              rewriter, loc,
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::ult, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowEnd, 32)));
          pred = arith::AndIOp::create(rewriter, loc, srcPred, inStrip);
        }
        auto *curBlock = rewriter.getInsertionBlock();
        auto curPoint = rewriter.getInsertionPoint();
        auto *endBlock = curBlock->splitBlock(curPoint);
        auto *thenBlock = rewriter.createBlock(endBlock);
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, pred, thenBlock, endBlock);
        rewriter.setInsertionPointToEnd(thenBlock);
        for (size_t i : srcByRow[rOff]) {
          int64_t cOff = srcCoords[i].second;
          Value idx =
              stripFlatIdx(srcBaseRow, srcBaseCol, rOff, cOff, rowStart);
          Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy,
                                          tgPtr, ArrayRef<LLVM::GEPArg>{idx});
          Value toStore = srcElems[i];
          if (isPointerElem)
            toStore = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, toStore);
          LLVM::StoreOp::create(rewriter, loc, toStore, gep);
        }
        LLVM::BrOp::create(rewriter, loc, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
      }

      // Barrier: all threads done scattering this strip.
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      // Gather this strip. Use wrapped dstBaseRow for the strip check; do NOT
      // gate by dstPred - when tileM > rows, multiple threads wrap to the same
      // row and all need the correct TG value. inStrip is shared per row.
      std::map<int64_t, SmallVector<size_t>> dstByRow;
      SmallVector<int64_t> dstRowOrder;
      for (size_t i = 0; i < dstCoords.size(); ++i) {
        int64_t rOff = dstCoords[i].first;
        if (dstByRow.find(rOff) == dstByRow.end())
          dstRowOrder.push_back(rOff);
        dstByRow[rOff].push_back(i);
      }
      // Single-strip fast path: whole tile fits one TG pass, so every dst row
      // is unconditionally in-strip and both selects collapse to the gathered
      // value
      // - emit plain GEP+load. This elides the select/icmp chain that is ~99%
      // of a 128x128x64 dot's #mma->#blocked output convert IR.
      bool singleStrip = (numStrips == 1);
      Value zeroIdx = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      for (int64_t rOff : dstRowOrder) {
        Value inStrip;
        if (!singleStrip) {
          Value actualRow = arith::AddIOp::create(
              rewriter, loc, dstBaseRow,
              arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
          inStrip = arith::AndIOp::create(
              rewriter, loc,
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
              arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::ult, actualRow,
                  arith::ConstantIntOp::create(rewriter, loc, rowEnd, 32)));
        }
        for (size_t i : dstByRow[rOff]) {
          int64_t cOff = dstCoords[i].second;
          Value idx =
              stripFlatIdx(dstBaseRow, dstBaseCol, rOff, cOff, rowStart);
          Value safeIdx = singleStrip
                              ? idx
                              : arith::SelectOp::create(rewriter, loc, inStrip,
                                                        idx, zeroIdx)
                                    .getResult();
          Value gep =
              LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                  ArrayRef<LLVM::GEPArg>{safeIdx});
          Value gathered =
              LLVM::LoadOp::create(rewriter, loc, tgElemTy, gep).getResult();
          if (isPointerElem)
            gathered =
                LLVM::IntToPtrOp::create(rewriter, loc, elemTy, gathered);
          dstElems[i] = singleStrip
                            ? gathered
                            : arith::SelectOp::create(rewriter, loc, inStrip,
                                                      gathered, dstElems[i])
                                  .getResult();
        }
      }

      // Barrier: all threads done gathering before next strip's scatter.
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});
    }

    auto outTy = getTypeConverter()->convertType(dstTy);
    if (!outTy)
      return failure();

    if (auto outSt = dyn_cast<LLVMStructType>(outTy)) {
      if (outSt.getBody().size() != dstElems.size())
        return failure();
      Value result = UndefOp::create(rewriter, loc, outSt);
      for (size_t i = 0; i < dstElems.size(); ++i)
        result =
            InsertValueOp::create(rewriter, loc, outSt, result, dstElems[i],
                                  ArrayRef<int64_t>{(int64_t)i});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, dstElems[0]);
    }
    return success();
  }

  // #mma -> W-blocked register-shuffle restripe (store epilogue). Each W dst
  // (row,col) is gathered from the #mma lane owning tile-local (row%8, col%8)
  // via air.simd_shuffle on the col-parity register. No shared memory, no
  // barriers; bit-exact in tools/fragment-oracle.
  LogicalResult convertMmaToWShuffle(ttg::ConvertLayoutOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter,
                                     AppleMmaEncodingAttr srcMmaEnc,
                                     ttg::BlockedEncodingAttr dstEnc,
                                     RankedTensorType srcTy,
                                     RankedTensorType dstTy, Location loc,
                                     MLIRContext *ctx, ModuleOp mod) const {
    unsigned rank = srcTy.getShape().size();
    if (rank < 2)
      return failure();
    unsigned rd = rank - 2, cd = rank - 1;
    for (unsigned d = 0; d < rd; ++d)
      if (srcTy.getShape()[d] != 1)
        return failure();

    // The fragment source must be the vectorized <64 x f32> struct.
    Value src = adaptor.getSrc();
    auto sStructTy = dyn_cast<LLVMStructType>(src.getType());
    if (!sStructTy || sStructTy.getBody().empty() ||
        !isa<VectorType>(sStructTy.getBody()[0]))
      return failure();

    auto outTy = getTypeConverter()->convertType(dstTy);
    auto outSt = dyn_cast_or_null<LLVMStructType>(outTy);
    if (!outSt)
      return failure();
    Type fragElemTy = cast<VectorType>(sStructTy.getBody()[0]).getElementType();
    if (!fragElemTy.isF32())
      return failure();

    auto i16Ty = IntegerType::get(ctx, 16);

    // Per-lane W base (row,col); + dstEnc per-register offsets = absolute pos.
    Value laneId = emitLaneId(rewriter, loc, mod);
    auto tpw = dstEnc.getThreadsPerWarp();
    auto spt = dstEnc.getSizePerThread();
    auto order = dstEnc.getOrder();
    int64_t tM = tpw[rd], tN = tpw[cd];
    int64_t sM = spt[rd], sN = spt[cd];
    bool colFastest = (order[0] == cd);
    Value tN_v = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
    Value tM_v = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
    Value lR, lC;
    if (colFastest) {
      lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_v);
      lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_v);
    } else {
      lR = arith::RemUIOp::create(rewriter, loc, laneId, tM_v);
      lC = arith::DivUIOp::create(rewriter, loc, laneId, tM_v);
    }
    Value sM_v = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
    Value sN_v = arith::ConstantIntOp::create(rewriter, loc, sN, 32);
    Value laneRow = arith::MulIOp::create(rewriter, loc, lR, sM_v);
    Value laneCol = arith::MulIOp::create(rewriter, loc, lC, sN_v);

    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);
    if (dstOffsets.size() != outSt.getBody().size())
      return failure();

    auto srcInfo = applegpu::getAppleMmaFragmentInfo(srcTy, srcMmaEnc);
    SmallVector<Value> frags;
    for (unsigned i = 0; i < sStructTy.getBody().size(); ++i)
      frags.push_back(ExtractValueOp::create(rewriter, loc,
                                             sStructTy.getBody()[i], src,
                                             ArrayRef<int64_t>{(int64_t)i}));

    auto bit = [&](Value v, int64_t shift) -> Value {
      Value s = arith::ShRUIOp::create(
          rewriter, loc, v,
          arith::ConstantIntOp::create(rewriter, loc, shift, 32));
      return arith::AndIOp::create(
          rewriter, loc, s, arith::ConstantIntOp::create(rewriter, loc, 1, 32));
    };
    auto shl = [&](Value v, int64_t shift) -> Value {
      return arith::ShLIOp::create(
          rewriter, loc, v,
          arith::ConstantIntOp::create(rewriter, loc, shift, 32));
    };

    Value result = UndefOp::create(rewriter, loc, outSt);
    for (size_t i = 0; i < dstOffsets.size(); ++i) {
      int64_t rOff = dstOffsets[i][rd], cOff = dstOffsets[i][cd];
      Value absRow = arith::AddIOp::create(
          rewriter, loc, laneRow,
          arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
      Value absCol = arith::AddIOp::create(
          rewriter, loc, laneCol,
          arith::ConstantIntOp::create(rewriter, loc, cOff, 32));
      // tile-local (row%8, col%8) drive the #mma physical source lane:
      //   srcLane = L0:(col>>1) L1:row L2:(row>>1) L3:(col>>2) L4:(row>>2)
      Value tRow = arith::AndIOp::create(
          rewriter, loc, absRow,
          arith::ConstantIntOp::create(rewriter, loc, 7, 32));
      Value tCol = arith::AndIOp::create(
          rewriter, loc, absCol,
          arith::ConstantIntOp::create(rewriter, loc, 7, 32));
      Value srcLane = arith::OrIOp::create(
          rewriter, loc,
          arith::OrIOp::create(rewriter, loc,
                               arith::OrIOp::create(rewriter, loc, bit(tCol, 1),
                                                    shl(bit(tRow, 0), 1)),
                               shl(bit(tRow, 1), 2)),
          arith::OrIOp::create(rewriter, loc, shl(bit(tCol, 2), 3),
                               shl(bit(tRow, 2), 4)));
      Value srcLane16 = arith::TruncIOp::create(rewriter, loc, i16Ty, srcLane);

      // Fragment register holding (absRow,absCol): one frag per 8x8 tile,
      // vecIdx = col parity.
      int64_t fragIdx, vecIdx;
      applegpu::appleMmaFragmentSlot(rOff, cOff, srcInfo, fragIdx, vecIdx);
      if (fragIdx >= (int64_t)frags.size())
        return failure();
      Value vIdx = arith::ConstantIntOp::create(rewriter, loc, vecIdx, 32);
      Value scalar =
          LLVM::ExtractElementOp::create(rewriter, loc, frags[fragIdx], vIdx);
      Value shuffled = emitFragShuffle(rewriter, loc, mod, scalar, srcLane16);
      // f16/bf16 accumulator: fragment is f32; narrow the shuffled scalar
      // (scalar fptrunc, off the simdgroup register).
      Type outElemTy = outSt.getBody()[i];
      if (shuffled.getType() != outElemTy &&
          isa<FloatType>(shuffled.getType()) && isa<FloatType>(outElemTy) &&
          outElemTy.getIntOrFloatBitWidth() <
              shuffled.getType().getIntOrFloatBitWidth())
        shuffled = arith::TruncFOp::create(rewriter, loc, outElemTy, shuffled);
      if (shuffled.getType() != outElemTy)
        return failure();
      result = InsertValueOp::create(rewriter, loc, outSt, result, shuffled,
                                     ArrayRef<int64_t>{(int64_t)i});
    }
    rewriter.replaceOp(op, result);
    return success();
  }

  // 1D blocked→blocked conversion via TG scatter/gather
  LogicalResult convertLayout1D(ttg::ConvertLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                ttg::BlockedEncodingAttr srcEnc,
                                ttg::BlockedEncodingAttr dstEnc,
                                RankedTensorType srcTy, RankedTensorType dstTy,
                                Location loc, MLIRContext *ctx,
                                ModuleOp mod) const {

    int64_t numElems = srcTy.getShape()[0];
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto tgPtrTy = LLVMPointerType::get(ctx, 3);

    // For pointer elements, use i64 in TG (Metal can't store ptrs in TG)
    bool isPointerElem = isa<LLVMPointerType>(elemTy);
    Type tgElemTy = isPointerElem ? i64Ty : elemTy;

    auto laneIdFnTy = LLVMFunctionType::get(i32Ty, {}, false);
    LLVMFuncOp laneIdFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn =
              mod.lookupSymbol<LLVMFuncOp>("air.thread_index_in_simdgroup"))
        laneIdFn = fn;
      else
        laneIdFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.thread_index_in_simdgroup",
                                      laneIdFnTy, Linkage::External);
    }

    auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    LLVMFuncOp tidFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>(
              "air.thread_position_in_threadgroup"))
        tidFn = fn;
      else
        tidFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                   "air.thread_position_in_threadgroup",
                                   tidFnTy, Linkage::External);
    }

    auto barrFnTy =
        LLVMFunctionType::get(LLVMVoidType::get(ctx), {i32Ty, i32Ty}, false);
    LLVMFuncOp tgBarrFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.threadgroup.barrier"))
        tgBarrFn = fn;
      else
        tgBarrFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.threadgroup.barrier", barrFnTy,
                                      Linkage::External);
    }

    Value laneId =
        LLVM::CallOp::create(rewriter, loc, laneIdFn, ValueRange{}).getResult();
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                               ArrayRef<int64_t>{0});
    Value c32 = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
    Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

    // TG global pooled per element type (fully fenced like the 2D path, so
    // same-type conversions share one barrier-disjoint buffer).
    std::string tgName =
        ("__tg_cvt_" + llvm::Twine(getCvtPoolKey(tgElemTy))).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto existing = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      if (!existing) {
        auto arrTy = LLVMArrayType::get(tgElemTy, numElems);
        LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                               Linkage::Internal, tgName, Attribute(),
                               isPointerElem ? 8 : 4, 3u);
      } else if (auto exAT = dyn_cast<LLVMArrayType>(existing.getGlobalType());
                 exAT && (int64_t)exAT.getNumElements() < numElems) {
        existing.setGlobalTypeAttr(
            TypeAttr::get(LLVMArrayType::get(tgElemTy, numElems)));
      }
    }
    auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
    Value tgPtr =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

    // base = warpId * (tpw * spt) + laneId * spt
    auto computeBase1D = [&](ttg::BlockedEncodingAttr enc) -> Value {
      auto spt = enc.getSizePerThread()[0];
      auto tpw = enc.getThreadsPerWarp()[0];
      auto wpc = enc.getWarpsPerCTA()[0];
      Value tpwSpt = arith::ConstantIntOp::create(rewriter, loc, tpw * spt, 32);
      Value sptV = arith::ConstantIntOp::create(rewriter, loc, spt, 32);
      Value tpwV = arith::ConstantIntOp::create(rewriter, loc, tpw, 32);
      Value wpcV = arith::ConstantIntOp::create(rewriter, loc, wpc, 32);
      // Clamp physical warp/lane into the layout's logical extent: when
      // warpsPerCTA[0] < the threadgroup warp count, raw warpId would index
      // past the buffer slot.
      Value warpIdx = arith::RemUIOp::create(rewriter, loc, warpId, wpcV);
      Value laneIdx = arith::RemUIOp::create(rewriter, loc, laneId, tpwV);
      Value base = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpIdx, tpwSpt),
          arith::MulIOp::create(rewriter, loc, laneIdx, sptV));
      return base;
    };

    auto srcOffsets = emitOffsetForLayout(srcEnc, srcTy);
    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);

    Value src = adaptor.getSrc();
    SmallVector<Value> srcElems;
    if (auto sTy = dyn_cast<LLVMStructType>(src.getType())) {
      for (unsigned i = 0; i < sTy.getBody().size(); ++i)
        srcElems.push_back(
            ExtractValueOp::create(rewriter, loc, sTy.getBody()[i], src,
                                   ArrayRef<int64_t>{(int64_t)i}));
    } else {
      srcElems = {src};
    }

    Value srcBase = computeBase1D(srcEnc);
    Value dstBase = computeBase1D(dstEnc);

    for (size_t i = 0; i < srcElems.size(); ++i) {
      int64_t elemOff = srcOffsets[i][0];
      Value idx = arith::AddIOp::create(
          rewriter, loc, srcBase,
          arith::ConstantIntOp::create(rewriter, loc, elemOff, 32));
      Value idx64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                      ArrayRef<LLVM::GEPArg>{idx64});
      Value toStore = srcElems[i];
      if (isPointerElem)
        toStore = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, toStore);
      LLVM::StoreOp::create(rewriter, loc, toStore, gep);
    }

    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

    SmallVector<Value> dstElems;
    for (size_t i = 0; i < dstOffsets.size(); ++i) {
      int64_t elemOff = dstOffsets[i][0];
      Value idx = arith::AddIOp::create(
          rewriter, loc, dstBase,
          arith::ConstantIntOp::create(rewriter, loc, elemOff, 32));
      Value idx64 = arith::ExtUIOp::create(rewriter, loc, i64Ty, idx);
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                      ArrayRef<LLVM::GEPArg>{idx64});
      Value loaded =
          LLVM::LoadOp::create(rewriter, loc, tgElemTy, gep).getResult();
      if (isPointerElem)
        loaded = LLVM::IntToPtrOp::create(rewriter, loc, elemTy, loaded);
      dstElems.push_back(loaded);
    }

    auto outTy = getTypeConverter()->convertType(dstTy);
    if (!outTy)
      return failure();

    if (auto outSt = dyn_cast<LLVMStructType>(outTy)) {
      if (outSt.getBody().size() != dstElems.size())
        return failure();
      Value result = UndefOp::create(rewriter, loc, outSt);
      for (size_t i = 0; i < dstElems.size(); ++i)
        result =
            InsertValueOp::create(rewriter, loc, outSt, result, dstElems[i],
                                  ArrayRef<int64_t>{(int64_t)i});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, dstElems[0]);
    }
    return success();
  }

  // 1D convert for arbitrary distributed encodings (e.g. #ttg.slice). The smem
  // slot is each register's actual tensor index (emitIndices), so scatter and
  // gather address the same slot by construction - unlike computeBase1D's
  // warp/lane formula, which only holds for contiguous blocked layouts.
  LogicalResult convertLayout1DGeneric(ttg::ConvertLayoutOp op,
                                       OpAdaptor adaptor,
                                       ConversionPatternRewriter &rewriter,
                                       Attribute srcEnc, Attribute dstEnc,
                                       RankedTensorType srcTy,
                                       RankedTensorType dstTy, Location loc,
                                       MLIRContext *ctx, ModuleOp mod) const {
    int64_t numElems = srcTy.getShape()[0];
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto tgPtrTy = LLVMPointerType::get(ctx, 3);
    if (isa<LLVMPointerType>(elemTy))
      return failure();
    Type tgElemTy = elemTy;

    TargetInfo targetInfo;
    auto srcIndices = emitIndices(loc, rewriter, targetInfo, srcEnc, srcTy,
                                  /*withCTA=*/false);
    auto dstIndices = emitIndices(loc, rewriter, targetInfo, dstEnc, dstTy,
                                  /*withCTA=*/false);

    std::string tgName =
        ("__tg_cvt_" + llvm::Twine(getCvtPoolKey(tgElemTy))).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto existing = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      if (!existing) {
        auto arrTy = LLVMArrayType::get(tgElemTy, numElems);
        LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                               Linkage::Internal, tgName, Attribute(), 4, 3u);
      } else if (auto exAT = dyn_cast<LLVMArrayType>(existing.getGlobalType());
                 exAT && (int64_t)exAT.getNumElements() < numElems) {
        existing.setGlobalTypeAttr(
            TypeAttr::get(LLVMArrayType::get(tgElemTy, numElems)));
      }
    }
    auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
    Value tgPtr =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

    auto barrFnTy =
        LLVMFunctionType::get(LLVMVoidType::get(ctx), {i32Ty, i32Ty}, false);
    LLVMFuncOp tgBarrFn;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.threadgroup.barrier"))
        tgBarrFn = fn;
      else
        tgBarrFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                                      "air.threadgroup.barrier", barrFnTy,
                                      Linkage::External);
    }

    Value src = adaptor.getSrc();
    SmallVector<Value> srcElems;
    if (auto sTy = dyn_cast<LLVMStructType>(src.getType())) {
      for (unsigned i = 0; i < sTy.getBody().size(); ++i)
        srcElems.push_back(
            ExtractValueOp::create(rewriter, loc, sTy.getBody()[i], src,
                                   ArrayRef<int64_t>{(int64_t)i}));
    } else {
      srcElems = {src};
    }
    if (srcElems.size() != srcIndices.size())
      return failure();

    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});
    for (size_t i = 0; i < srcElems.size(); ++i) {
      Value idx64 =
          arith::ExtUIOp::create(rewriter, loc, i64Ty, srcIndices[i].front());
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                      ArrayRef<LLVM::GEPArg>{idx64});
      LLVM::StoreOp::create(rewriter, loc, srcElems[i], gep);
    }
    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

    SmallVector<Value> dstElems;
    for (size_t i = 0; i < dstIndices.size(); ++i) {
      Value idx64 =
          arith::ExtUIOp::create(rewriter, loc, i64Ty, dstIndices[i].front());
      Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                      ArrayRef<LLVM::GEPArg>{idx64});
      dstElems.push_back(
          LLVM::LoadOp::create(rewriter, loc, tgElemTy, gep).getResult());
    }

    auto outTy = getTypeConverter()->convertType(dstTy);
    if (!outTy)
      return failure();
    if (auto outSt = dyn_cast<LLVMStructType>(outTy)) {
      if (outSt.getBody().size() != dstElems.size())
        return failure();
      Value result = UndefOp::create(rewriter, loc, outSt);
      for (size_t i = 0; i < dstElems.size(); ++i)
        result =
            InsertValueOp::create(rewriter, loc, outSt, result, dstElems[i],
                                  ArrayRef<int64_t>{(int64_t)i});
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, dstElems[0]);
    }
    return success();
  }
};

} // anonymous namespace

void populateConvertLayoutOpPattern(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  patterns.add<ConvertLayoutOpAppleConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
}

} // namespace mlir::triton::applegpu
