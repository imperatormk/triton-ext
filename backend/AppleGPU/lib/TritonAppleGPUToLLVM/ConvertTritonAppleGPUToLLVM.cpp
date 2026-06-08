// ConvertTritonAppleGPUToLLVM pass
//
// Lowers TritonGPU IR → LLVM IR for Apple MPS using shared Triton patterns
// and an Apple-specific TargetInfo.

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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::applegpu {

namespace {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

// ConvertLayoutOp for DotOperandEncoding or blocked→blocked:
//
// - DotOperandEncoding target: identity pass-through (elements same per thread)
// - blocked→blocked: TG scatter/gather redistribution
struct ConvertLayoutOpAppleConversion
    : public mlir::ConvertOpToLLVMPattern<ttg::ConvertLayoutOp> {
  using mlir::ConvertOpToLLVMPattern<
      ttg::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  // Per-context counter for unique TG global names
  static unsigned &getCounter(MLIRContext *ctx) {
    static llvm::DenseMap<MLIRContext *, unsigned> counters;
    return counters[ctx];
  }

  LogicalResult
  matchAndRewrite(ttg::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getResult().getType());

    // DotOperandEncoding target — identity pass-through when source is
    // blocked encoding.  Our dot lowering looks through convert_layout to
    // get source blocked values and uses blocked encoding for scatter/gather.
    // When the source is MMA encoding, the element count per thread differs
    // from the DotOperandEncoding count, so we must NOT pass through
    // (the shared LinearLayout-based convert_layout handles that case).
    if (isa<ttg::DotOperandEncodingAttr>(dstTy.getEncoding()) &&
        isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding())) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    // blocked→blocked redistribution via TG scatter/gather
    auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    auto dstEnc = dyn_cast<ttg::BlockedEncodingAttr>(dstTy.getEncoding());
    if (!srcEnc || !dstEnc)
      return failure();

    // Same encoding — identity
    if (srcEnc == dstEnc) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();
    auto shape = srcTy.getShape();

    // 1D blocked→blocked: simple TG scatter/gather with flat indices
    if (shape.size() == 1) {
      return convertLayout1D(op, adaptor, rewriter, srcEnc, dstEnc, srcTy,
                             dstTy, loc, ctx, mod);
    }

    // ND blocked->blocked: handle rank>=2 by operating on the trailing two
    // dimensions (rows, cols). For rank>2 we require every leading dim to be
    // size 1 so the whole tensor is a single (rows x cols) tile; this is the
    // case the upstream transferWithinBlockSwizzling miscompiles for fp16/bf16
    // replicated layouts (it vectorizes the replicated registers into a
    // <2 x half> store at a 2-byte stride but reads them back at a 4-byte
    // stride, corrupting the row+16 slot). Our scatter/gather is fully scalar
    // and addresses TG by (row,col) coordinate, so it is correct regardless of
    // replication. Routing the convert here keeps the fix entirely in-tree.
    unsigned rank = shape.size();
    if (rank < 2)
      return failure();
    unsigned rd = rank - 2; // row dim index
    unsigned cd = rank - 1; // col dim index
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

    // Get lane/warp IDs (same helpers as DotOp)
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

    // Create TG global for tiled scatter/gather.
    // Use the largest strip height (multiple of 8) that fits in 32KB TG.
    // Fewer strips = fewer barriers = better performance.
    // Floor to rows when full tensor already fits.
    // Account for global_smem (from allocate-shared-memory pass) and
    // MMA dot TG buffers (from tt.dot pre-scan) in the 32KB TG budget.
    constexpr int64_t tgBudgetBytes = 32 * 1024;
    int64_t smemBytes = 0;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
      smemBytes = attr.getValue().getZExtValue();
    int64_t mmaBytes = 0;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.mma_shared"))
      mmaBytes = attr.getValue().getZExtValue();
    int64_t availBytes = tgBudgetBytes - smemBytes - mmaBytes;
    int64_t elemBytes = isPointerElem ? 8 : elemTy.getIntOrFloatBitWidth() / 8;
    // Reserve 1 slot for garbage bin, then fit as many rows as possible
    int64_t maxStripRows = (availBytes / elemBytes - 1) / cols;
    maxStripRows = std::max<int64_t>(maxStripRows - (maxStripRows % 8),
                                     8); // round down to 8, min 8
    int64_t stripRows = std::min(maxStripRows, rows);
    int64_t tgStripSize = stripRows * cols;
    int64_t tgSize = tgStripSize + 1; // +1 garbage bin slot
    unsigned id = getCounter(ctx)++;
    std::string tgName = ("__tg_cvt_" + llvm::Twine(id)).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto arrTy = LLVMArrayType::get(tgElemTy, tgSize);
      LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                             Linkage::Internal, tgName, Attribute(),
                             isPointerElem ? 8 : 4, 3u);
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

      // Respect layout order: order[0] is the fastest-changing dimension
      auto order = enc.getOrder();
      bool colFastest =
          (order[0] == cd); // order=[..,cd,rd] => col fastest (default)

      // Warp decomposition: faster dim uses mod, slower uses div
      Value wR, wC;
      if (colFastest) {
        wR = arith::DivUIOp::create(rewriter, loc, warpId, wN_v);
        wC = arith::RemUIOp::create(rewriter, loc, warpId, wN_v);
      } else {
        Value wM_v = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
        wR = arith::RemUIOp::create(rewriter, loc, warpId, wM_v);
        wC = arith::DivUIOp::create(rewriter, loc, warpId, wM_v);
      }
      // Lane decomposition: faster dim uses mod, slower uses div
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

      // Compute in-bounds predicate before wrapping
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

    // Use LinearLayout-based offsets (matches upstream element ordering)
    auto srcOffsets = emitOffsetForLayout(srcEnc, srcTy);
    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);

    // Convert to (row, col) pairs
    SmallVector<std::pair<int64_t, int64_t>> srcCoords, dstCoords;
    for (auto &off : srcOffsets)
      srcCoords.push_back({off[rd], off[cd]});
    for (auto &off : dstOffsets)
      dstCoords.push_back({off[rd], off[cd]});

    // Unpack source elements
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

    if (srcElems.size() != srcCoords.size())
      return failure();

    auto [srcBaseRow, srcBaseCol, srcPred] = makeBase(srcEnc);
    auto [dstBaseRow, dstBaseCol, dstPred] = makeBase(dstEnc);

    // Strip flat index: row offset relative to strip start
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
    Value garbageIdx =
        arith::ConstantIntOp::create(rewriter, loc, tgStripSize, 64);

    // Initialize destination elements with undef (will be filled strip by
    // strip)
    SmallVector<Value> dstElems(dstCoords.size());
    Value zeroElem;
    if (isPointerElem) {
      // The masked-out fallback for a pointer-typed lane must NOT be a null
      // (address 0) device pointer. Metal's AIR materializer refuses to build a
      // PSO for a kernel that can store through a constant-null device pointer
      // ("Failed to materializeAll"), even when that store is predicated off.
      // Reuse a real input pointer (srcElems[0]) as the neutral element: it is
      // a valid pointer of the right type and address space, and is only ever
      // selected into masked-out lanes whose stores are disabled, so it is
      // never actually dereferenced. This keeps the materializer happy without
      // changing observable behaviour.
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

      // Scatter source elements for this strip (garbage-bin for out-of-strip)
      for (size_t i = 0; i < srcElems.size(); ++i) {
        auto [rOff, cOff] = srcCoords[i];
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
        // Combine with in-bounds predicate
        Value pred = arith::AndIOp::create(rewriter, loc, srcPred, inStrip);
        Value idx = stripFlatIdx(srcBaseRow, srcBaseCol, rOff, cOff, rowStart);
        Value safeIdx =
            arith::SelectOp::create(rewriter, loc, pred, idx, garbageIdx);
        Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                        ArrayRef<LLVM::GEPArg>{safeIdx});
        Value toStore = srcElems[i];
        if (isPointerElem)
          toStore = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, toStore);
        LLVM::StoreOp::create(rewriter, loc, toStore, gep);
      }

      // Barrier: all threads done scattering this strip
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});

      // Gather destination elements for this strip.
      // Use wrapped dstBaseRow (already < rows) for strip check — do NOT
      // gate by dstPred. When tileM > rows, multiple threads wrap to the
      // same row; all need the correct TG value regardless of dstPred.
      for (size_t i = 0; i < dstCoords.size(); ++i) {
        auto [rOff, cOff] = dstCoords[i];
        Value actualRow = arith::AddIOp::create(
            rewriter, loc, dstBaseRow,
            arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
        Value inStrip = arith::AndIOp::create(
            rewriter, loc,
            arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::uge, actualRow,
                arith::ConstantIntOp::create(rewriter, loc, rowStart, 32)),
            arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::ult, actualRow,
                arith::ConstantIntOp::create(rewriter, loc, rowEnd, 32)));
        Value idx = stripFlatIdx(dstBaseRow, dstBaseCol, rOff, cOff, rowStart);
        Value safeIdx =
            arith::SelectOp::create(rewriter, loc, inStrip, idx, garbageIdx);
        Value gep = LLVM::GEPOp::create(rewriter, loc, tgPtrTy, tgElemTy, tgPtr,
                                        ArrayRef<LLVM::GEPArg>{safeIdx});
        Value gathered =
            LLVM::LoadOp::create(rewriter, loc, tgElemTy, gep).getResult();
        if (isPointerElem)
          gathered = LLVM::IntToPtrOp::create(rewriter, loc, elemTy, gathered);
        // Use gathered value if in strip, keep previous otherwise
        dstElems[i] = arith::SelectOp::create(rewriter, loc, inStrip, gathered,
                                              dstElems[i]);
      }

      // Barrier: all threads done gathering before next strip's scatter
      LLVM::CallOp::create(rewriter, loc, tgBarrFn,
                           ValueRange{fenceTG, execMod});
    }

    // Pack result
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

    // Get thread/warp IDs
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

    // Create TG global
    unsigned id = getCounter(ctx)++;
    std::string tgName = ("__tg_cvt_" + llvm::Twine(id)).str();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto arrTy = LLVMArrayType::get(tgElemTy, numElems);
      LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                             Linkage::Internal, tgName, Attribute(),
                             isPointerElem ? 8 : 4, 3u);
    }
    auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
    Value tgPtr =
        LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

    // Compute per-thread base index for 1D blocked layout
    auto computeBase1D = [&](ttg::BlockedEncodingAttr enc) -> Value {
      auto spt = enc.getSizePerThread()[0];
      auto tpw = enc.getThreadsPerWarp()[0];
      auto wpc = enc.getWarpsPerCTA()[0];
      // base = warpId * (tpw * spt) + laneId * spt
      Value tpwSpt = arith::ConstantIntOp::create(rewriter, loc, tpw * spt, 32);
      Value sptV = arith::ConstantIntOp::create(rewriter, loc, spt, 32);
      Value base = arith::AddIOp::create(
          rewriter, loc, arith::MulIOp::create(rewriter, loc, warpId, tpwSpt),
          arith::MulIOp::create(rewriter, loc, laneId, sptV));
      return base;
    };

    // Use emitOffsetForLayout to get canonical element ordering
    auto srcOffsets = emitOffsetForLayout(srcEnc, srcTy);
    auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);

    // Unpack source elements
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

    // Scatter source elements to TG
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

    // Barrier
    Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
    LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

    // Gather destination elements from TG
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

    // Pack result
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

// Lower triton::AtomicRMWOp → air.atomic.global.{op}.{type}
//
// Metal uses explicit AIR intrinsics for atomics:
//   float @air.atomic.global.add.f32(float addrspace(1)*, float, i32 order, i32
//   scope, i1 volatile) i32   @air.atomic.global.add.s.i32(i32 addrspace(1)*,
//   i32, i32 order, i32 scope, i1 volatile) i32
//   @air.atomic.global.max.s.i32(...) i32   @air.atomic.global.min.s.i32(...)
//   i32   @air.atomic.global.xchg.s.i32(...)
//
// For unsupported native atomics (f32 max/min, f16/bf16 add), we emit a CAS
// loop:
//   air.atomic.global.cmpxchg.weak.i32(ptr, expected_ptr, desired, succ_order,
//   fail_order, scope, vol) returns old i32. Expected is passed by pointer and
//   updated on failure.

// Emit a predicate that is true only for threads that own unique data.
// This prevents redundant threads from executing atomics (e.g. 128 threads
// for a 4-element tensor → only 4 should execute).
// Returns null Value if no predication is needed.
static Value emitAppleRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod) {
  auto *ctx = rewriter.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");

  // Get thread_position_in_threadgroup (tid within threadgroup)
  auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
  auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
      LLVMFuncOp::create(rewriter, mod.getLoc(),
                         "air.thread_position_in_threadgroup", tidFnTy,
                         Linkage::External);
  }
  auto tidFn =
      mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");
  Value tidStruct =
      LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
  Value tid = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                           ArrayRef<int64_t>{0});

  // laneId = tid % 32, warpId = tid / 32
  int tpw = ttg::lookupThreadsPerWarp(rewriter);
  Value warpSize = arith::ConstantIntOp::create(rewriter, loc, tpw, 32);
  Value laneId = arith::RemUIOp::create(rewriter, loc, tid, warpSize);
  Value warpId = arith::DivUIOp::create(rewriter, loc, tid, warpSize);

  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  Value pred;

  // Check each dimension: (dimId & mask) == 0 means this thread is canonical
  auto dimNames = {kLane, kWarp};
  auto dimIds = {laneId, warpId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    auto it = freeVarMasks.find(dimName);
    int32_t mask = (it != freeVarMasks.end()) ? it->second : 0;
    if (mask != 0) {
      Value maskVal = arith::ConstantIntOp::create(rewriter, loc, mask, 32);
      Value masked = LLVM::AndOp::create(rewriter, loc, dimId, maskVal);
      Value dimPred = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::eq, masked, zero);
      if (pred)
        pred = LLVM::AndOp::create(rewriter, loc, pred, dimPred);
      else
        pred = dimPred;
    }
  }
  return pred;
}

// Combine two predicates with AND, handling null values.
static Value maybeAnd(ConversionPatternRewriter &rewriter, Location loc,
                      Value a, Value b) {
  if (!a)
    return b;
  if (!b)
    return a;
  return LLVM::AndOp::create(rewriter, loc, a, b);
}

struct AtomicRMWOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Declare an AIR intrinsic function if not already declared
  LLVMFuncOp declareAIR(ConversionPatternRewriter &rewriter, ModuleOp mod,
                        StringRef name, Type retTy,
                        ArrayRef<Type> argTys) const {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
      return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto fnTy = LLVMFunctionType::get(retTy, argTys, false);
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                              Linkage::External);
  }

  // Emit a direct AIR atomic intrinsic call (no CAS loop)
  Value emitDirectAtomic(ConversionPatternRewriter &rewriter, Location loc,
                         ModuleOp mod, StringRef airName, Type valueTy,
                         Value ptr) const {
    auto *ctx = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fn = declareAIR(rewriter, mod, airName, valueTy,
                         {ptrTy, valueTy, i32Ty, i32Ty, i1Ty});
    // Unused return needed: the call still needs a value operand.
    // Actually this helper is called from emitDirectAtomicCall below.
    (void)fn;
    return {};
  }

  // Emit CAS loop for f32 max/min:
  //   alloca expected
  //   load old from *ptr (via xchg 0 trick or just initial load)
  //   loop:
  //     store old → expected
  //     new_f = max/min(old_f, val_f)
  //     new_i = bitcast new_f → i32
  //     old_i = bitcast old_f → i32
  //     store old_i → expected
  //     old_ret = cmpxchg(ptr_i32, &expected, new_i, ...)
  //     expected_after = load expected
  //     cmp = icmp eq old_ret, old_i (success if unchanged)
  //     br cmp → done, loop
  //   done:
  //     result = bitcast old_ret → float
  Value emitF32CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                       ModuleOp mod, Value ptr, Value val, RMWOp rmwOp) const {
    auto *ctx = rewriter.getContext();
    auto f32Ty = Float32Type::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);  // device
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0); // private (alloca)

    // Declare cmpxchg intrinsic
    auto cmpxchgFn =
        declareAIR(rewriter, mod, "air.atomic.global.cmpxchg.weak.i32", i32Ty,
                   {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

    // Alloca for expected value (i32)
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                                  one, /*alignment=*/4);

    // Initial load: use xchg to atomically read current value
    // Actually, a simple non-atomic load is fine for the initial guess —
    // the CAS loop will retry if it's stale.
    Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, ptr);
    Value oldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldI32);

    // Create loop and exit blocks
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(afterBlock);

    // Branch from current block to loop
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{oldF32, oldI32}, loopBlock);

    // Loop block: phi for old_f32, old_i32
    loopBlock->addArgument(f32Ty, loc);
    loopBlock->addArgument(i32Ty, loc);
    Value phiOldF32 = loopBlock->getArgument(0);
    Value phiOldI32 = loopBlock->getArgument(1);

    rewriter.setInsertionPointToStart(loopBlock);

    // Compute new value
    Value newF32;
    if (rmwOp == RMWOp::MAX)
      newF32 = LLVM::MaximumOp::create(rewriter, loc, phiOldF32, val);
    else
      newF32 = LLVM::MinimumOp::create(rewriter, loc, phiOldF32, val);

    Value newI32 = LLVM::BitcastOp::create(rewriter, loc, i32Ty, newF32);

    // Store expected (old) into alloca
    LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);

    // CAS: cmpxchg(ptr, &expected, desired, succ_order=0, fail_order=0,
    // scope=2, vol=true)
    Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value volT = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldRet =
        LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
                             ValueRange{ptr, expectedAlloca, newI32, order0,
                                        order0, scope2, volT})
            .getResult();

    // Check success: old_ret == old_i32 means no other thread changed it
    Value success = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                         oldRet, phiOldI32);

    // On failure, the expected alloca now contains the current value
    Value failedOldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldRet);

    // Branch: success → afterBlock, failure → loopBlock with new old values
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock, ValueRange{},
                           loopBlock, ValueRange{failedOldF32, oldRet});

    // After block: result is the last successful old value (bitcast old_ret to
    // f32) But we need the result in afterBlock. Add a block arg.
    afterBlock->addArgument(f32Ty, loc);
    // Fix: afterBlock needs args from both paths. Actually we always arrive
    // from the success path of the CondBr above. Let me restructure.

    // Actually, CondBrOp success path goes to afterBlock — we need to pass the
    // result. Let me redo: erase the CondBr and rebuild with the right args.
    rewriter.eraseOp(success.getDefiningOp()->getBlock()->getTerminator());
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock,
                           ValueRange{phiOldF32}, loopBlock,
                           ValueRange{failedOldF32, oldRet});

    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock->getArgument(0);
  }

  // Emit CAS loop for f16/bf16 atomic add.
  // Strategy: bitcast ptr to i32*, load i32, extract the target half, compute,
  // pack back, cmpxchg i32.
  // Since Triton scalar atomics always target a single element, and the pointer
  // is already to the specific f16/bf16 element, we need to:
  //   1. Align ptr down to i32 boundary
  //   2. Determine which half (low/high) within the i32
  //   3. CAS loop on the i32
  // But actually, Triton's atomic_rmw on f16 gives us a ptr to a single f16.
  // We need to widen to i32 for the CAS. The element could be at an odd offset.
  //
  // Simpler approach: just use i16 CAS if Metal supports it.
  // Metal does NOT have i16 cmpxchg. So we must use i32.
  //
  // For the i32 widening approach:
  //   - ptr_i32 = ptr & ~3  (align down)
  //   - byte_offset = ptr & 3  → 0 or 2
  //   - shift = byte_offset * 8  → 0 or 16
  //   - mask = 0xFFFF << shift
  Value emitF16BF16CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                           ModuleOp mod, Value ptr, Value val, Type elemTy,
                           RMWOp rmwOp) const {
    auto *ctx = rewriter.getContext();
    auto i16Ty = IntegerType::get(ctx, 16);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);

    auto cmpxchgFn =
        declareAIR(rewriter, mod, "air.atomic.global.cmpxchg.weak.i32", i32Ty,
                   {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

    // Alloca for expected
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                                  one, /*alignment=*/4);

    // Compute aligned i32 pointer and shift amount
    // ptr_as_int = ptrtoint ptr
    Value ptrInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptr);
    // byte_offset = ptr_as_int & 3
    Value three64 = arith::ConstantIntOp::create(rewriter, loc, 3, 64);
    Value byteOff64 = LLVM::AndOp::create(rewriter, loc, ptrInt, three64);
    // aligned_ptr_int = ptr_as_int & ~3
    Value notThree64 = arith::ConstantIntOp::create(rewriter, loc, ~3LL, 64);
    Value alignedInt = LLVM::AndOp::create(rewriter, loc, ptrInt, notThree64);
    Value alignedPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, alignedInt);
    // shift = byte_offset * 8 (in i32)
    Value byteOff32 = LLVM::TruncOp::create(rewriter, loc, i32Ty, byteOff64);
    Value eight = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
    Value shift = LLVM::MulOp::create(rewriter, loc, byteOff32, eight);
    // mask = 0xFFFF << shift
    Value mask16 = arith::ConstantIntOp::create(rewriter, loc, 0xFFFF, 32);
    Value mask = LLVM::ShlOp::create(rewriter, loc, mask16, shift);
    Value notMask = LLVM::XOrOp::create(
        rewriter, loc, mask,
        arith::ConstantIntOp::create(rewriter, loc, -1, 32));

    // Convert val to f32 for computation, then back
    // Actually: val is already f16 or bf16. We do the add in f32 for
    // simplicity.
    Value valF32;
    if (elemTy.isF16())
      valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);
    else // bf16
      valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);

    // Initial load
    Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, alignedPtr);

    // Create loop and exit blocks
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.createBlock(afterBlock);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{oldI32}, loopBlock);

    loopBlock->addArgument(i32Ty, loc);
    Value phiOldI32 = loopBlock->getArgument(0);

    rewriter.setInsertionPointToStart(loopBlock);

    // Extract the target i16 from the i32 word
    Value shifted = LLVM::LShrOp::create(rewriter, loc, phiOldI32, shift);
    Value oldI16 = LLVM::TruncOp::create(rewriter, loc, i16Ty, shifted);

    // Convert old i16 to f32
    Value oldF32;
    if (elemTy.isF16()) {
      Value oldF16 =
          LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
      oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldF16);
    } else {
      Value oldBF16 = LLVM::BitcastOp::create(rewriter, loc,
                                              BFloat16Type::get(ctx), oldI16);
      oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldBF16);
    }

    // Compute: add in f32
    Value newF32 = arith::AddFOp::create(rewriter, loc, oldF32, valF32);

    // Convert back to i16
    Value newI16;
    if (elemTy.isF16()) {
      Value newF16 =
          arith::TruncFOp::create(rewriter, loc, Float16Type::get(ctx), newF32);
      newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newF16);
    } else {
      Value newBF16 = arith::TruncFOp::create(rewriter, loc,
                                              BFloat16Type::get(ctx), newF32);
      newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newBF16);
    }

    // Pack back into i32: (old & ~mask) | (new_i16_zext << shift)
    Value newI32Ext = LLVM::ZExtOp::create(rewriter, loc, i32Ty, newI16);
    Value newShifted = LLVM::ShlOp::create(rewriter, loc, newI32Ext, shift);
    Value cleared = LLVM::AndOp::create(rewriter, loc, phiOldI32, notMask);
    Value newI32 = LLVM::OrOp::create(rewriter, loc, cleared, newShifted);

    // Store expected, call cmpxchg
    LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);
    Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value volT = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldRet =
        LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
                             ValueRange{alignedPtr, expectedAlloca, newI32,
                                        order0, order0, scope2, volT})
            .getResult();

    Value success = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                         oldRet, phiOldI32);

    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock, ValueRange{},
                           loopBlock, ValueRange{oldRet});

    // After block: return the old element value (before our update)
    rewriter.setInsertionPointToStart(afterBlock);

    // Extract old element from the last successful i32
    // We need the pre-update value. The phiOldI32 at success is the value
    // that matched. Extract the element from it.
    // Actually, we need to pass the extracted old value out. Let me add a block
    // arg.

    // Reconstruct: on success, phiOldI32 was the matched expected.
    // The element we care about is oldI16 (extracted above). But that's in the
    // loop block. Simpler: add afterBlock arg with the old f16/bf16 value.

    // Redo: erase terminator and rebuild
    // The loop block terminator is the CondBrOp we just created.
    // We need to pass oldI16 to afterBlock on success.
    afterBlock->addArgument(elemTy, loc);

    auto *term = loopBlock->getTerminator();
    rewriter.setInsertionPoint(term);

    // Convert oldI16 to the element type
    Value oldElem;
    if (elemTy.isF16())
      oldElem =
          LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
    else
      oldElem = LLVM::BitcastOp::create(rewriter, loc, BFloat16Type::get(ctx),
                                        oldI16);

    rewriter.eraseOp(term);
    LLVM::CondBrOp::create(rewriter, loc, success, afterBlock,
                           ValueRange{oldElem}, loopBlock, ValueRange{oldRet});

    rewriter.setInsertionPointToStart(afterBlock);
    return afterBlock->getArgument(0);
  }

  // Emit a single scalar atomic: either direct AIR intrinsic or CAS loop.
  // Returns the old value.
  Value emitOneAtomic(ConversionPatternRewriter &rewriter, Location loc,
                      ModuleOp mod, Value ptr, Value val, Value mask,
                      Type valueElemTy, RMWOp rmwOp, const std::string &airName,
                      bool needsCAS) const {
    auto *ctx = rewriter.getContext();

    // CAS loop path
    if (needsCAS) {
      auto emitCAS = [&]() -> Value {
        if (valueElemTy.isF32())
          return emitF32CASLoop(rewriter, loc, mod, ptr, val, rmwOp);
        else
          return emitF16BF16CASLoop(rewriter, loc, mod, ptr, val, valueElemTy,
                                    rmwOp);
      };

      if (mask) {
        // Wrap CAS in conditional to skip non-canonical threads
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *casBlock = rewriter.createBlock(afterBlock);

        afterBlock->addArgument(valueElemTy, loc);

        rewriter.setInsertionPointToEnd(currentBlock);
        Value zeroVal = LLVM::ConstantOp::create(
            rewriter, loc, valueElemTy, rewriter.getZeroAttr(valueElemTy));
        LLVM::CondBrOp::create(rewriter, loc, mask, casBlock, ValueRange{},
                               afterBlock, ValueRange{zeroVal});

        rewriter.setInsertionPointToStart(casBlock);
        Value casResult = emitCAS();
        LLVM::BrOp::create(rewriter, loc, ValueRange{casResult}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        return afterBlock->getArgument(0);
      }
      return emitCAS();
    }

    // Direct AIR intrinsic path
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto fnTy = LLVMFunctionType::get(
        valueElemTy, {ptrTy, valueElemTy, i32Ty, i32Ty, i1Ty}, false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName))
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
    }
    auto atomicFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    if (mask) {
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(afterBlock);

      afterBlock->addArgument(valueElemTy, loc);

      rewriter.setInsertionPointToEnd(currentBlock);
      Value zeroVal = LLVM::ConstantOp::create(
          rewriter, loc, valueElemTy, rewriter.getZeroAttr(valueElemTy));
      LLVM::CondBrOp::create(rewriter, loc, mask, atomicBlock, ValueRange{},
                             afterBlock, ValueRange{zeroVal});

      rewriter.setInsertionPointToStart(atomicBlock);
      Value atomicResult =
          LLVM::CallOp::create(rewriter, loc, atomicFn,
                               ValueRange{ptr, val, order, scope, vol})
              .getResult();
      LLVM::BrOp::create(rewriter, loc, ValueRange{atomicResult}, afterBlock);

      rewriter.setInsertionPointToStart(afterBlock);
      return afterBlock->getArgument(0);
    } else {
      return LLVM::CallOp::create(rewriter, loc, atomicFn,
                                  ValueRange{ptr, val, order, scope, vol})
          .getResult();
    }
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto rmwOp = op.getAtomicRmwOp();
    auto tensorTy = dyn_cast<RankedTensorType>(op.getType());

    // Determine element type
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(op.getType());

    // Determine AIR intrinsic name or CAS loop
    std::string airName;
    bool needsCAS = false;
    if (valueElemTy.isF32()) {
      switch (rmwOp) {
      case RMWOp::FADD:
        airName = "air.atomic.global.add.f32";
        break;
      case RMWOp::XCHG:
        airName = "air.atomic.global.xchg.f32";
        break;
      default:
        return failure();
      }
    } else if (valueElemTy.isF16() || valueElemTy.isBF16()) {
      switch (rmwOp) {
      case RMWOp::FADD:
        needsCAS = true;
        break;
      default:
        return failure();
      }
    } else if (valueElemTy.isInteger(32)) {
      switch (rmwOp) {
      case RMWOp::ADD:
        airName = "air.atomic.global.add.s.i32";
        break;
      case RMWOp::MAX:
        airName = "air.atomic.global.max.s.i32";
        break;
      case RMWOp::MIN:
        airName = "air.atomic.global.min.s.i32";
        break;
      case RMWOp::UMAX:
        airName = "air.atomic.global.max.u.i32";
        break;
      case RMWOp::UMIN:
        airName = "air.atomic.global.min.u.i32";
        break;
      case RMWOp::AND:
        airName = "air.atomic.global.and.s.i32";
        break;
      case RMWOp::OR:
        airName = "air.atomic.global.or.s.i32";
        break;
      case RMWOp::XOR:
        airName = "air.atomic.global.xor.s.i32";
        break;
      case RMWOp::XCHG:
        airName = "air.atomic.global.xchg.i32";
        break;
      default:
        return failure();
      }
    } else {
      return failure();
    }

    if (!tensorTy) {
      // Scalar atomic: only thread 0 executes, broadcast result via TG.
      // Without this, all threads execute the atomic independently,
      // which corrupts spin-lock patterns (e.g. all threads doing
      // xchg(Lock, 0) allows another group to acquire between threads).
      auto i32Ty = IntegerType::get(ctx, 32);

      // Get thread_position_in_threadgroup[0]
      auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
      auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
          LLVMFuncOp::create(rewriter, mod.getLoc(),
                             "air.thread_position_in_threadgroup", tidFnTy,
                             Linkage::External);
      }
      auto tidFn =
          mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");

      Value tidStruct =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
      Value tid0 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                                ArrayRef<int64_t>{0});
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value isThread0 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, tid0, zero);

      // Combine thread-0 predicate with the op's own mask (e.g. sign-based
      // masks from float atomic_max/min decomposition).
      Value combinedMask = isThread0;
      if (llMask) {
        combinedMask = LLVM::AndOp::create(rewriter, loc, isThread0, llMask);
      }

      // Thread 0 (with mask) executes the atomic; others get a default value
      Value result =
          emitOneAtomic(rewriter, loc, mod, llPtr, llVal, combinedMask,
                        valueElemTy, rmwOp, airName, needsCAS);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Tensor atomic: unpack → per-element atomic → pack
    // Compute redundant thread predicate to mask out threads that
    // don't own unique elements (e.g. 128 threads, 4 elements → 124 idle)
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
    uint32_t regMask = freeVarMasks[StringAttr::get(ctx, "reg")];

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> resultVals(ptrElements.size());
    for (size_t i = 0; i < ptrElements.size(); ++i) {
      // Skip redundant register elements — reuse canonical result
      if (!isCanonicalIndex(i, regMask)) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      // Combine thread predicate with per-element mask
      Value mask = llMask ? maybeAnd(rewriter, loc, threadPred, maskElements[i])
                          : threadPred;
      Value res =
          emitOneAtomic(rewriter, loc, mod, ptrElements[i], valElements[i],
                        mask, valueElemTy, rmwOp, airName, needsCAS);
      resultVals[i] = res;
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

// Lower triton::AtomicCASOp → air.atomic.global.cmpxchg.weak.{i32,i64}
//
// Metal CAS signature:
//   i32 @air.atomic.global.cmpxchg.weak.i32(
//       ptr addrspace(1) ptr, ptr addrspace(0) expected,
//       i32 desired, i32 succ_order, i32 fail_order, i32 scope, i1 volatile)
// expected is passed by pointer and updated on failure.
// Returns old value.
struct AtomicCASOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Emit f16/bf16 CAS via i32 CAS on aligned word.
  // Strategy: align ptr to i32 boundary, read i32 word, replace the target
  // half with cmp/val, do i32 CAS. No loop needed — CAS semantics guarantee
  // atomicity. If the CAS fails (other half changed), return the old f16 value.
  Value emitF16BF16CAS(ConversionPatternRewriter &rewriter, Location loc,
                       ModuleOp mod, Value ptr, Value cmp, Value val,
                       Type elemTy) const {
    auto *ctx = rewriter.getContext();
    auto i16Ty = IntegerType::get(ctx, 16);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);

    // Declare i32 CAS
    auto airName = StringRef("air.atomic.global.cmpxchg.weak.i32");
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName)) {
        auto fnTy = LLVMFunctionType::get(
            i32Ty, {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty}, false);
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
      }
    }
    auto casFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    // Align pointer to i32 boundary
    Value ptrInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptr);
    Value one64 = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Value offset = LLVM::AndOp::create(
        rewriter, loc, ptrInt,
        one64); // 0 or 1 (byte offset within i16 pair → 0 or 2 bytes)
    // Actually: ptr points to f16 (2 bytes). Aligned i32 = ptr & ~3.
    Value three64 = arith::ConstantIntOp::create(rewriter, loc, 3, 64);
    Value notThree64 = arith::ConstantIntOp::create(rewriter, loc, ~3LL, 64);
    Value alignedInt = LLVM::AndOp::create(rewriter, loc, ptrInt, notThree64);
    Value alignedPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, alignedInt);

    // Byte offset within i32 word (0 or 2)
    Value byteOffset = LLVM::AndOp::create(rewriter, loc, ptrInt, three64);
    Value byteOffset32 =
        LLVM::TruncOp::create(rewriter, loc, i32Ty, byteOffset);
    // Shift in bits
    Value shift = LLVM::ShlOp::create(
        rewriter, loc, byteOffset32,
        arith::ConstantIntOp::create(rewriter, loc, 3, 32)); // bytes → bits

    // Mask for the target half: 0xFFFF shifted to position
    Value mask16 = arith::ConstantIntOp::create(rewriter, loc, 0xFFFF, 32);
    Value mask = LLVM::ShlOp::create(rewriter, loc, mask16, shift);
    Value notMask = LLVM::XOrOp::create(
        rewriter, loc, mask,
        arith::ConstantIntOp::create(rewriter, loc, -1, 32));

    // Truncate f32 cmp/val to f16/bf16 if needed (Triton CAS passes f32 cmp/val
    // for f16 ptrs)
    Value cmpElem = cmp, valElem = val;
    if (cmp.getType().isF32()) {
      cmpElem = arith::TruncFOp::create(rewriter, loc, elemTy, cmp);
      valElem = arith::TruncFOp::create(rewriter, loc, elemTy, val);
    }
    // Bitcast to i16 then zext to i32
    Value cmpI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, cmpElem);
    Value valI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, valElem);
    Value cmpI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, cmpI16);
    Value valI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, valI16);

    // Shift cmp/val to position within i32 word
    Value cmpShifted = LLVM::ShlOp::create(rewriter, loc, cmpI32, shift);
    Value valShifted = LLVM::ShlOp::create(rewriter, loc, valI32, shift);

    // Read current i32 word (non-atomic, as initial guess)
    Value curI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, alignedPtr);

    // Build expected i32 = (curI32 & ~mask) | cmpShifted
    Value otherBits = LLVM::AndOp::create(rewriter, loc, curI32, notMask);
    Value expectedI32 =
        LLVM::OrOp::create(rewriter, loc, otherBits, cmpShifted);
    // Build desired i32 = (curI32 & ~mask) | valShifted
    Value desiredI32 = LLVM::OrOp::create(rewriter, loc, otherBits, valShifted);

    // Alloca for expected — must be in entry block
    Value expectedAlloca;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *funcOp = rewriter.getInsertionBlock()->getParent()->getParentOp();
      auto &entryBlock = cast<LLVM::LLVMFuncOp>(funcOp).getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      Value oneI64 = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty,
                                              oneI64, /*alignment=*/4);
    }

    LLVM::StoreOp::create(rewriter, loc, expectedI32, expectedAlloca);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldI32 =
        LLVM::CallOp::create(rewriter, loc, casFn,
                             ValueRange{alignedPtr, expectedAlloca, desiredI32,
                                        order, order, scope, vol})
            .getResult();

    // Extract old f16/bf16 from the returned i32 word
    Value oldShifted = LLVM::LShrOp::create(rewriter, loc, oldI32, shift);
    Value oldI16 = LLVM::TruncOp::create(rewriter, loc, i16Ty, oldShifted);
    return LLVM::BitcastOp::create(rewriter, loc, elemTy, oldI16);
  }

  // Emit a single scalar CAS operation. Returns the old value.
  Value emitOneCAS(ConversionPatternRewriter &rewriter, Location loc,
                   ModuleOp mod, Value ptr, Value cmp, Value val,
                   Type valueTy) const {
    auto *ctx = rewriter.getContext();

    // f16/bf16 CAS via i32 word CAS
    if (valueTy.isF16() || valueTy.isBF16())
      return emitF16BF16CAS(rewriter, loc, mod, ptr, cmp, val, valueTy);

    // Only 32-bit CAS reaches here — i64/f64 are rejected at the pattern entry
    // (no 64-bit atomics on Apple GPU). f16/bf16 take the emitF16BF16CAS path
    // above.
    std::string airName;
    Type casTy;
    if (valueTy.isInteger(32) || valueTy.isF32()) {
      airName = "air.atomic.global.cmpxchg.weak.i32";
      casTy = IntegerType::get(ctx, 32);
    } else {
      llvm_unreachable("unsupported CAS type");
    }

    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i1Ty = IntegerType::get(ctx, 1);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>(airName)) {
        auto fnTy = LLVMFunctionType::get(
            casTy, {ptrTy, ptrTy0, casTy, i32Ty, i32Ty, i32Ty, i1Ty}, false);
        LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy,
                           Linkage::External);
      }
    }
    auto casFn = mod.lookupSymbol<LLVMFuncOp>(airName);

    Value cmpI = cmp, valI = val;
    bool needBitcast = (valueTy != casTy);
    if (needBitcast) {
      cmpI = LLVM::BitcastOp::create(rewriter, loc, casTy, cmp);
      valI = LLVM::BitcastOp::create(rewriter, loc, casTy, val);
    }

    // Alloca for expected — entry block
    Value one, expectedAlloca;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *funcOp = rewriter.getInsertionBlock()->getParent()->getParentOp();
      auto &entryBlock = cast<LLVM::LLVMFuncOp>(funcOp).getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, casTy, one,
                                              /*alignment=*/4);
    }

    LLVM::StoreOp::create(rewriter, loc, cmpI, expectedAlloca);

    Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
    Value vol = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

    Value oldI = LLVM::CallOp::create(rewriter, loc, casFn,
                                      ValueRange{ptr, expectedAlloca, valI,
                                                 order, order, scope, vol})
                     .getResult();

    if (needBitcast)
      return LLVM::BitcastOp::create(rewriter, loc, valueTy, oldI);
    return oldI;
  }

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto tensorTy = dyn_cast<RankedTensorType>(op.getType());
    Type valueTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : getTypeConverter()->convertType(op.getType());

    // Check supported types. Apple GPUs have NO 64-bit atomics — there is no
    // `air.atomic.global.cmpxchg.weak.i64` intrinsic, and emitting one crashes
    // the Metal compiler service (XPC_ERROR_CONNECTION_INTERRUPTED). Fail the
    // lowering cleanly (the op stays illegal -> compile error) rather than
    // producing AIR that brings down the GPU compiler.
    if (!(valueTy.isInteger(32) || valueTy.isF32() || valueTy.isF16() ||
          valueTy.isBF16()))
      return rewriter.notifyMatchFailure(
          op, "Apple GPU has no 64-bit atomics (i64/f64 CAS unsupported)");

    if (!tensorTy) {
      // Scalar CAS: only thread 0 executes, broadcast result to all threads.
      // Without this, a spin-lock pattern (while CAS == 1) deadlocks:
      // all threads spin independently but only one succeeds, and subsequent
      // barriers can never be reached by the blocked threads.
      auto *ctx = rewriter.getContext();
      auto i32Ty = IntegerType::get(ctx, 32);

      // Get thread_position_in_threadgroup[0] to identify thread 0
      auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
      auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
          LLVMFuncOp::create(rewriter, mod.getLoc(),
                             "air.thread_position_in_threadgroup", tidFnTy,
                             Linkage::External);
      }
      auto tidFn =
          mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");

      // Declare barrier
      auto voidTy = LLVMVoidType::get(ctx);
      auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
          LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier", barrFnTy,
                             Linkage::External);
      }
      auto barrFn = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier");

      // Create TG global to broadcast the CAS result
      Type tgElemTy = valueTy.isF32()   ? (Type)i32Ty
                      : valueTy.isF64() ? (Type)IntegerType::get(ctx, 64)
                                        : valueTy;
      std::string tgName = "__tg_cas_bcast";
      auto tgPtrTy = LLVMPointerType::get(ctx, 3);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        if (!mod.lookupSymbol<LLVM::GlobalOp>(tgName)) {
          auto arrTy = LLVMArrayType::get(tgElemTy, 1);
          LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                                 Linkage::Internal, tgName, Attribute(), 4, 3u);
        }
      }
      auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
      Value tgPtr =
          LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

      // Get tid and check if tid == 0
      Value tidStruct =
          LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
      Value tid0 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                                ArrayRef<int64_t>{0});
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      Value isThread0 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, tid0, zero);

      // Create blocks: thread0 does CAS and stores to TG; others skip to
      // barrier
      auto *currentBlock = rewriter.getInsertionBlock();
      auto *afterBlock =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      auto *casBlock = rewriter.createBlock(afterBlock);
      auto *mergeBlock = rewriter.createBlock(afterBlock);

      // Branch: thread 0 → casBlock, others → mergeBlock
      rewriter.setInsertionPointToEnd(currentBlock);
      LLVM::CondBrOp::create(rewriter, loc, isThread0, casBlock, mergeBlock);

      // casBlock: execute CAS, store result to TG
      rewriter.setInsertionPointToStart(casBlock);
      Value casResult =
          emitOneCAS(rewriter, loc, mod, llPtr, llCmp, llVal, valueTy);
      Value resultToStore = casResult;
      if (valueTy.isF32())
        resultToStore =
            LLVM::BitcastOp::create(rewriter, loc, i32Ty, casResult);
      else if (valueTy.isF64())
        resultToStore = LLVM::BitcastOp::create(
            rewriter, loc, IntegerType::get(ctx, 64), casResult);
      LLVM::StoreOp::create(rewriter, loc, resultToStore, tgPtr);
      LLVM::BrOp::create(rewriter, loc, mergeBlock);

      // mergeBlock: barrier then load from TG
      rewriter.setInsertionPointToStart(mergeBlock);
      // device memory fence (flag=1) to ensure the CAS side-effects are visible
      Value flagDev = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      Value scope1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
      LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flagDev, scope1});
      // TG memory fence (flag=2) to ensure the TG store is visible
      Value flagTG = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
      LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flagTG, scope1});
      Value loaded =
          LLVM::LoadOp::create(rewriter, loc, tgElemTy, tgPtr).getResult();
      if (valueTy.isF32())
        loaded = LLVM::BitcastOp::create(rewriter, loc, valueTy, loaded);
      else if (valueTy.isF64())
        loaded = LLVM::BitcastOp::create(rewriter, loc, valueTy, loaded);

      // Move remaining ops after the load
      rewriter.setInsertionPointAfter(loaded.getDefiningOp());
      // Splice afterBlock's contents into mergeBlock
      mergeBlock->getOperations().splice(mergeBlock->end(),
                                         afterBlock->getOperations());
      afterBlock->erase();

      rewriter.replaceOp(op, loaded);
      return success();
    }

    // Tensor CAS: unpack → per-element CAS → pack
    // Compute redundant thread predicate
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
    uint32_t regMask =
        freeVarMasks[StringAttr::get(rewriter.getContext(), "reg")];

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    SmallVector<Value> resultVals(ptrElements.size());
    for (size_t i = 0; i < ptrElements.size(); ++i) {
      // Skip redundant register elements
      if (!isCanonicalIndex(i, regMask)) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      // For CAS, wrap with thread predicate: skip non-canonical threads
      if (threadPred) {
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *casBlock = rewriter.createBlock(afterBlock);
        afterBlock->addArgument(valueTy, loc);

        rewriter.setInsertionPointToEnd(currentBlock);
        Value zeroVal = LLVM::ConstantOp::create(rewriter, loc, valueTy,
                                                 rewriter.getZeroAttr(valueTy));
        LLVM::CondBrOp::create(rewriter, loc, threadPred, casBlock,
                               ValueRange{}, afterBlock, ValueRange{zeroVal});

        rewriter.setInsertionPointToStart(casBlock);
        Value casResult = emitOneCAS(rewriter, loc, mod, ptrElements[i],
                                     cmpElements[i], valElements[i], valueTy);
        LLVM::BrOp::create(rewriter, loc, ValueRange{casResult}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        resultVals[i] = afterBlock->getArgument(0);
      } else {
        resultVals[i] = emitOneCAS(rewriter, loc, mod, ptrElements[i],
                                   cmpElements[i], valElements[i], valueTy);
      }
    }

    Value packed = packLLElements(loc, getTypeConverter(), resultVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

// Safe tt.store lowering: use conditional branch instead of read-modify-write.
//
// The LoadStoreToLLVM.cpp StoreOpConversion uses a read-modify-write pattern
// for masked stores: load(ptr); select(mask, val, loaded); store(ptr).
// This is broken when masked-out pointers alias with other threads' valid
// addresses (e.g., when M < RBLOCK and row strides cause overlap), creating
// race conditions and data corruption.
//
// This pattern uses a conditional branch: if (mask) store(val, ptr), which
// is safe regardless of the pointer value when the mask is false.
struct SafeStoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static SmallVector<Value> unpackElems(Value v, OpBuilder &b, Location loc) {
    if (!v)
      return {};
    if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
      SmallVector<Value> elems(sTy.getBody().size());
      for (size_t i = 0; i < elems.size(); ++i)
        elems[i] = ExtractValueOp::create(b, loc, sTy.getBody()[i], v,
                                          ArrayRef<int64_t>{(int64_t)i});
      return elems;
    }
    return {v};
  }

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Value val = adaptor.getValue();

    auto ptrs = unpackElems(ptr, rewriter, loc);
    auto vals = unpackElems(val, rewriter, loc);

    if (ptrs.size() != vals.size())
      return failure();

    Value maskOperand = adaptor.getMask();
    auto masks = maskOperand ? unpackElems(maskOperand, rewriter, loc)
                             : SmallVector<Value>{};

    // When the stored tensor is replicated across lanes/warps (the threadgroup
    // has more threads than the tensor has elements — e.g. a 64-thread group
    // writing a 1-element reduction result), every redundant thread computes
    // the same destination pointer. Without predication they all race-store to
    // that address, and a thread holding a stale/zeroed replica can win, so the
    // result is corrupted (observed as ~all-but-one programs storing 0). Emit a
    // redundant-thread predicate so only the canonical owner of each element
    // stores, and skip redundant register-replicated copies entirely.
    Value threadPred;
    uint32_t regMask = 0;
    if (isa<RankedTensorType>(op.getPtr().getType())) {
      auto mod = op->getParentOfType<ModuleOp>();
      auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
      threadPred =
          emitAppleRedundantThreadPredicate(freeVarMasks, rewriter, loc, mod);
      regMask = freeVarMasks[StringAttr::get(rewriter.getContext(), "reg")];
    }

    for (size_t i = 0; i < ptrs.size(); ++i) {
      // Skip redundant register-replicated elements: a non-canonical register
      // index holds the same value as its canonical sibling and addresses the
      // same location, so storing it is just a duplicate write.
      if (!isCanonicalIndex(i, regMask))
        continue;

      // Combine the redundant-thread predicate with the per-element store mask.
      Value mask;
      if (!masks.empty() && masks[i])
        mask = maybeAnd(rewriter, loc, threadPred, masks[i]);
      else
        mask = threadPred;

      if (mask) {
        // Use conditional branch: if (mask) store(val, ptr)
        // This avoids the read-modify-write of LoadStoreToLLVM which
        // causes data corruption when masked-out pointers alias valid data.
        auto *curBlock = rewriter.getInsertionBlock();
        auto curPoint = rewriter.getInsertionPoint();
        auto *endBlock = curBlock->splitBlock(curPoint);
        auto *thenBlock = rewriter.createBlock(endBlock);
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, mask, thenBlock, endBlock);
        rewriter.setInsertionPointToEnd(thenBlock);
        LLVM::StoreOp::create(rewriter, loc, vals[i], ptrs[i]);
        LLVM::BrOp::create(rewriter, loc, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
      } else {
        LLVM::StoreOp::create(rewriter, loc, vals[i], ptrs[i]);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Safe tt.load lowering: use conditional branch for masked loads.
//
// Similar to SafeStoreOpConversion, the LoadStoreToLLVM.cpp LoadOpConversion
// unconditionally loads from the pointer (even when masked out), then selects
// the result. Loading from out-of-bounds pointers is undefined behavior on
// Metal. This pattern uses a conditional branch to avoid the invalid load.
struct SafeLoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static SmallVector<Value> unpackElems(Value v, OpBuilder &b, Location loc) {
    if (!v)
      return {};
    if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
      SmallVector<Value> elems(sTy.getBody().size());
      for (size_t i = 0; i < elems.size(); ++i)
        elems[i] = ExtractValueOp::create(b, loc, sTy.getBody()[i], v,
                                          ArrayRef<int64_t>{(int64_t)i});
      return elems;
    }
    return {v};
  }

  static Value packElems(ArrayRef<Value> elems, OpBuilder &b, Location loc) {
    SmallVector<Type> tys;
    for (auto v : elems)
      tys.push_back(v.getType());
    auto sTy = LLVMStructType::getLiteral(b.getContext(), tys);
    Value result = UndefOp::create(b, loc, sTy);
    for (size_t i = 0; i < elems.size(); ++i)
      result = InsertValueOp::create(b, loc, sTy, result, elems[i],
                                     ArrayRef<int64_t>{(int64_t)i});
    return result;
  }

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // Scalar load: bare pointer
    if (!isa<LLVMStructType>(ptr.getType())) {
      Value maskOperand = adaptor.getMask();
      Value otherOperand = adaptor.getOther();
      if (maskOperand) {
        Value other = otherOperand
                          ? otherOperand
                          : LLVM::ZeroOp::create(rewriter, loc, resultTy);
        // Load unconditionally, select result.
        // For scalar loads the pointer is always valid.
        Value val = LLVM::LoadOp::create(rewriter, loc, resultTy, ptr);
        val = LLVM::SelectOp::create(rewriter, loc, maskOperand, val, other);
        rewriter.replaceOp(op, val);
      } else {
        Value val = LLVM::LoadOp::create(rewriter, loc, resultTy, ptr);
        rewriter.replaceOp(op, val);
      }
      return success();
    }

    // Tensor load: struct of pointers
    auto ptrs = unpackElems(ptr, rewriter, loc);
    auto sTy = dyn_cast<LLVMStructType>(resultTy);
    if (!sTy || sTy.getBody().size() != ptrs.size())
      return failure();

    Value maskOperand = adaptor.getMask();
    Value otherOperand = adaptor.getOther();
    auto masks = maskOperand ? unpackElems(maskOperand, rewriter, loc)
                             : SmallVector<Value>{};
    auto others = otherOperand ? unpackElems(otherOperand, rewriter, loc)
                               : SmallVector<Value>{};

    SmallVector<Value> loaded;
    for (size_t i = 0; i < ptrs.size(); ++i) {
      if (!masks.empty()) {
        Value other = others.empty() ? LLVM::ZeroOp::create(rewriter, loc,
                                                            sTy.getBody()[i])
                                     : others[i];
        // Conditional load via branch to avoid accessing invalid pointers
        // when the mask is false (e.g., rindex >= M with M < RBLOCK).
        auto *curBlock = rewriter.getInsertionBlock();
        auto curPoint = rewriter.getInsertionPoint();
        auto *endBlock = curBlock->splitBlock(curPoint);
        auto *thenBlock = rewriter.createBlock(endBlock);
        endBlock->addArgument(sTy.getBody()[i], loc);
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, masks[i], thenBlock, ValueRange{},
                               endBlock, ValueRange{other});
        rewriter.setInsertionPointToEnd(thenBlock);
        Value val =
            LLVM::LoadOp::create(rewriter, loc, sTy.getBody()[i], ptrs[i]);
        LLVM::BrOp::create(rewriter, loc, ValueRange{val}, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
        loaded.push_back(endBlock->getArgument(0));
      } else {
        loaded.push_back(
            LLVM::LoadOp::create(rewriter, loc, sTy.getBody()[i], ptrs[i]));
      }
    }
    rewriter.replaceOp(op, packElems(loaded, rewriter, loc));
    return success();
  }
};

// Lower ttg::WarpIdOp → air.dispatch_thread_id[0] / threadsPerWarp.
struct WarpIdOpConversion
    : public mlir::ConvertOpToLLVMPattern<triton::gpu::WarpIdOp> {
  using mlir::ConvertOpToLLVMPattern<
      triton::gpu::WarpIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpIdOp op,
                  triton::gpu::WarpIdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto mod = op->getParentOfType<ModuleOp>();

    // Use air.thread_position_in_threadgroup (returns [3 x i32]) + extractvalue
    // 0. _add_air_metadata() rewrites this call+extractvalue to a function arg.
    auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
        LLVMFuncOp::create(rewriter, mod.getLoc(),
                           "air.thread_position_in_threadgroup", tidFnTy,
                           Linkage::External);
    }
    auto tidFn =
        mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");
    Value tidStruct =
        LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
    Value tid = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                             ArrayRef<int64_t>{0});

    // warpId = tid / threadsPerWarp (32 on Apple simdgroup)
    int tpw = ttg::lookupThreadsPerWarp(rewriter);
    Value warpSize = arith::ConstantIntOp::create(rewriter, loc, tpw, 32);
    Value warpId = arith::DivUIOp::create(rewriter, loc, tid, warpSize);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

// Lower triton::GetNumProgramsOp → call @air.threadgroups_per_grid() +
// extractvalue Returns the grid dimension (number of threadgroups) for the
// given axis.
struct GetNumProgramsOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto arrTy = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto fnTy = LLVMFunctionType::get(arrTy, {}, false);

    auto fnName = StringRef("air.threadgroups_per_grid");
    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod.lookupSymbol<LLVMFuncOp>(fnName)) {
      OpBuilder b(mod.getBodyRegion());
      b.setInsertionPointToStart(mod.getBody());
      LLVMFuncOp::create(b, mod.getLoc(), fnName, fnTy, Linkage::External);
    }
    auto fn = mod.lookupSymbol<LLVMFuncOp>(fnName);

    Value gridStruct =
        LLVM::CallOp::create(rewriter, loc, fn, ValueRange{}).getResult();
    int axis = static_cast<int>(op.getAxis());
    Value result = LLVM::ExtractValueOp::create(
        rewriter, loc, i32Ty, gridStruct, ArrayRef<int64_t>{(int64_t)axis});
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower triton::FuncOp → LLVM::LLVMFuncOp for Apple Metal kernels.
//
// Metal passes scalar kernel args (i32, i64, etc.) via setBytes — a pointer
// to constant address space (addrspace 2). The LLVM IR must reflect this:
// scalar args become `i32 addrspace(2)*` pointers, and we insert explicit
// loads at function entry. This matches what `xcrun metal` emits for
// `constant T&` parameters, and eliminates the Python regex workaround.
//
// Pointer args (addrspace 1 = device) are passed through unchanged.
struct AppleFuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = funcOp.getContext();
    auto loc = funcOp.getLoc();
    bool isKernel = triton::isKernel(funcOp);

    // Build new LLVM arg types.
    // Kernel: scalar i32/i64/etc → addrspace(2)* pointer (Metal constant
    // buffer). Device function: convert types directly, no addrspace wrapping.
    SmallVector<Type> newArgTypes;
    SmallVector<bool> isScalar;
    for (auto argTy : funcOp.getFunctionType().getInputs()) {
      Type converted = getTypeConverter()->convertType(argTy);
      if (!converted)
        return failure();
      if (isKernel && isa<IntegerType>(converted)) {
        auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/2);
        newArgTypes.push_back(ptrTy);
        isScalar.push_back(true);
      } else {
        newArgTypes.push_back(converted);
        isScalar.push_back(false);
      }
    }

    // Build return type: void for kernels, converted type for device functions.
    Type retTy = LLVM::LLVMVoidType::get(ctx);
    if (!isKernel) {
      auto results = funcOp.getFunctionType().getResults();
      if (results.size() == 1) {
        retTy = getTypeConverter()->convertType(results[0]);
        if (!retTy)
          return failure();
      } else if (results.size() > 1) {
        // Pack multiple return values into a struct
        SmallVector<Type> memberTypes;
        for (auto resTy : results) {
          Type converted = getTypeConverter()->convertType(resTy);
          if (!converted)
            return failure();
          memberTypes.push_back(converted);
        }
        retTy = LLVM::LLVMStructType::getLiteral(ctx, memberTypes);
      }
    }

    auto llvmFuncTy = LLVM::LLVMFunctionType::get(retTy, newArgTypes);
    auto newFuncOp = LLVM::LLVMFuncOp::create(
        rewriter, loc, funcOp.getName(), llvmFuncTy, LLVM::Linkage::External);

    // Move function body into new func
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Fix up block argument types and insert loads for scalar kernel args
    Block &entryBlock = newFuncOp.getBody().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&entryBlock);

    for (unsigned i = 0; i < newArgTypes.size(); ++i) {
      BlockArgument oldArg = entryBlock.getArgument(i);
      if (isScalar[i]) {
        oldArg.setType(newArgTypes[i]);
        auto origTy = getTypeConverter()->convertType(
            funcOp.getFunctionType().getInput(i));
        // Mark the scalar-arg load volatile so it is never eliminated. Once the
        // arg becomes an opaque addrspace(2) pointer, its original width is not
        // recoverable from the type; the scalar-buffer-packing pass recovers it
        // from this load's result type. If the load were dead (e.g. it feeds
        // only a bounds-check assert that later gets elided), the pass would
        // have to guess the width, mis-size the packed slot, and corrupt every
        // following scalar's byte offset. Keeping the load alive guarantees an
        // exact, signature-matching layout. A constant-buffer load is cheap and
        // any genuinely unused result is dropped after packing rewrites it.
        Value loaded = LLVM::LoadOp::create(rewriter, loc, origTy, oldArg,
                                            /*alignment=*/0,
                                            /*isVolatile=*/true);
        oldArg.replaceAllUsesExcept(loaded, loaded.getDefiningOp());
      } else {
        oldArg.setType(newArgTypes[i]);
      }
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct AppleReturnOpConversion
    : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    if (operands.size() <= 1) {
      // Void or single return — direct lowering
      LLVM::ReturnOp::create(rewriter, op.getLoc(), operands);
    } else {
      // Multi-return: pack values into a struct (matches the struct return
      // type created by AppleFuncOpConversion for multi-result functions).
      auto loc = op.getLoc();
      auto *ctx = rewriter.getContext();
      SmallVector<Type> memberTypes;
      for (auto v : operands)
        memberTypes.push_back(v.getType());
      auto structTy = LLVM::LLVMStructType::getLiteral(ctx, memberTypes);

      Value packed = LLVM::UndefOp::create(rewriter, loc, structTy);
      for (unsigned i = 0; i < operands.size(); ++i) {
        packed = LLVM::InsertValueOp::create(
            rewriter, loc, packed, operands[i],
            ArrayRef<int64_t>{static_cast<int64_t>(i)});
      }
      LLVM::ReturnOp::create(rewriter, loc, ValueRange{packed});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower triton::PrintOp → no-op (Metal has no printf).
// Erase the op so it doesn't block legalization.
struct ApplePrintOpConversion : public ConvertOpToLLVMPattern<triton::PrintOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower triton::AssertOp → no-op (Metal has no device-side assert).
struct AppleAssertOpConversion
    : public ConvertOpToLLVMPattern<triton::AssertOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower triton::CallOp → LLVM::CallOp for Apple device function calls.
// Unlike CUDA, we don't append shared memory stack pointers.
struct AppleCallOpConversion : public ConvertOpToLLVMPattern<triton::CallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = callOp.getLoc();
    auto promotedOperands = getTypeConverter()->promoteOperands(
        loc, callOp->getOperands(), adaptor.getOperands(), rewriter);

    // Build result type
    SmallVector<Type> resultTypes;
    for (auto resTy : callOp.getResultTypes()) {
      Type converted = getTypeConverter()->convertType(resTy);
      if (!converted)
        return failure();
      resultTypes.push_back(converted);
    }

    if (resultTypes.size() <= 1) {
      // Single or void return — direct lowering
      auto newCallOp = LLVM::CallOp::create(
          rewriter, loc,
          resultTypes.empty() ? TypeRange() : TypeRange(resultTypes),
          promotedOperands, callOp->getAttrs());
      newCallOp.getProperties().setOpBundleSizes(
          rewriter.getDenseI32ArrayAttr({}));
      newCallOp.getProperties().setOperandSegmentSizes(
          {static_cast<int>(promotedOperands.size()), 0});
      rewriter.replaceOp(callOp, newCallOp.getResults());
    } else {
      // Multi-return: call returns a struct, extract each field
      auto *ctx = rewriter.getContext();
      auto structTy = LLVM::LLVMStructType::getLiteral(ctx, resultTypes);
      auto newCallOp =
          LLVM::CallOp::create(rewriter, loc, TypeRange(structTy),
                               promotedOperands, callOp->getAttrs());
      newCallOp.getProperties().setOpBundleSizes(
          rewriter.getDenseI32ArrayAttr({}));
      newCallOp.getProperties().setOperandSegmentSizes(
          {static_cast<int>(promotedOperands.size()), 0});

      SmallVector<Value> extracted;
      Value structResult = newCallOp.getResult();
      for (unsigned i = 0; i < resultTypes.size(); ++i) {
        extracted.push_back(LLVM::ExtractValueOp::create(
            rewriter, loc, resultTypes[i], structResult,
            ArrayRef<int64_t>{static_cast<int64_t>(i)}));
      }
      rewriter.replaceOp(callOp, extracted);
    }
    return success();
  }
};

// Lower ExternElementwiseOp (libdevice calls) to LLVM intrinsics.
// Maps __nv_exp → llvm.exp.f32, __nv_sin → llvm.sin.f32, etc.
struct ExternElementwiseOpAppleConversion
    : public mlir::triton::gpu::ElementwiseOpConversionBase<
          triton::ExternElementwiseOp, ExternElementwiseOpAppleConversion> {
  using Base = mlir::triton::gpu::ElementwiseOpConversionBase<
      triton::ExternElementwiseOp, ExternElementwiseOpAppleConversion>;
  using Base::Base;

  SmallVector<Value>
  createDestOps(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                gpu::MultipleOperandsRange operands, Location loc) const {

    StringRef symbol = op.getSymbol();

    // Map libdevice symbols to LLVM intrinsic names
    // __nv_exp → llvm.exp, __nv_sin → llvm.sin, etc.
    static const llvm::StringMap<StringRef> unaryMap = {
        {"__nv_exp", "llvm.exp"},
        {"__nv_exp2", "llvm.exp2"},
        {"__nv_log", "llvm.log"},
        {"__nv_log2", "llvm.log2"},
        {"__nv_log10", "llvm.log10"},
        {"__nv_sin", "llvm.sin"},
        {"__nv_cos", "llvm.cos"},
        {"__nv_sqrt", "llvm.sqrt"},
        {"__nv_rsqrt", "llvm.sqrt"}, // will invert
        {"__nv_fabs", "llvm.fabs"},
        {"__nv_fabsf", "llvm.fabs"},
        {"__nv_floor", "llvm.floor"},
        {"__nv_floorf", "llvm.floor"},
        {"__nv_ceil", "llvm.ceil"},
        {"__nv_ceilf", "llvm.ceil"},
        {"__nv_trunc", "llvm.trunc"},
        {"__nv_truncf", "llvm.trunc"},
        {"__nv_nearbyint", "llvm.nearbyint"},
        {"__nv_rint", "llvm.rint"},
        {"__nv_llrint", "llvm.lrint"},
        {"__nv_expm1", "llvm.exp"}, // approx: will subtract 1
    };
    static const llvm::StringMap<StringRef> binaryMap = {
        {"__nv_copysign", "llvm.copysign"},
        {"__nv_copysignf", "llvm.copysign"},
        {"__nv_fmax", "llvm.maxnum"},
        {"__nv_fmaxf", "llvm.maxnum"},
        {"__nv_fmin", "llvm.minnum"},
        {"__nv_fminf", "llvm.minnum"},
        {"__nv_pow", "llvm.pow"},
        {"__nv_powf", "llvm.pow"},
        {"__nv_atan2", ""}, // no direct intrinsic
        {"__nv_atan2f", ""},
        {"__nv_fmod", ""},
        {"__nv_fmodf", ""},
    };

    // Unary intrinsics
    auto uit = unaryMap.find(symbol);
    if (uit != unaryMap.end() && !uit->second.empty()) {
      StringRef intrName = uit->second;
      // Build type-suffixed name: llvm.exp → llvm.exp.f32
      std::string fullName = (intrName + "." +
                              (elemTy.isF32()   ? "f32"
                               : elemTy.isF64() ? "f64"
                                                : "f16"))
                                 .str();

      auto funcTy = LLVM::LLVMFunctionType::get(elemTy, {elemTy});
      auto funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
          rewriter, op, fullName, funcTy);
      Value result = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0])
                         .getResult();

      // rsqrt = 1.0 / sqrt
      if (symbol.contains("rsqrt")) {
        Value one = LLVM::ConstantOp::create(
            rewriter, loc, elemTy, rewriter.getFloatAttr(elemTy, 1.0));
        result = LLVM::FDivOp::create(rewriter, loc, one, result);
      }
      // expm1 = exp(x) - 1
      if (symbol.contains("expm1")) {
        Value one = LLVM::ConstantOp::create(
            rewriter, loc, elemTy, rewriter.getFloatAttr(elemTy, 1.0));
        result = LLVM::FSubOp::create(rewriter, loc, result, one);
      }
      return {result};
    }

    // Binary intrinsics
    auto bit = binaryMap.find(symbol);
    if (bit != binaryMap.end() && !bit->second.empty()) {
      StringRef intrName = bit->second;
      std::string fullName = (intrName + "." +
                              (elemTy.isF32()   ? "f32"
                               : elemTy.isF64() ? "f64"
                                                : "f16"))
                                 .str();
      auto funcTy = LLVM::LLVMFunctionType::get(elemTy, {elemTy, elemTy});
      auto funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
          rewriter, op, fullName, funcTy);
      SmallVector<Value> args = {operands[0][0], operands[0][1]};
      return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, args).getResult()};
    }

    // fmod → LLVM::FRemOp
    if (symbol.contains("fmod")) {
      return {LLVM::FRemOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }

    // Trig functions not in LLVM intrinsics — use math lib calls
    // tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh
    // For now, emit as external function calls; metal-llc's
    // MetalLLVMToAIRIntrinsics pass maps them to air.* builtins.
    {
      auto funcTy = operands[0].size() == 1
                        ? LLVM::LLVMFunctionType::get(elemTy, {elemTy})
                        : LLVM::LLVMFunctionType::get(elemTy, {elemTy, elemTy});
      auto funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(rewriter, op,
                                                               symbol, funcTy);
      return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0])
                  .getResult()};
    }
  }
};

// ── Pipeliner async copy lowering ────────────────────────────────────────
//
// Lower ttg.async_copy_global_to_local → synchronous per-element copy
// Lower ttg.async_commit_group → no-op (token = 0)
// Lower ttg.async_wait → threadgroup barrier
//
// The Triton software pipeliner generates these ops for multi-buffered
// load-compute overlap. On NVIDIA, async_copy lowers to cp.async (hw DMA).
// On Apple GPU, we lower to per-element loads + shared memory stores
// using Triton's lowerLocalLdSt for correct layout mapping.
//
// The pipeliner's multi-buffering still provides benefit by structuring
// the code for compute/copy overlap across loop iterations.
//
// When possible, we emit true async DMA via air.simdgroup_async_copy_2d.
// This requires: (1) extractable row stride from the MLIR def chain,
// (2) no mask (unmasked copy), (3) 2D tile.
// Otherwise we fall back to sync per-element copy via lowerLocalLdSt.

// Helper: get or create an external function declaration in the module
static LLVMFuncOp getOrCreateFn(ModuleOp mod, RewriterBase &rewriter,
                                StringRef name, Type retTy,
                                ArrayRef<Type> argTys) {
  if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto fnTy = LLVMFunctionType::get(retTy, argTys, false);
  return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                            Linkage::External);
}

// Async copy event storage notes:
//
// IMPORTANT: Metal's air.wait_simdgroup_events expects a thread-local
// (addrspace 0) pointer-to-pointer, NOT a threadgroup (addrspace 3) pointer.
// Using a TG global crashes the GPU compiler.
//
// The alloca type is `ptr addrspace(3)` (a single event pointer), matching
// the reference pattern: `%ev = alloca %event_t addrspace(3)*, align 8`.
// Metal v1 bitcode doesn't handle arrays of typed pointers well, so each copy
// uses its own scalar slot rather than one shared array.

// Create a FRESH single-event alloca in the function entry block, one per async
// copy. Each async copy owns its own scalar event slot; the slot pointer is
// threaded out as the op's !ttg.async.token result so the matching async_wait
// waits on exactly this copy (see the AsyncToken type conversion). A single
// shared slot made air.wait_simdgroup_events wait on only the LAST copy, and a
// scavenged "wait on every slot" set waited on the WRONG (loop-rotated) buffer
// and on slots not yet stored on the first iteration (UB). Per-copy slots stay
// scalar to avoid the Metal-v1 "array of typed pointers" bitcode limitation.
//
// The slot is zero-initialized in the entry block so that a token reaching a
// wait WITHOUT a preceding store (the masked-skip branch, the sync-copy
// fallback, or a wait that is loop-hoisted ahead of the store on iteration 0)
// holds a complete/empty event: air.wait_simdgroup_events on a zero event slot
// is a real no-op, never a read of an uninitialized pointer.
static Value createEventAlloca(Operation *op, RewriterBase &rewriter) {
  auto *ctx = op->getContext();
  auto ptrTy3 = LLVMPointerType::get(ctx, 3);
  auto ptrTy0 = LLVMPointerType::get(ctx, 0);
  auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  OpBuilder::InsertionGuard guard(rewriter);
  if (funcOp)
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  auto i64Ty = IntegerType::get(ctx, 64);
  Value one = LLVM::ConstantOp::create(rewriter, op->getLoc(), i64Ty,
                                       rewriter.getI64IntegerAttr(1));
  auto alloca = LLVM::AllocaOp::create(rewriter, op->getLoc(), ptrTy0, ptrTy3,
                                       one, /*alignment=*/8);
  Value nullEv = LLVM::ZeroOp::create(rewriter, op->getLoc(), ptrTy3);
  LLVM::StoreOp::create(rewriter, op->getLoc(), nullEv, alloca.getResult());
  return alloca.getResult();
}

// A standalone zero-initialized event slot for an async-copy lowering path that
// produces no real event (the sync fallback). Waiting on it is a no-op.
static Value createCompletedEventSlot(Operation *op, RewriterBase &rewriter) {
  return createEventAlloca(op, rewriter);
}

// air.simdgroup_async_copy_2d is a SIMDGROUP-cooperative DMA: each warp issues
// its own copy with a single WARP-UNIFORM tile origin and waits its own event.
// When warpsPerCTA[outerDim] > 1 the staged pipeline buffer's outer (slowest)
// dim is split across warps, but the warp-uniform origin makes every warp
// redundantly DMA the WHOLE tile into the same shared region. Those concurrent
// copies write-write race: a per-simdgroup air.wait_simdgroup_events only
// drains the issuing warp's own copy, and the threadgroup barrier after the
// wait fences regular threadgroup stores, not the async DMA engine's writes, so
// a sibling warp can read the buffer before another warp's still-in-flight copy
// has finished (verified: both the partitioned A operand and the replicated B
// operand of the matmul_layer_norm 32x64x16 nw4 kernel corrupt nondeterminis-
// tically). The AIR JIT has no cooperative-copy lowering that merges/serializes
// them. Keep any such multi-warp-outer-dim copy on the layout-exact synchronous
// copy (real threadgroup stores, which the membar passes order correctly).
// Single-warp-per-outer-dim copies (warpsPerCTA[outerDim] == 1) are race-free
// and stay on the fast async DMA path.
//
// CAVEAT: this predicate is deliberately BROAD: it also routes correct nw>=2
// GEMM operand copies to the sync path, a PERF (not correctness) regression on
// those configs, because a tighter race-vs-safe discriminator could not be
// proven sound here. The proper fix is a cooperative multi-warp async copy (or
// a narrower predicate); tracked as the async-vs-sync path unification rework.
static bool asyncCopyOuterDimCrossWarp(ttg::AsyncCopyGlobalToLocalOp op) {
  auto srcTy = op.getSrc().getType();
  auto enc = srcTy.getEncoding();
  auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(enc);
  if (!blocked)
    return false;
  auto warpsPerCTA = blocked.getWarpsPerCTA();
  auto order = blocked.getOrder();
  if (warpsPerCTA.empty() || order.empty())
    return false;
  // Outer (slowest-varying) dim is the last entry of order.
  unsigned outerDim = order.back();
  if (outerDim >= warpsPerCTA.size())
    return false;
  return warpsPerCTA[outerDim] > 1;
}

// Partition factor for a multi-warp-outer async copy. Returns the number of
// warps along the tile's outer (row) dim when the row dim divides evenly across
// them (so the tile splits into that many disjoint per-warp row bands), else 0
// (not cleanly partitionable;
// caller keeps the sync bail). When >0 the copy is split into that many
// disjoint horizontal bands, one per outer-dim warp, so the warps no longer
// write-write race on a shared region.
static unsigned asyncCopyOuterPartition(ttg::AsyncCopyGlobalToLocalOp op) {
  auto srcTy = op.getSrc().getType();
  auto blocked =
      dyn_cast_or_null<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
  if (!blocked)
    return 0;
  auto warpsPerCTA = blocked.getWarpsPerCTA();
  auto order = blocked.getOrder();
  auto shape = srcTy.getShape();
  if (warpsPerCTA.empty() || order.empty() || shape.size() != 2)
    return 0;
  unsigned outerDim = order.back();
  if (outerDim >= warpsPerCTA.size())
    return 0;
  unsigned nWarpsOuter = warpsPerCTA[outerDim];
  if (nWarpsOuter <= 1)
    return 0;
  if (shape[0] % nWarpsOuter != 0) // row dim must split evenly into bands
    return 0;
  return nWarpsOuter;
}

// Emit the outer-dim warp index wR for the partitioned copy: warpId = tid/32
// decomposed against the layout's warp order so wR selects which row band this
// warp owns. Mirrors the wR decomposition at the MMA index emitters.
static Value emitOuterWarpIndex(ttg::AsyncCopyGlobalToLocalOp op,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {
  auto *ctx = op.getContext();
  auto mod = op->getParentOfType<ModuleOp>();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto srcTy = op.getSrc().getType();
  auto blocked = cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
  auto warpsPerCTA = blocked.getWarpsPerCTA();
  auto order = blocked.getOrder();
  unsigned outerDim = order.back();
  unsigned innerDim = order.front();
  // colFastest: inner (fastest) dim is order[0]; outer warps are the SLOW
  // factor of warpId (wR = warpId / warpsInner). Otherwise wR = warpId %
  // warpsOuter.
  bool colFastest = (order[0] == innerDim) && (innerDim != outerDim);
  unsigned warpsOuter = warpsPerCTA[outerDim];
  unsigned warpsInner = warpsPerCTA[innerDim];

  // tid = thread_position_in_threadgroup[0].
  auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
  auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
      LLVMFuncOp::create(rewriter, mod.getLoc(),
                         "air.thread_position_in_threadgroup", tidFnTy,
                         Linkage::External);
  }
  auto tidFn =
      mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");
  Value tidStruct =
      LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{}).getResult();
  Value tid = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty, tidStruct,
                                           ArrayRef<int64_t>{0});
  Value c32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                       rewriter.getI32IntegerAttr(32));
  Value warpId = LLVM::UDivOp::create(rewriter, loc, tid, c32);
  if (colFastest) {
    Value wIn = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(warpsInner));
    return LLVM::UDivOp::create(rewriter, loc, warpId, wIn);
  }
  Value wOut = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(warpsOuter));
  return LLVM::URemOp::create(rewriter, loc, warpId, wOut);
}

// Struct to hold all components extracted from the async_copy pointer chain.
struct AsyncCopyPtrInfo {
  Value stride;   // Row stride scalar (MLIR, in elements)
  Value basePtr;  // Scalar base pointer (MLIR, tt.ptr)
  Value rowStart; // Scalar first-row index (MLIR, i32/i64), or nullptr if 0
  Value colStart; // Scalar first-col index (MLIR, i32/i64), or nullptr if 0
  // Compile-time fallbacks used when the first-element offset is a FOLDED splat
  // constant (arith.constant dense<C>) rather than a tt.splat of an SSA scalar.
  // The software pipeliner emits the prefetched buffer's K-block offset this
  // way (make_range + dense<BLOCK_K>); without capturing it, every prefetched
  // slab read K-block 0, corrupting num_stages>=3. INT64_MIN means "no
  // constant".
  int64_t rowStartConst = 0;
  int64_t colStartConst = 0;
  // Row stride as a compile-time constant (inductor dense<C>); INT64_MIN means
  // use the `stride` SSA Value instead.
  int64_t strideConst = INT64_MIN;
};

// First element of a 1D index tensor as a compile-time constant, when the
// tensor is `addi(make_range, dense<C>)` / `dense<C>` / `make_range` (=0).
// Returns true
// + sets `out` on success; false when the offset is not a folded constant (the
// caller then falls back to the SSA-scalar extractFirstElemScalar path).
static bool extractFirstElemConst(Value tensor, int64_t &out,
                                  bool peelModulo = false) {
  auto *defOp = tensor.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<triton::ExpandDimsOp>(defOp))
    return extractFirstElemConst(defOp->getOperand(0), out, peelModulo);
  // See extractFirstElemScalar: peel a proven-no-op boundary modulo so a
  // constant row origin survives the wrap instead of folding to 0.
  if (peelModulo && isa<arith::RemSIOp, arith::RemUIOp>(defOp))
    return extractFirstElemConst(defOp->getOperand(0), out, peelModulo);
  if (isa<triton::MakeRangeOp>(defOp)) {
    out = 0;
    return true;
  }
  if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
      if (dense.isSplat()) {
        out = dense.getSplatValue<APInt>().getSExtValue();
        return true;
      }
    return false;
  }
  if (isa<arith::AddIOp>(defOp)) {
    int64_t a, b;
    if (extractFirstElemConst(defOp->getOperand(0), a, peelModulo) &&
        extractFirstElemConst(defOp->getOperand(1), b, peelModulo)) {
      out = a + b;
      return true;
    }
  }
  return false;
}

// Extract the first-element scalar from a 1D index tensor.
//
// Patterns:
//   addi(splat(scalar), make_range(0, N)) → scalar
//   splat(scalar) → scalar
//   make_range(0, N) → nullptr (first element is 0)
//   expand_dims(inner_1d, axis) → recurse on inner_1d
static Value extractFirstElemScalar(Value tensor, bool peelModulo = false) {
  auto *defOp = tensor.getDefiningOp();
  if (!defOp)
    return nullptr;

  // Peel through expand_dims
  if (isa<triton::ExpandDimsOp>(defOp))
    return extractFirstElemScalar(defOp->getOperand(0), peelModulo);

  // Peel through a boundary-wrap modulo to its dividend. The caller sets
  // peelModulo ONLY when the wrap was already proven block-aligned (a no-op
  // over the tile), so the first element of (dividend % M) equals the first
  // element of the dividend. Without this, a program-id-dependent row origin
  // such as (pid_m*BLOCK_M + arange) % M_total is dropped to 0 and the async
  // DMA reads every program's tile from row 0 (correct only for pid_m == 0).
  if (peelModulo && isa<arith::RemSIOp, arith::RemUIOp>(defOp))
    return extractFirstElemScalar(defOp->getOperand(0), peelModulo);

  // addi(splat(scalar), make_range(0, N)) → first elem = scalar + 0 = scalar
  if (isa<arith::AddIOp>(defOp)) {
    for (unsigned i = 0; i < 2; i++) {
      auto *op = defOp->getOperand(i).getDefiningOp();
      if (op && isa<triton::SplatOp>(op))
        return op->getOperand(0);
    }
    return nullptr;
  }

  // splat(scalar) → scalar (uniform tensor)
  if (isa<triton::SplatOp>(defOp))
    return defOp->getOperand(0);

  // make_range(0, N) → first element is 0 (return nullptr to signal zero)
  if (isa<triton::MakeRangeOp>(defOp))
    return nullptr; // caller treats nullptr as zero

  return nullptr;
}

// Detect whether a tensor value's def chain contains a modulo (arith.remui /
// arith.remsi). The async-DMA fast path below reconstructs each row's device
// address as `basePtr + rowStart*stride + ...` with a single constant stride,
// which is only valid when the index is an affine function of the program/
// thread coordinates. Inductor's mm template wraps row/col indices with
// `(pid*BLOCK + arange) % M` (the standard bounds-wrapping idiom); that modulo
// makes the per-row stride non-constant (it folds back to the tensor origin at
// the wrap), so the linear DMA reads the wrong rows for any program_id > 0.
// When detected we fall back to the synchronous per-element copy, which uses
// each element's own (already-correct) pointer and is modulo-safe.
static bool defChainContainsModulo(Value v, unsigned budget = 128) {
  llvm::SmallVector<Value, 16> worklist;
  llvm::SmallPtrSet<Operation *, 32> visited;
  worklist.push_back(v);
  while (!worklist.empty() && budget-- > 0) {
    Value cur = worklist.pop_back_val();
    Operation *def = cur.getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    if (isa<arith::RemUIOp, arith::RemSIOp>(def))
      return true;
    // Stop at splat/make_range leaves; they cannot hide a modulo upstream that
    // affects the per-element row pattern.
    if (isa<triton::SplatOp, triton::MakeRangeOp>(def))
      continue;
    for (Value in : def->getOperands())
      worklist.push_back(in);
  }
  return false;
}

// Read a dense-splat integer attribute (e.g. tt.contiguity = dense<16>) off an
// op, returning its splat value or 0 when absent / non-splat.
static int64_t denseSplatAttr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<DenseIntElementsAttr>(name);
  if (!attr || !attr.isSplat())
    return 0;
  return attr.getSplatValue<APInt>().getSExtValue();
}

// Decide whether every boundary-wrap modulo (rm%M / rn%N) in the source def
// chain is a NO-OP over this tile, so the constant-stride / runtime-affine DMA
// form is exact. This is the IR-based replacement for the AxisInfo contiguity
// probe (which returns null here because the analysis is built on the original
// module before conversion). The inductor mm template annotates each
// remsi/remui it emits with tt.contiguity = dense<C> and tt.divisibility =
// dense<C>; when that C is >= the tile extent the index runs contiguously
// across the entire wrap period inside the tile, i.e. no element in the tile
// actually wraps, so the affine reconstruction reads the correct rows. For an
// UNALIGNED shape the annotated contiguity drops below the tile extent
// (gcd-based), and we refuse, keeping the modulo-safe sync copy. Returns true
// only if there is at least one modulo AND all of them are block-aligned to the
// tile.
static bool allModuloBlockAligned(Value v, ArrayRef<int64_t> tileShape,
                                  unsigned budget = 128) {
  (void)tileShape;
  llvm::SmallVector<Value, 16> worklist;
  llvm::SmallPtrSet<Operation *, 32> visited;
  worklist.push_back(v);
  bool sawModulo = false;
  while (!worklist.empty() && budget-- > 0) {
    Value cur = worklist.pop_back_val();
    Operation *def = cur.getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    if (isa<arith::RemUIOp, arith::RemSIOp>(def)) {
      sawModulo = true;
      int64_t contig = denseSplatAttr(def, "tt.contiguity");
      // The wrap is a no-op only if the result is contiguous across the full
      // extent of the dimension the modulo indexes. That dimension's extent is
      // exactly the number of elements in the (1D slice) modulo result, so the
      // self-consistent test is contig >= numElements(result). This correctly
      // accepts an aligned rm%M on a 16-row tile (contig 16 >= 16) regardless
      // of the other, larger tile dimension, and refuses an unaligned wrap
      // where the gcd-based contiguity drops below the extent.
      int64_t extent = 1;
      if (auto rt = dyn_cast<RankedTensorType>(def->getResult(0).getType()))
        extent = rt.getNumElements();
      if (contig < extent)
        return false;
      // Do not recurse past a proven-aligned modulo; its dividend's own
      // indexing is subsumed by the contiguity guarantee.
      continue;
    }
    if (isa<triton::SplatOp, triton::MakeRangeOp>(def))
      continue;
    for (Value in : def->getOperands())
      worklist.push_back(in);
  }
  return sawModulo;
}

// Function-level safety gate for the boundary-wrap (rm%M / rn%N) async path.
//
// The software pipeliner hoists the modulo into the loop PROLOGUE: the in-loop
// async-copy source is just an iter-arg pointer incremented by BK each step, so
// a def-chain walk from that op never reaches the remsi/remui. Per-op modulo
// detection (defChainContainsModulo / allModuloBlockAligned on op.getSrc())
// therefore cannot tell an ALIGNED kernel (wrap is a no-op, async is exact)
// from an UNALIGNED one (wrap is live, async silently corrupts). We
// disambiguate at FUNCTION scope: if the enclosing function contains ANY
// remsi/remui that is not proven block-aligned (its result is not annotated
// tt.contiguity >= its own extent), the kernel has a live wrap and the async
// modulo path is unsafe for EVERY copy in it. Aligned kernels carry the
// dense<extent> contiguity attr on every wrap (Triton's AxisInfo proves the
// divisibility); unaligned kernels emit the bare remsi with no such attr.
// Returns true when the function is safe.
static bool functionModuloIsSafe(Operation *op) {
  auto func = op->getParentOfType<FunctionOpInterface>();
  if (!func)
    return false;
  bool safe = true;
  func.walk([&](Operation *m) {
    if (!isa<arith::RemUIOp, arith::RemSIOp>(m))
      return;
    // Only per-element INDEX wraps threaten the affine form. A scalar modulo
    // (extent 1) is the GROUP-M program-id swizzle (pid % group_size): it picks
    // which tile a program computes, not the intra-tile element addresses, so
    // it never breaks the per-tile affine access. Skip non-tensor / 1-element
    // results.
    auto rt = dyn_cast<RankedTensorType>(m->getResult(0).getType());
    if (!rt || rt.getNumElements() <= 1)
      return;
    int64_t contig = 0;
    if (auto attr = m->getAttrOfType<DenseIntElementsAttr>("tt.contiguity"))
      if (attr.isSplat())
        contig = attr.getSplatValue<APInt>().getSExtValue();
    int64_t extent = rt.getNumElements();
    if (contig < extent)
      safe = false;
  });
  return safe;
}

// Extract all pointer components from a pointer tensor's MLIR def chain.
//
// Pattern: async_copy src = tt.addptr(broadcast(addptr(splat(base),
//                          muli(expand_dims(row_offs), splat(STRIDE)))),
//                          broadcast(col_offs))
//
// Returns true if extraction succeeded. Populates `info` with:
//   - stride: row stride scalar (in elements)
//   - basePtr: scalar base pointer
//   - rowStart: first-row index scalar (or nullptr if 0)
//   - colStart: first-col index scalar (or nullptr if 0)
// Flattened-index pattern (inductor / gemm_bench GEMM):
//   addptr(splat(base), addi(rowTerm, colTerm))
// where rowTerm = broadcast(muli(expand_dims(rowRange), splat(stride))) (row*K)
// and   colTerm = broadcast(expand_dims(colRange)) (col,1) This differs from
// the nested addptr(broadcast(addptr(splat,muli)),col) shape handled below:
// here a single addptr adds a COMBINED 2D index, so the row (strided) and col
// (unit-stride) terms are summed before the addptr.
static bool extractFlatAsyncCopyPtrInfo(triton::AddPtrOp addptrOp,
                                        AsyncCopyPtrInfo &info,
                                        bool allowModulo) {
  auto *baseSplatOp = addptrOp->getOperand(0).getDefiningOp();
  if (!baseSplatOp || !isa<triton::SplatOp>(baseSplatOp)) {
    return false;
  }
  Value combinedIdx = addptrOp->getOperand(1);
  // A modulo normally defeats constant-stride reconstruction. allowModulo is
  // set by the caller when AxisInfo proved the tile is contiguous in the inner
  // dim, i.e. the rm%M / rn%N boundary-wrap is a no-op over this access, so the
  // strided-DMA form is exact.
  if (!allowModulo && defChainContainsModulo(combinedIdx)) {
    return false;
  }
  auto *addiOp = combinedIdx.getDefiningOp();
  if (!addiOp || !isa<arith::AddIOp>(addiOp)) {
    return false;
  }

  // Identify which addi operand is the strided (row) term vs the unit (col)
  // term: the row term's def chain contains a muli(expand_dims, splat(stride)).
  auto peelBroadcast = [](Value v) -> Value {
    if (auto *bc = v.getDefiningOp())
      if (isa<triton::BroadcastOp>(bc))
        return bc->getOperand(0);
    return v;
  };
  // Recover the expand_dims axis of an index term, i.e. which logical tensor
  // dimension this index varies along. The DMA copies a row-major tile (rows
  // `stride` apart, each row `tileCols` CONTIGUOUS elements). That is only
  // valid when the strided term indexes the OUTER dim (axis-1 expand -> Nx1
  // column vector broadcast across columns) and the unit term indexes the INNER
  // dim (axis-0 expand -> 1xM broadcast across rows). A TRANSPOSED operand
  // swaps these (the strided term indexes the inner dim), which would make the
  // DMA read the tile transposed -> silent miscompile. Returns the varying
  // logical dim (0 = outer/row, 1 = inner/col) or -1 if it cannot be
  // determined.
  auto termVaryingDim = [&](Value term) -> int {
    Value inner = peelBroadcast(term);
    // Peel a muli(expand_dims, stride) to reach the expand_dims, or take the
    // term directly if it is already an expand_dims (unit-stride col term).
    if (auto *muli = inner.getDefiningOp())
      if (isa<arith::MulIOp>(muli))
        for (unsigned i = 0; i < 2; i++)
          if (auto *e = muli->getOperand(i).getDefiningOp())
            if (isa<triton::ExpandDimsOp>(e))
              inner = muli->getOperand(i);
    auto exp = inner.getDefiningOp<triton::ExpandDimsOp>();
    if (!exp)
      return -1;
    // expand_dims axis A inserts a size-1 dim at A; the original index then
    // varies along the OTHER dim. For 2D: axis 1 -> varies along dim 0 (row),
    // axis 0 -> varies along dim 1 (col).
    return exp.getAxis() == 1 ? 0 : 1;
  };
  auto matchRowTerm = [&](Value term, Value &strideOut,
                          Value &rangeOut) -> bool {
    Value inner = peelBroadcast(term);
    auto *muli = inner.getDefiningOp();
    if (!muli || !isa<arith::MulIOp>(muli))
      return false;
    for (unsigned i = 0; i < 2; i++) {
      Value opnd = muli->getOperand(i);
      auto *op = opnd.getDefiningOp();
      if (op && isa<triton::SplatOp>(op)) {
        // gemm_bench shape: stride = tt.splat(scalar).
        strideOut = op->getOperand(0);
      } else if (op && isa<triton::ExpandDimsOp>(op)) {
        rangeOut = opnd;
      } else if (auto cst = dyn_cast_or_null<arith::ConstantOp>(op)) {
        // inductor shape: stride = arith.constant dense<C>. Record the scalar
        // C; the LLVM emitter materializes it (creating an op here would leave
        // an unconvertible arith.constant post-legalization).
        if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
          if (dense.isSplat())
            info.strideConst = dense.getSplatValue<APInt>().getSExtValue();
      }
    }
    return (strideOut || info.strideConst != INT64_MIN) && rangeOut;
  };

  Value lhs = addiOp->getOperand(0), rhs = addiOp->getOperand(1);
  Value strideVal, rowRange, colTerm;
  if (matchRowTerm(lhs, strideVal, rowRange))
    colTerm = rhs;
  else if (matchRowTerm(rhs, strideVal, rowRange))
    colTerm = lhs;
  else {
    return false;
  }

  // The strided term must index the OUTER dim and the unit-stride term the
  // INNER dim. If they are swapped (transposed operand: inner dim is strided),
  // the contiguous-row DMA would read the tile transposed. Bail so the caller
  // falls back to the layout-exact sync copy.
  {
    Value stridedTerm = (colTerm == rhs) ? lhs : rhs;
    int stridedDim = termVaryingDim(stridedTerm);
    int colDim = termVaryingDim(colTerm);
    if (stridedDim != 0 || colDim != 1)
      return false;
  }

  info.basePtr = baseSplatOp->getOperand(0);
  info.stride = strideVal;
  // The row range may pass through a boundary-wrap modulo. When the wrap was
  // proven block-aligned (allowModulo), peel it so the program-id-dependent row
  // origin is preserved; otherwise the DMA reads from row 0 for every program.
  info.rowStart = extractFirstElemScalar(rowRange, allowModulo);
  if (!info.rowStart)
    extractFirstElemConst(rowRange, info.rowStartConst, allowModulo);

  if (!allowModulo && defChainContainsModulo(colTerm))
    return false;
  Value colInner = peelBroadcast(colTerm);
  // The column index may pass through a proven-no-op boundary wrap (rn % N),
  // exactly like the row index above. Peel it under allowModulo so the
  // program-id-dependent column origin (pid_n * BLOCK_N) is preserved; without
  // this the wrap defeats scalar/const extraction and colStart silently drops
  // to 0, so every N-block's DMA reads B from column 0 -> only pid_n == 0 is
  // correct and the right output half is wrong.
  info.colStart = extractFirstElemScalar(colInner, allowModulo);
  if (!info.colStart)
    extractFirstElemConst(colInner, info.colStartConst, allowModulo);
  // With a contiguity-proven no-op modulo, the tile origin is the unwrapped
  // first index; the wrap contributes nothing, so default start = 0.
  return true;
}

static bool extractAsyncCopyPtrInfo(Value ptrTensor, AsyncCopyPtrInfo &info,
                                    bool allowModulo) {
  // Walk: ptrTensor → addptr → broadcast → addptr → muli → splat(stride)
  auto *addptrOp = ptrTensor.getDefiningOp();
  if (!addptrOp || !isa<triton::AddPtrOp>(addptrOp))
    return false;

  // The first operand of the outer addptr is broadcast(inner_addptr).
  // If it is instead a splat (flattened combined-index GEMM), use the flat
  // matcher.
  Value broadcastedBase = addptrOp->getOperand(0);
  auto *broadcastOp = broadcastedBase.getDefiningOp();
  if (broadcastOp && isa<triton::SplatOp>(broadcastOp))
    return extractFlatAsyncCopyPtrInfo(cast<triton::AddPtrOp>(addptrOp), info,
                                       allowModulo);
  if (!broadcastOp || !isa<triton::BroadcastOp>(broadcastOp))
    return false;

  Value innerAddptr = broadcastOp->getOperand(0);
  auto *innerOp = innerAddptr.getDefiningOp();
  if (!innerOp || !isa<triton::AddPtrOp>(innerOp))
    return false;

  // Extract scalar base pointer from splat(base) — first operand of inner
  // addptr
  auto *baseSplatOp = innerOp->getOperand(0).getDefiningOp();
  if (!baseSplatOp || !isa<triton::SplatOp>(baseSplatOp))
    return false;
  info.basePtr = baseSplatOp->getOperand(0);

  // The second operand of innerAddptr is the row offset:
  // muli(expand_dims(row_range), splat(stride))
  Value rowOffset = innerOp->getOperand(1);
  auto *muliOp = rowOffset.getDefiningOp();
  if (!muliOp || !isa<arith::MulIOp>(muliOp))
    return false;

  // Non-affine (modulo-indexed) row offset defeats the constant-stride DMA
  // reconstruction below. Bail so the caller uses the sync per-element copy.
  if (defChainContainsModulo(rowOffset))
    return false;

  // One operand of muli is expand_dims(range), the other is splat(stride)
  Value expandDimsVal;
  bool foundStride = false;
  for (unsigned i = 0; i < 2; i++) {
    auto *op = muliOp->getOperand(i).getDefiningOp();
    if (op && isa<triton::SplatOp>(op)) {
      info.stride = op->getOperand(0);
      foundStride = true;
    } else if (op && isa<triton::ExpandDimsOp>(op)) {
      expandDimsVal = muliOp->getOperand(i);
    }
  }
  if (!foundStride)
    return false;

  // The strided (row) offset must index the OUTER dim (expand_dims axis 1 ->
  // Nx1 column vector). A transposed operand instead puts the explicit stride
  // on the INNER dim (expand_dims axis 0), and the contiguous-row DMA would
  // then read the tile transposed -> silent miscompile. Bail to the sync copy.
  if (expandDimsVal)
    if (auto exp = expandDimsVal.getDefiningOp<triton::ExpandDimsOp>())
      if (exp.getAxis() != 1)
        return false;

  // Extract first-row scalar from expand_dims(row_offs_1d)
  if (expandDimsVal) {
    info.rowStart = extractFirstElemScalar(expandDimsVal, allowModulo);
    if (!info.rowStart)
      extractFirstElemConst(expandDimsVal, info.rowStartConst, allowModulo);
  }
  // nullptr (and rowStartConst 0) means first row = 0

  // Extract col offset from outer addptr's second operand:
  // broadcast(expand_dims(col_offs_1d)) or broadcast(col_offs)
  Value colOffset = addptrOp->getOperand(1);
  // A modulo on the column index is likewise non-affine; bail to sync copy.
  if (defChainContainsModulo(colOffset))
    return false;
  auto *colBroadcastOp = colOffset.getDefiningOp();
  if (colBroadcastOp && (isa<triton::BroadcastOp>(colBroadcastOp) ||
                         isa<triton::ExpandDimsOp>(colBroadcastOp))) {
    info.colStart = extractFirstElemScalar(colBroadcastOp->getOperand(0));
    if (!info.colStart)
      extractFirstElemConst(colBroadcastOp->getOperand(0), info.colStartConst);
  }
  // nullptr (and colStartConst 0) means first col = 0

  return true;
}

// Conservative compile-time proof that the INNER (dim-1) index of a 2D source
// pointer tensor is unit-stride, i.e. adjacent columns are adjacent in memory.
// The async 2D DMA copies each tile row as `tileCols` CONTIGUOUS elements, so a
// non-unit inner stride (transposed/strided-view operand) makes it read the
// tile transposed -> silent miscompile. This walks the addptr index chain and
// returns true ONLY when it positively proves the inner dim carries no stride
// multiply; any unrecognized shape returns false (forces the exact sync copy).
//
// The offset feeding the addptr is a sum of per-dim terms; a strided dim looks
// like broadcast?(muli(expand_dims(range, axis), splat/const stride)). The
// inner dim is the one whose expand_dims axis == 0 (1xN, broadcast over rows).
// If that inner term is multiplied by anything other than 1, columns are not
// contiguous.
static bool innerDimIsUnitStride(Value ptrTensor) {
  auto *addptrOp = ptrTensor.getDefiningOp();
  if (!addptrOp || !isa<triton::AddPtrOp>(addptrOp))
    return false;

  // Collect the additive index terms. Handle both the flat shape
  // (addptr(splat(base), addi(rowTerm, colTerm))) and the nested shape
  // (addptr(broadcast(addptr(splat(base), rowTerm)), colTerm)).
  SmallVector<Value> terms;
  SmallVector<Value> work;
  work.push_back(addptrOp->getOperand(1));
  if (auto *b = addptrOp->getOperand(0).getDefiningOp())
    if (isa<triton::BroadcastOp>(b))
      work.push_back(addptrOp->getOperand(0));
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (auto *bc = v.getDefiningOp()) {
      if (isa<triton::BroadcastOp>(bc)) {
        work.push_back(bc->getOperand(0));
        continue;
      }
      if (auto add = dyn_cast<arith::AddIOp>(bc)) {
        work.push_back(add.getLhs());
        work.push_back(add.getRhs());
        continue;
      }
      if (auto inAddptr = dyn_cast<triton::AddPtrOp>(bc)) {
        // nested: the base carries another index term
        work.push_back(inAddptr->getOperand(1));
        continue;
      }
    }
    terms.push_back(v);
  }

  auto peelBroadcast = [](Value v) -> Value {
    if (auto *bc = v.getDefiningOp())
      if (isa<triton::BroadcastOp>(bc))
        return bc->getOperand(0);
    return v;
  };

  // Find the inner-dim term (expand_dims axis 0). It must NOT be wrapped in a
  // muli by a non-unit stride. We require that exactly one term is the inner
  // dim and it is a bare expand_dims (unit stride).
  bool sawInner = false;
  for (Value t : terms) {
    Value inner = peelBroadcast(t);
    // A muli(expand_dims(.. axis 0 ..), stride) on the inner dim is non-unit.
    if (auto muli = inner.getDefiningOp<arith::MulIOp>()) {
      for (unsigned i = 0; i < 2; i++)
        if (auto exp =
                muli->getOperand(i).getDefiningOp<triton::ExpandDimsOp>())
          if (exp.getAxis() == 0)
            return false; // inner dim multiplied by a stride -> not unit
      continue;
    }
    if (auto exp = inner.getDefiningOp<triton::ExpandDimsOp>())
      if (exp.getAxis() == 0)
        sawInner = true;
  }
  return sawInner;
}

struct AsyncCopyGlobalToLocalOpAppleConversion
    : public ConvertOpToLLVMPattern<ttg::AsyncCopyGlobalToLocalOp> {
  // AxisInfo lets us prove an access is affine (constant-stride, with no-op
  // modulo folded via divisibility) instead of brittle syntactic pattern walks.
  ModuleAxisInfoAnalysis *axisInfo = nullptr;

  AsyncCopyGlobalToLocalOpAppleConversion(LLVMTypeConverter &tc,
                                          ModuleAxisInfoAnalysis *ai,
                                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(tc, benefit), axisInfo(ai) {}

  // Sync fallback: per-element load from device + store to shared memory
  LogicalResult lowerSyncCopy(ttg::AsyncCopyGlobalToLocalOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llSrc = adaptor.getSrc();
    Value llDst = adaptor.getResult();
    Value llMask = adaptor.getMask();

    auto srcElems = unpackLLElements(loc, llSrc, rewriter);
    if (srcElems.empty())
      return failure();

    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> loadedVals;
    unsigned numElems = srcElems.size();
    for (unsigned i = 0; i < numElems; i++) {
      Value loaded;
      if (maskElems.empty()) {
        loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, srcElems[i]);
      } else {
        Value mask = maskElems[i];
        Value zero;
        if (elemTy.isIntOrIndex())
          zero = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                          rewriter.getIntegerAttr(elemTy, 0));
        else
          zero = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                          rewriter.getFloatAttr(elemTy, 0.0));
        Value val = LLVM::LoadOp::create(rewriter, loc, elemTy, srcElems[i]);
        loaded = LLVM::SelectOp::create(rewriter, loc, mask, val, zero);
      }
      loadedVals.push_back(loaded);
    }

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);

    auto regLayout = ttg::toLinearLayout(srcTy);
    auto sharedLayout = ttg::toLinearLayout(dstTy);
    auto cvt = regLayout.invertAndCompose(sharedLayout);

    TargetInfo targetInfo;
    lowerLocalLdSt(loc, ctx, cvt, loadedVals, elemTy, dstTy, smemObj, rewriter,
                   targetInfo);

    // The data is already in shared memory synchronously, so the matching
    // async_wait must be a no-op for this token. Return a zero-initialized
    // event slot: waiting on it does nothing. (Token type is ptr addrspace(0).)
    Value evSlot = createCompletedEventSlot(op, rewriter);
    rewriter.replaceOp(op, evSlot);
    return success();
  }

  // Check if the MLIR mask is a tt.splat of a scalar i1.
  // The pipeliner generates: %mask_scalar = arith.cmpi ... ; %mask = tt.splat
  // %mask_scalar When this pattern holds, the mask is uniform (all-true or
  // all-false), so we can gate the async DMA on the scalar boolean. Returns the
  // scalar MLIR Value, or nullptr if not a splat.
  static Value extractScalarMask(Value mask) {
    if (!mask)
      return nullptr;
    auto *defOp = mask.getDefiningOp();
    if (!defOp || !isa<triton::SplatOp>(defOp))
      return nullptr;
    Value scalar = defOp->getOperand(0);
    if (!scalar.getType().isInteger(1))
      return nullptr;
    return scalar;
  }

  // Runtime tile-origin + row-stride extraction fallback.
  //
  // When the syntactic IR walk (extractAsyncCopyPtrInfo) cannot reconstruct a
  // uniform affine tile description, we derive the SAME two scalars the async
  // 2D intrinsic needs - a uniform tile-origin device pointer and a uniform
  // src row stride (bytes) - directly from the MATERIALIZED per-element pointer
  // tensor, which always exists. This needs no IR pattern matching and is exact
  // for ANY affine access (every real GEMM/conv/inductor matmul operand).
  //
  // How it stays uniform across the simdgroup:
  //   The intrinsic is cooperative; all lanes must pass the same origin. For an
  //   affine access ptr(row,col) = base + row*rowStrideBytes + col*elemBytes,
  //   every lane computes
  //     origin = ptrtoint(srcElems[k]) - (row_k*rowStrideBytes +
  //     col_k*elemBytes)
  //   where (row_k,col_k) is the FULL tile logical coord of that lane's element
  //   k (obtained at runtime from emitIndices, which already folds in the
  //   lane/warp base). The subtraction cancels the per-lane part, so all lanes
  //   land on ptrtoint(base) - the value is uniform BY CONSTRUCTION.
  //
  // How the runtime row stride is derived:
  //   Pick two registers k0,k1 of THIS thread whose compile-time intra-thread
  //   logical offsets differ by exactly one row and zero cols. Then
  //     rowStrideBytes = ptrtoint(srcElems[k1]) - ptrtoint(srcElems[k0])
  //   is exactly one row step in bytes, computed with no pattern matching.
  //   If no such single-thread row-adjacent register pair exists (each thread
  //   owns a single tile row), we cannot derive the stride locally and bail to
  //   the sync copy. Real GEMM/conv operands always own multiple rows.
  LogicalResult
  lowerAsyncFromRuntimePtrs(ttg::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto shape = srcTy.getShape();
    if (shape.size() != 2)
      return failure();

    Value llSrc = adaptor.getSrc();
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);
    if (srcElems.empty())
      return failure();

    // Compile-time per-register intra-thread logical offsets (register order
    // matches srcElems and emitIndices).
    auto regOffsets = emitOffsetForLayout(srcTy.getEncoding(), srcTy);
    if (regOffsets.size() != srcElems.size())
      return failure();

    // Find a register pair (k0,k1) within THIS thread sharing a column but on
    // different rows. The row gap dr need not be 1 (blocked layouts hand a
    // thread rows like 0 and 8); we divide the byte delta by dr to recover one
    // row step. Pick the SMALLEST positive dr to minimize rounding exposure
    // (the access is affine so any dr is exact, but a small dr is robust).
    int kRow0 = -1, kRow1 = -1;
    int64_t rowGap = 0;
    for (unsigned a = 0; a < regOffsets.size(); a++) {
      for (unsigned b = 0; b < regOffsets.size(); b++) {
        if (a == b || regOffsets[a][1] != regOffsets[b][1])
          continue;
        int64_t dr = regOffsets[b][0] - regOffsets[a][0];
        if (dr > 0 && (kRow0 < 0 || dr < rowGap)) {
          kRow0 = a;
          kRow1 = b;
          rowGap = dr;
        }
      }
    }
    if (kRow0 < 0)
      return failure();

    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy1 = LLVMPointerType::get(ctx, 1); // device
    auto ptrTy3 = LLVMPointerType::get(ctx, 3); // threadgroup
    auto vec2i64Ty = VectorType::get({2}, i64Ty);

    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;
    int64_t tileRows = shape[0];
    int64_t tileCols = shape[1];
    int64_t tileWidthBytes = tileCols * elemBytes;

    // Runtime row stride in bytes = (ptr(k1) - ptr(k0)) / rowGap, where k0,k1
    // share a column and are rowGap rows apart.
    Value pk0 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, srcElems[kRow0]);
    Value pk1 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, srcElems[kRow1]);
    Value gapDelta = LLVM::SubOp::create(rewriter, loc, i64Ty, pk1, pk0);
    Value srcStrideBytes = gapDelta;
    if (rowGap != 1) {
      Value gapVal = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(rowGap));
      srcStrideBytes =
          LLVM::SDivOp::create(rewriter, loc, i64Ty, gapDelta, gapVal);
    }

    // Full per-element tile logical coords for THIS thread (runtime SSA).
    TargetInfo targetInfo;
    auto indices = emitIndices(loc, rewriter, targetInfo, srcTy.getEncoding(),
                               srcTy, /*withCTAOffset=*/true);
    if (indices.size() != srcElems.size() || indices[0].size() != 2) {
      return failure();
    }

    // origin = ptrtoint(srcElems[0]) - (row_0*rowStrideBytes + col_0*elemBytes)
    // Uniform across the whole simdgroup by the affine cancellation above.
    Value row0 = indices[0][0];
    Value col0 = indices[0][1];
    if (row0.getType() != i64Ty)
      row0 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, row0);
    if (col0.getType() != i64Ty)
      col0 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, col0);
    Value elemBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemBytes));
    Value rowByteOff =
        LLVM::MulOp::create(rewriter, loc, i64Ty, row0, srcStrideBytes);
    Value colByteOff =
        LLVM::MulOp::create(rewriter, loc, i64Ty, col0, elemBytesVal);
    Value intraOff =
        LLVM::AddOp::create(rewriter, loc, i64Ty, rowByteOff, colByteOff);
    Value p0 = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, srcElems[0]);
    Value originInt = LLVM::SubOp::create(rewriter, loc, i64Ty, p0, intraOff);
    Value srcBase = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy1, originInt);

    // Destination base pointer: shared memory object base.
    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    Value dstBase = smemObj.getBase();

    Value dstStrideBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));

    // (The runtime-ptr path carries only NON-cross-warp-partitioned async
    // copies: a cross-warp-consumed operand is routed to the sync copy, so no
    // per-warp band partition is applied here. The exclusive-consumption
    // partition is done in the inline emitter below.)
    int64_t bandRows = tileRows;

    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));
    Value heightVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bandRows));

    Value idx0 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(0));
    Value idx1 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    Value tileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            widthVal, idx0);
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            heightVal, idx1);

    Value zeroI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
    Value offsetVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx0);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx1);

    Value one64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));
    Value clamp = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));

    auto asyncCopyFn = getOrCreateFn(
        mod, rewriter, "air.simdgroup_async_copy_2d.p3i8.p1i8", ptrTy3,
        {i64Ty, i64Ty, ptrTy3, i64Ty, i64Ty, vec2i64Ty, ptrTy1, i64Ty, i64Ty,
         vec2i64Ty, vec2i64Ty, i32Ty});

    Value evAlloca = createEventAlloca(op, rewriter);
    // Mask is intentionally dropped: clamp_to_zero covers the boundary.
    Value event =
        LLVM::CallOp::create(rewriter, loc, asyncCopyFn,
                             ValueRange{one64, one64, dstBase, dstStrideBytes,
                                        one64, tileVec, srcBase, srcStrideBytes,
                                        one64, tileVec, offsetVec, clamp})
            .getResult();
    LLVM::StoreOp::create(rewriter, loc, event, evAlloca);

    // Token IS this copy's event slot; the matching async_wait waits on it.
    rewriter.replaceOp(op, evAlloca);
    return success();
  }

  LogicalResult
  matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llMask = adaptor.getMask();

    // Try async DMA path: requires 2D and extractable stride.
    // Mask must be absent or a uniform splat (scalar boolean).
    auto shape = srcTy.getShape();
    bool canAsyncDMA = (shape.size() == 2);
    {
      const char *e = std::getenv("TRITON_DMA_DISABLE");
      if (e && std::string(e) == "1")
        canAsyncDMA = false;
    }
    // A multi-warp-outer staged copy (warpsPerCTA[outer]>1) cannot use the
    // per-simdgroup async DMA with a single warp-uniform origin: every warp
    // would DMA the WHOLE tile into the same shared region (same-byte
    // write-write race; no fence resolves it, the discriminator is the layout
    // which only exists here). The fix is to PARTITION the tile into one
    // disjoint row band per outer warp (each warp DMAs band rows
    // [wR*bandRows, (wR+1)*bandRows) -> disjoint regions, race-free). When the
    // row dim divides evenly across the outer warps partitionFactor>0 and the
    // copy stays async (partitioned); otherwise it bails to the sync copy.
    bool outerCrossWarp = asyncCopyOuterDimCrossWarp(op);
    unsigned partitionFactor = outerCrossWarp ? asyncCopyOuterPartition(op) : 0;
    if (outerCrossWarp && partitionFactor == 0)
      canAsyncDMA = false;

    // A non-uniform per-element mask is a real boundary predicate that zeros
    // out-of-range rows/cols of a PARTIAL tile. Our async 2D copy reads the
    // full tile (clamp is not wired to the real M/N extent), so dropping such a
    // mask is only safe when the tile is PROVEN fully in-bounds (block-aligned
    // shape, where the mask is always-true). Compute that proof now; it also
    // gates the boundary-wrap modulo below.
    bool tileFullyInBounds =
        allModuloBlockAligned(op.getSrc(), shape) && functionModuloIsSafe(op);

    // Mask handling. The async DMA takes no mask or a UNIFORM scalar
    // (all-true/all-false) mask gating the whole copy. A non-uniform mask is
    // dropped ONLY when the tile is proven in-bounds; otherwise refuse async
    // and fall back to the per-element sync copy (which honors the mask
    // exactly), preventing out-of-matrix reads on unaligned shapes.
    Value mlirMaskScalar;
    Value llvmMaskScalar;
    if (canAsyncDMA && llMask) {
      mlirMaskScalar = extractScalarMask(op.getMask());
      if (mlirMaskScalar) {
        llvmMaskScalar = rewriter.getRemappedValue(mlirMaskScalar);
      } else if (!tileFullyInBounds) {
        // Non-uniform boundary mask on a possibly-partial tile: async cannot
        // honor per-element bounds, so use the sync copy.
        canAsyncDMA = false;
      }
    }

    // Use AxisInfo to decide whether a boundary-wrap modulo (rm%M / rn%N) in
    // the source index is a no-op for this tile: if the source pointer tensor
    // is fully contiguous in its inner dim, the wrap never crosses a tile edge,
    // so the constant-stride DMA form is exact. This lets inductor's triton_mm
    // template (which always emits the % wrap) take the async path.
    bool allowModulo = false;
    if (canAsyncDMA && axisInfo) {
      AxisInfo *ai = axisInfo->getAxisInfo(op.getSrc());
      if (ai) {
        int innerDim = shape.size() - 1;
        if (ai->getContiguity(innerDim) >= shape[innerDim])
          allowModulo = true;
      }
    }
    // AxisInfo is built on the pre-conversion module and frequently returns
    // null for the async-copy source here. Fall back to an IR-based proof:
    // accept the boundary-wrap modulo only when every remsi/remui in the def
    // chain is annotated contiguous across the full tile extent (no element
    // wraps inside the tile). This is exact for aligned shapes and refuses
    // unaligned ones.
    if (canAsyncDMA && !allowModulo &&
        allModuloBlockAligned(op.getSrc(), shape))
      allowModulo = true;

    // Function-level live-wrap guard. The pipeliner hides the modulo in the
    // prologue, so neither allowModulo nor the per-op def-chain walk can see an
    // UNALIGNED wrap on the in-loop copy. If the enclosing kernel contains any
    // non-block-aligned remsi/remui, the wrap is live and BOTH the
    // affine-modulo and the runtime affine paths would silently miscompile;
    // force every copy in such a kernel onto the modulo-safe sync path.
    bool funcModuloSafe = functionModuloIsSafe(op);
    if (!funcModuloSafe)
      allowModulo = false;

    AsyncCopyPtrInfo ptrInfo;
    Value llvmStride;
    if (canAsyncDMA) {
      if (extractAsyncCopyPtrInfo(op.getSrc(), ptrInfo, allowModulo)) {
        if (ptrInfo.strideConst != INT64_MIN)
          llvmStride = LLVM::ConstantOp::create(
              rewriter, loc, IntegerType::get(ctx, 32),
              rewriter.getI32IntegerAttr((int32_t)ptrInfo.strideConst));
        else
          llvmStride = rewriter.getRemappedValue(ptrInfo.stride);
        if (!llvmStride) {
          canAsyncDMA = false;
        }
      } else {
        canAsyncDMA = false;
      }
    }

    // Also need the LLVM base pointer and tile-origin offsets
    Value llvmBasePtr;
    Value llvmRowStart;
    Value llvmColStart;
    if (canAsyncDMA) {
      llvmBasePtr = rewriter.getRemappedValue(ptrInfo.basePtr);
      if (!llvmBasePtr) {
        canAsyncDMA = false;
      }
      if (ptrInfo.rowStart) {
        llvmRowStart = rewriter.getRemappedValue(ptrInfo.rowStart);
        if (!llvmRowStart) {
          canAsyncDMA = false;
        }
      }
      if (ptrInfo.colStart) {
        llvmColStart = rewriter.getRemappedValue(ptrInfo.colStart);
        if (!llvmColStart) {
          canAsyncDMA = false;
        }
      }
    }
    if (!canAsyncDMA) {
      // Affine-from-IR reconstruction failed. Try the runtime tile-origin /
      // row-stride extraction from the materialized pointer tensor, which
      // covers ANY affine access (all real GEMM/conv/inductor matmul shapes).
      //
      // Gate on a 2D tile AND a function with no LIVE wrap (funcModuloSafe).
      // The runtime origin/stride derivation assumes the access is globally
      // affine over the tile. That holds when the kernel has no modulo at all,
      // or every modulo is block-aligned (no element wraps inside any tile).
      // When a boundary wrap is LIVE (unaligned shape) the materialized
      // per-element pointers fold back at the tile edge, breaking affinity, so
      // we keep the modulo-safe sync copy. Correctness over coverage. Note the
      // pipeliner hides the wrap in the prologue, so a per-op def-chain check
      // on op.getSrc() is insufficient; only the function-level guard is sound.
      // The runtime async path reads the FULL tile with no per-element mask, so
      // it is only correct when the tile is proven fully in-bounds. A live
      // boundary (non-uniform mask that we could not drop above, i.e.
      // canAsyncDMA was cleared) or an unaligned shape must NOT take it, else a
      // partial edge tile reads past the matrix and corrupts the result.
      bool noLiveBoundary = !llMask || mlirMaskScalar || tileFullyInBounds;
      // The runtime DMA assumes contiguous columns (inner unit stride). For a
      // transposed/strided-view operand the inner dim is strided, so the
      // contiguous-row copy reads the tile transposed. Require a positive
      // unit-inner-stride proof; otherwise use the layout-exact sync copy.
      bool innerUnit = innerDimIsUnitStride(op.getSrc());
      // A multi-warp-outer copy on the runtime-ptr path is safe only when it is
      // partitioned (partitionFactor>0): each warp DMAs a disjoint row band.
      // The runtime-ptr path is NOT given the partition. It carries operands
      // whose partitioned (outer) dim is the CONTRACTION (K) dim, read IN FULL
      // by every warp. Partitioning into per-warp K-bands then needs a sibling
      // warp to see another warp's async-DMA band: air.wait_simdgroup_events
      // drains only the issuing simdgroup, and even a device-memory fence at
      // the barrier only REDUCES (not eliminates, ~7% residual at 100x) the
      // cross-simdgroup DMA visibility race. So a cross-warp-consumed operand
      // stays on the layout-exact sync copy. Only the inline emitter
      // partitions, and only the operand whose outer dim is consumed
      // exclusively per-warp (the M dim, where warp w reads only its own band).
      bool runtimeSafe = (shape.size() == 2) && funcModuloSafe &&
                         noLiveBoundary && innerUnit && !outerCrossWarp;
      {
        const char *e = std::getenv("TRITON_DMA_DISABLE");
        if (e && std::string(e) == "1")
          runtimeSafe = false;
      }
      if (runtimeSafe) {
        if (succeeded(lowerAsyncFromRuntimePtrs(op, adaptor, rewriter)))
          return success();
      }
      return lowerSyncCopy(op, adaptor, rewriter);
    }

    // ── Async DMA path via air.simdgroup_async_copy_2d ──

    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy1 = LLVMPointerType::get(ctx, 1); // device
    auto ptrTy3 = LLVMPointerType::get(ctx, 3); // threadgroup
    auto vec2i64Ty = VectorType::get({2}, i64Ty);

    // Element size in bytes
    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;

    // Tile geometry
    int64_t tileRows = shape[0];
    int64_t tileCols = shape[1];
    int64_t tileWidthBytes = tileCols * elemBytes;

    // Source stride in bytes: llvmStride (elements) * elemBytes
    // llvmStride may be i32 — extend to i64
    Value strideI64 = llvmStride;
    if (llvmStride.getType() != i64Ty) {
      strideI64 = LLVM::SExtOp::create(rewriter, loc, i64Ty, llvmStride);
    }
    Value elemBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemBytes));
    Value srcStrideBytes =
        LLVM::MulOp::create(rewriter, loc, i64Ty, strideI64, elemBytesVal);

    // Destination stride in bytes (TG is packed, no padding)
    Value dstStrideBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));

    // Source base pointer: compute UNIFORM tile origin from extracted scalars.
    //
    // air.simdgroup_async_copy_2d is a simdgroup-cooperative operation — all
    // threads in the simdgroup must pass the SAME base pointer (the tile's
    // top-left corner). We cannot use per-thread pointers from srcElems[0].
    //
    // tile_origin = basePtr + rowStart * stride + colStart
    // where basePtr, rowStart, colStart, stride are all scalar (uniform).
    Value srcBase = llvmBasePtr;
    if (llvmRowStart) {
      // GEP by rowStart * stride (in elements)
      Value rowOff = LLVM::MulOp::create(rewriter, loc, llvmRowStart.getType(),
                                         llvmRowStart, llvmStride);
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{rowOff});
    } else if (ptrInfo.rowStartConst != 0) {
      // Folded-constant first row: GEP by (rowStartConst * stride) elements.
      Value rc = LLVM::ConstantOp::create(
          rewriter, loc, llvmStride.getType(),
          rewriter.getIntegerAttr(llvmStride.getType(), ptrInfo.rowStartConst));
      Value rowOff = LLVM::MulOp::create(rewriter, loc, llvmStride.getType(),
                                         rc, llvmStride);
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{rowOff});
    }
    if (llvmColStart) {
      // GEP by colStart (in elements)
      srcBase =
          LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy, srcBase,
                              ArrayRef<LLVM::GEPArg>{llvmColStart});
    } else if (ptrInfo.colStartConst != 0) {
      // Folded-constant first col (the prefetched K-block offset): GEP by
      // colStartConst elements. THIS is the num_stages>=3 correctness fix.
      Value cc = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)ptrInfo.colStartConst));
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{cc});
    }

    // Destination base pointer: shared memory object base
    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    Value dstBase = smemObj.getBase();

    // PARTITIONED multi-warp copy: split the tile's row
    // (outer) dim into partitionFactor disjoint horizontal bands, one per outer
    // warp. Warp wR copies rows [wR*bandRows, (wR+1)*bandRows) from a per-warp
    // source/dest origin -> warps write DISJOINT regions, so the per-simdgroup
    // wait correctly drains each warp's own copy and there is no write-write
    // race. bandRows = tileRows / partitionFactor (exact by asyncCopyOuter-
    // Partition). srcBase/dstBase are typed elemTy* pointers, so offset in
    // ELEMENTS: src by bandStart*stride, dst by bandStart*tileCols (TG packed).
    int64_t bandRows = tileRows;
    if (partitionFactor > 1) {
      bandRows = tileRows / partitionFactor;
      Value wR = emitOuterWarpIndex(op, rewriter, loc);
      Value bandRowsV = LLVM::ConstantOp::create(
          rewriter, loc, wR.getType(),
          rewriter.getIntegerAttr(wR.getType(), bandRows));
      Value bandStart = LLVM::MulOp::create(rewriter, loc, wR, bandRowsV);
      // src += bandStart * stride (elements).
      Value srcRowOff = LLVM::MulOp::create(rewriter, loc, bandStart.getType(),
                                            bandStart, llvmStride);
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{srcRowOff});
      // dst += bandStart * tileCols (elements; TG rows are packed, width=cols).
      Value tileColsV = LLVM::ConstantOp::create(
          rewriter, loc, bandStart.getType(),
          rewriter.getIntegerAttr(bandStart.getType(), tileCols));
      Value dstRowOff = LLVM::MulOp::create(rewriter, loc, bandStart.getType(),
                                            bandStart, tileColsV);
      dstBase = LLVM::GEPOp::create(rewriter, loc, dstBase.getType(), elemTy,
                                    dstBase, ArrayRef<LLVM::GEPArg>{dstRowOff});
    }

    // Build tile size vectors: <width_bytes, height_rows>. height = bandRows so
    // each warp copies only its own band (== tileRows when not partitioned).
    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));
    Value heightVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bandRows));

    // Source tile = <tileWidthBytes, tileRows>
    Value tileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    Value idx0 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(0));
    Value idx1 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            widthVal, idx0);
    tileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty, tileVec,
                                            heightVal, idx1);

    // Offset = <0, 0>
    Value zeroI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(0));
    Value offsetVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx0);
    offsetVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                              offsetVec, zeroI64, idx1);

    // sizeof=1, alignof=1 (byte-granularity copy)
    Value one64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));

    // clamp = 0
    Value clamp = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));

    // Declare air.simdgroup_async_copy_2d.p3i8.p1i8
    // Returns ptr addrspace(3) (event pointer)
    auto asyncCopyFn = getOrCreateFn(
        mod, rewriter, "air.simdgroup_async_copy_2d.p3i8.p1i8", ptrTy3,
        {i64Ty, i64Ty, // sizeof, alignof
         ptrTy3, i64Ty, i64Ty,
         vec2i64Ty, // dst, dstStride, dstElemStride, dstTile
         ptrTy1, i64Ty, i64Ty,
         vec2i64Ty,          // src, srcStride, srcElemStride, srcTile
         vec2i64Ty, i32Ty}); // offset, clamp

    // One event alloca per copy so concurrent A/B copies do not clobber a
    // shared slot (see createEventAlloca).
    Value evAlloca = createEventAlloca(op, rewriter);

    if (llvmMaskScalar) {
      // Masked path: gate the async DMA on the scalar boolean.
      // When mask is false (boundary), skip the DMA entirely.
      // Use if/then/else: if (mask) { async_copy; store event } else { skip }
      auto *parentBlock = rewriter.getInsertionBlock();
      auto insertPt = rewriter.getInsertionPoint();

      auto *thenBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
      auto *afterBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(thenBlock)));

      // Move everything after the current insertion point to afterBlock
      afterBlock->getOperations().splice(afterBlock->begin(),
                                         parentBlock->getOperations(), insertPt,
                                         parentBlock->end());

      // Conditional branch: if mask_scalar → thenBlock, else → afterBlock
      rewriter.setInsertionPointToEnd(parentBlock);
      LLVM::CondBrOp::create(rewriter, loc, llvmMaskScalar, thenBlock,
                             afterBlock);

      // thenBlock: emit async copy + store event + branch to afterBlock
      rewriter.setInsertionPointToStart(thenBlock);
      Value event = LLVM::CallOp::create(
                        rewriter, loc, asyncCopyFn,
                        ValueRange{one64, one64, dstBase, dstStrideBytes, one64,
                                   tileVec, srcBase, srcStrideBytes, one64,
                                   tileVec, offsetVec, clamp})
                        .getResult();
      LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);

      // Continue in afterBlock. The else (mask false) branch falls through
      // without storing, leaving the entry-block zero-init event in the slot;
      // air.wait_simdgroup_events on it is a no-op (the masked tile is a
      // boundary the consumer does not read).
      rewriter.setInsertionPointToStart(afterBlock);
    } else {
      // Unmasked: always emit async DMA
      Value event = LLVM::CallOp::create(
                        rewriter, loc, asyncCopyFn,
                        ValueRange{one64, one64, dstBase, dstStrideBytes, one64,
                                   tileVec, srcBase, srcStrideBytes, one64,
                                   tileVec, offsetVec, clamp})
                        .getResult();
      LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
    }

    // The token IS this copy's event slot. async_wait waits on exactly the
    // slots carried by its operand tokens; because the token is a scf.for
    // iter_arg the loop-carried value selects the correct alternating buffer
    // each iteration. (Token type converts to ptr addrspace(0); see the
    // AsyncToken type conversion registered on the converter.)
    rewriter.replaceOp(op, evAlloca);
    return success();
  }
};

struct AsyncCommitGroupOpAppleConversion
    : public ConvertOpToLLVMPattern<ttg::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttg::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Forward the copy's event slot through the group token. Metal events are
    // per-copy, so a "group" is just its member copies' slots threaded onward;
    // the pipeliner emits exactly one copy per commit_group here. With no input
    // token (empty group) return a zero-initialized completed slot so a later
    // wait on it is a no-op.
    auto inTokens = adaptor.getInputTokens();
    if (!inTokens.empty()) {
      rewriter.replaceOp(op, inTokens.front());
      return success();
    }
    Value evSlot = createCompletedEventSlot(op, rewriter);
    rewriter.replaceOp(op, evSlot);
    return success();
  }
};

struct AsyncWaitOpAppleConversion
    : public ConvertOpToLLVMPattern<ttg::AsyncWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttg::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto mod = op->getParentOfType<ModuleOp>();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVMVoidType::get(ctx);
    auto ptrTy0 = LLVMPointerType::get(ctx, 0);

    // Check if any async copy was emitted by looking for the
    // air.simdgroup_async_copy_2d declaration in the module.
    bool hasAsyncDMA = mod.lookupSymbol<LLVMFuncOp>(
                           "air.simdgroup_async_copy_2d.p3i8.p1i8") != nullptr;

    if (hasAsyncDMA) {
      // Wait on EXACTLY the copies whose tokens this wait consumes. Each token
      // (adaptor operand) is that copy's event slot (ptr addrspace(0)); see the
      // AsyncToken type conversion and the async_copy lowering. The token is a
      // scf.for iter_arg, so the loop-carried value selects the correct
      // alternating (double/triple-buffered) buffer each iteration: this is the
      // num_stages>=3 correctness fix. A separate wait_simdgroup_events(1,
      // slot) per token keeps the slot scalar (Metal v1 bitcode handles arrays
      // of typed pointers poorly). Each slot is zero-initialized in the entry
      // block, so a token from a sync/skip path holds a complete event and the
      // wait is a real no-op, never a read of an uninitialized pointer.
      auto waitFn = getOrCreateFn(mod, rewriter, "air.wait_simdgroup_events",
                                  voidTy, {i32Ty, ptrTy0});
      Value oneI32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                              rewriter.getI32IntegerAttr(1));
      for (Value evSlot : adaptor.getAsyncToken())
        LLVM::CallOp::create(rewriter, loc, waitFn, ValueRange{oneI32, evSlot});
    }

    // Always emit TG barrier (needed for both sync and async paths
    // to ensure shared memory visibility across all threads).
    // NOTE: flag 2 (threadgroup fence) suffices here because the only async
    // copies that reach a consumer are single-simdgroup or per-warp-EXCLUSIVE
    // partitioned (warp w reads only its own band). A device fence (flag 3)
    // would be required for a CROSS-warp-consumed async band, but those are
    // routed to the sync copy instead (see runtimeSafe), since even flag 3 only
    // reduces (~7% residual) the cross-simdgroup DMA visibility race.
    auto barrFn =
        getOrCreateFn(mod, rewriter, "air.wg.barrier", voidTy, {i32Ty, i32Ty});
    Value barrFlag = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                              rewriter.getI32IntegerAttr(2));
    Value barrScope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrFn,
                         ValueRange{barrFlag, barrScope});

    // The wait's result token (retToken) is consumed; after the wait all its
    // input copies are complete, so return a zero-initialized completed event
    // slot. Waiting on it later is a no-op.
    Value zeroToken = createCompletedEventSlot(op, rewriter);
    rewriter.replaceOp(op, zeroToken);
    return success();
  }
};

// ttg.barrier → air.wg.barrier with proper addrSpace→flag mapping.
struct AppleBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op,
                  triton::gpu::BarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto mod = op->getParentOfType<ModuleOp>();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVMVoidType::get(ctx);
    auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);

    LLVMFuncOp barrFn;
    if (auto existing = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
      barrFn = existing;
    else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      barrFn = LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier",
                                  barrFnTy, Linkage::External);
    }

    // AIR flag 1 = device, flag 2 = TG, flag 3 = both
    auto addrSpace = op.getAddrSpace();
    bool needsDevice =
        static_cast<uint32_t>(addrSpace) &
        (static_cast<uint32_t>(triton::gpu::AddrSpace::GlobalRead) |
         static_cast<uint32_t>(triton::gpu::AddrSpace::GlobalWrite));
    bool needsTG = static_cast<uint32_t>(addrSpace) &
                   static_cast<uint32_t>(triton::gpu::AddrSpace::Local);
    int flag = 0;
    if (needsDevice)
      flag |= 1;
    if (needsTG)
      flag |= 2;
    if (flag == 0)
      flag = 2;

    Value flags = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(flag));
    Value scope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flags, scope});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTritonAppleGPUToLLVMPass
    : public PassWrapper<ConvertTritonAppleGPUToLLVMPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonAppleGPUToLLVMPass)

  StringRef getArgument() const override {
    return "convert-triton-apple-gpu-to-llvm";
  }
  StringRef getDescription() const override {
    return "Lower TritonGPU ops (Apple MPS) to LLVM IR";
  }

  void runOnOperation() override {
    auto mod = getOperation();
    auto *ctx = &getContext();

    TargetInfo targetInfo;
    TritonGPUToLLVMTypeConverter typeConverter(ctx, targetInfo);

    // Thread the async-DMA completion event through the !ttg.async.token SSA
    // value instead of throwing it away as i32 0. The Triton software
    // pipeliner double/triple-buffers the K-loop and carries "which buffer's
    // copy to wait on" through a token that is a scf.for iter_arg, so it
    // alternates buffers each iteration. Mapping the token to the event-slot
    // pointer (ptr addrspace(0), the thread-local alloca that holds the
    // simdgroup event handle) lets air.wait_simdgroup_events wait on exactly
    // the copy whose token the wait consumes; the loop-carried iter_arg then
    // selects the correct alternating buffer. Registered last so it overrides
    // the upstream i32 mapping (TypeConverter tries conversions newest-first).
    typeConverter.addConversion(
        [ctx](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
          return LLVM::LLVMPointerType::get(ctx, 0);
        });

    // Membar analysis: insert barriers between conflicting TG memory accesses.
    {
      ModuleAllocation allocation(mod);
      ModuleMembarAnalysis membarAnalysis(&allocation);
      membarAnalysis.run();
    }

    {
      int64_t smemSize = 0;
      if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
        smemSize = attr.getValue().getZExtValue();
      if (smemSize == 0)
        smemSize = 8;
      int64_t smemAligned = (smemSize + 15) & ~15;
      mod->setAttr("ttg.shared",
                   IntegerAttr::get(IntegerType::get(ctx, 64), smemAligned));
      {
        OpBuilder b(mod.getBodyRegion());
        auto loc = mod.getLoc();
        auto elemTy = typeConverter.convertType(b.getIntegerType(8));
        auto arrayTy = LLVM::LLVMArrayType::get(elemTy, smemSize);
        LLVM::GlobalOp::create(b, loc, arrayTy, /*isConstant=*/false,
                               LLVM::Linkage::Internal, "global_smem",
                               /*value=*/Attribute(), /*alignment=*/16,
                               /*addrSpace=*/3u);
      }
    }

    // Pre-compute MMA threadgroup memory usage from tt.dot ops.
    // Each dot creates a __tg_dot_ab TG buffer with potential bank-conflict
    // padding (TG_PAD extra elements per row). Must account for the padded
    // size so ConvertLayoutOp can correctly budget the 32KB TG limit.
    // The IR pipeline coalesces all dot TG globals into one (taking the max),
    // so total MMA TG cost = max over all dots.
    // Set as module attribute so ConvertLayoutOp can account for it in
    // the 32KB TG budget when sizing its own TG buffers.
    {
      int64_t maxMmaBytes = 0;
      mod.walk([&](mlir::triton::DotOp dot) {
        auto aType = cast<RankedTensorType>(dot.getA().getType());
        auto cType = cast<RankedTensorType>(dot.getC().getType());
        unsigned dotRank = cType.getRank();
        int64_t K = aType.getShape()[dotRank - 1];
        int64_t N = cType.getShape()[dotRank - 1];
        int64_t maxStride = std::max(K, N);
        // Match the padding logic in DotOpToLLVM: pad when stride % 8 == 0
        // and padded buffer fits in 16KB budget.
        int64_t paddedMaxStride = maxStride;
        if (maxStride % 8 == 0) {
          int64_t candidateStride = maxStride + 4; // TG_PAD = 4
          if ((8 * candidateStride + 1) * 4 <= 16384)
            paddedMaxStride = candidateStride;
        }
        int64_t tgFloats = 8 * paddedMaxStride + 1;
        int64_t tgBytes = tgFloats * 4; // f32
        maxMmaBytes = std::max(maxMmaBytes, tgBytes);
      });
      if (maxMmaBytes > 0)
        mod->setAttr("ttg.mma_shared",
                     IntegerAttr::get(IntegerType::get(ctx, 64), maxMmaBytes));
    }

    RewritePatternSet patterns(ctx);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Apple func lowering: kernel args → addrspace(2)* + load, device fns
    // direct. Higher priority than shared FuncOpConversion (which is
    // NVIDIA-specific).
    patterns.add<AppleFuncOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));
    // Apple call lowering: no shared memory stack pointer appending.
    patterns.add<AppleCallOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));
    patterns.add<AppleReturnOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 20));

    // Shared Triton → LLVM patterns (handles device functions, non-kernel)
    mlir::triton::populateSPMDOpToLLVMPattern(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, patternBenefitDefault);
    mlir::triton::populateMakeRangeOpToLLVMPattern(
        typeConverter, targetInfo, patterns, patternBenefitDefault);
    mlir::triton::populateControlFlowOpToLLVMPattern(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, patternBenefitDefault);
    mlir::triton::populateReduceOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);
    mlir::triton::populateScanOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);

    mlir::triton::populateGatherOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);

    mlir::triton::populateHistogramOpToLLVMPatterns(
        typeConverter, patterns, targetInfo, patternBenefitDefault);

    populateDotOpToLLVMPatterns(typeConverter, patterns, patternBenefitDefault);
    populateLoadStoreToLLVMPatterns(typeConverter, patterns,
                                    patternBenefitDefault);

    patterns.add<AppleBarrierOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));

    patterns.add<SafeStoreOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<SafeLoadOpConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));

    patterns.add<WarpIdOpConversion>(typeConverter, patternBenefitDefault);
    patterns.add<ApplePrintOpConversion>(typeConverter,
                                         patternBenefitDefault + 10);
    patterns.add<AppleAssertOpConversion>(typeConverter,
                                          patternBenefitDefault + 10);
    patterns.add<GetNumProgramsOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AtomicRMWOpAppleConversion>(typeConverter,
                                             patternBenefitDefault + 10);
    patterns.add<AtomicCASOpAppleConversion>(typeConverter,
                                             patternBenefitDefault + 10);
    patterns.add<ConvertLayoutOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AsyncCopyGlobalToLocalOpAppleConversion>(
        typeConverter, &axisInfoAnalysis,
        PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AsyncCommitGroupOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));
    patterns.add<AsyncWaitOpAppleConversion>(
        typeConverter, PatternBenefit(patternBenefitDefault + 10));

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    // Triton elementwise + view patterns override arith scalar patterns
    // for tensor types (higher benefit wins for same op).
    mlir::triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo,
        patternBenefitDefault + 1);
    mlir::triton::populateClampFOpToLLVMPattern(typeConverter, patterns,
                                                axisInfoAnalysis, targetInfo,
                                                patternBenefitDefault + 1);
#define POPULATE_FLOAT_OP(SRC_OP, DST_OP)                                      \
  patterns.add<mlir::triton::gpu::ElementwiseOpConversion<SRC_OP, DST_OP>>(    \
      typeConverter, axisInfoAnalysis, patternBenefitDefault + 1)
    POPULATE_FLOAT_OP(arith::AddFOp, LLVM::FAddOp);
    POPULATE_FLOAT_OP(arith::SubFOp, LLVM::FSubOp);
    POPULATE_FLOAT_OP(arith::MulFOp, LLVM::FMulOp);
    POPULATE_FLOAT_OP(arith::DivFOp, LLVM::FDivOp);
    POPULATE_FLOAT_OP(triton::PreciseDivFOp, LLVM::FDivOp);
    POPULATE_FLOAT_OP(triton::PreciseSqrtOp, LLVM::SqrtOp);
    POPULATE_FLOAT_OP(arith::ExtFOp, LLVM::FPExtOp);
    POPULATE_FLOAT_OP(arith::TruncFOp, LLVM::FPTruncOp);
    POPULATE_FLOAT_OP(arith::SIToFPOp, LLVM::SIToFPOp);
    POPULATE_FLOAT_OP(arith::FPToSIOp, LLVM::FPToSIOp);
#undef POPULATE_FLOAT_OP

    // ExternElementwiseOp: lower libdevice calls to LLVM intrinsics.
    // Inductor emits libdevice.exp, libdevice.sin, etc. which on CUDA
    // link to __nv_* functions. On MPS, map to llvm.* intrinsics.
    patterns.add<ExternElementwiseOpAppleConversion>(
        typeConverter, axisInfoAnalysis, patternBenefitDefault + 10);

    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               patternBenefitDefault + 1);
    // Expand math::ErfOp to polynomial approximation before MathToLLVM
    // (there is no llvm.erf intrinsic — NVIDIA uses libdevice, we expand
    // inline)
    mlir::populatePolynomialApproximateErfPattern(patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    // Conversion target: everything must lower to LLVM dialect
    ConversionTarget target(*ctx);
    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();
    target.addIllegalDialect<applegpu::TritonAppleGPUDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    // gpu.thread_id is emitted by shared make_range/SPMD patterns;
    // it will be lowered to air intrinsics by a subsequent pass.
    target.addLegalOp<mlir::gpu::ThreadIdOp>();
    target.addLegalOp<mlir::gpu::BlockDimOp>();
    target.addLegalOp<mlir::gpu::BarrierOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();

    // Fix up llvm.loop_annotation on llvm.br / llvm.cond_br ops.
    //
    // The ControlFlowToLLVM BranchOpLowering copies cf.br attrs via
    // setAttrs(getAttrDictionary()), but the loop_annotation attr name
    // in the CF dict is "llvm.loop_annotation" (discardable, dialect-
    // prefixed), while the LLVM BrOp's inherent property is named
    // "loop_annotation" (no prefix). setAttrs doesn't match them, so
    // the attr stays discardable and getLoopAnnotationAttr() returns
    // null, causing translateModuleToLLVMIR to drop the !llvm.loop
    // metadata.
    //
    // Walk all branch ops and move the discardable attr to the proper
    // inherent property.
    mod.walk([](LLVM::BrOp brOp) {
      if (auto attr = brOp->getAttrOfType<LLVM::LoopAnnotationAttr>(
              "llvm.loop_annotation")) {
        brOp.setLoopAnnotationAttr(attr);
        brOp->removeDiscardableAttr("llvm.loop_annotation");
      }
    });
    mod.walk([](LLVM::CondBrOp brOp) {
      if (auto attr = brOp->getAttrOfType<LLVM::LoopAnnotationAttr>(
              "llvm.loop_annotation")) {
        brOp.setLoopAnnotationAttr(attr);
        brOp->removeDiscardableAttr("llvm.loop_annotation");
      }
    });
  }
};

// ── LowerGPUToAirPass ─────────────────────────────────────────────────────
//
// Converts remaining gpu.thread_id / gpu.block_dim ops (emitted by shared
// Triton patterns like make_range / SPMD) to air intrinsics / constants so
// the MLIR module is pure LLVM dialect before llvm::toModule().
//
//   gpu.thread_id x  →  call @air.dispatch_thread_id[0]() : i32, index_cast
//   gpu.thread_id y/z → arith.constant 0 : index
//   gpu.block_dim x  →  arith.constant <numThreads> : index   (from module
//   attr) gpu.block_dim y/z → arith.constant 1 : index
//
struct LowerGPUToAirPass
    : public PassWrapper<LowerGPUToAirPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUToAirPass)

  StringRef getArgument() const override { return "lower-gpu-to-air"; }
  StringRef getDescription() const override {
    return "Lower gpu.thread_id / gpu.block_dim to air intrinsics / constants";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *ctx = &getContext();
    auto i32Ty = IntegerType::get(ctx, 32);

    // Declare air.thread_position_in_threadgroup once at module start.
    // Returns [3 x i32]; we extractvalue index 0 for the flat thread ID.
    // _add_air_metadata() rewrites this call+extractvalue pattern to an arg.
    auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
    auto tidFnName = StringRef("air.thread_position_in_threadgroup");
    auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
    if (!mod.lookupSymbol<LLVMFuncOp>(tidFnName)) {
      OpBuilder b(mod.getBodyRegion());
      b.setInsertionPointToStart(mod.getBody());
      LLVMFuncOp::create(b, mod.getLoc(), tidFnName, tidFnTy,
                         Linkage::External);
    }
    auto tidFn = mod.lookupSymbol<LLVMFuncOp>(tidFnName);

    // Read total thread count from module attributes for gpu.block_dim.
    int64_t threadsPerWarp = 32;
    int64_t numWarps = 4;
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
      threadsPerWarp = a.getInt();
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      numWarps = a.getInt();
    int64_t totalThreads = threadsPerWarp * numWarps;

    IRRewriter rewriter(ctx);

    // Walk and replace gpu.thread_id / gpu.block_dim
    // gpu.thread_id/block_dim return `index` type. Downstream users (e.g.
    // make_range) have already been lowered to LLVM i64/i32 ops by this
    // point. We need to produce a value of the same `index` type and let
    // the existing index-to-llvm lowering handle it — but that already
    // ran. So we emit LLVM ops directly:
    //   gpu.thread_id x → llvm.call @air.dispatch_thread_id[0]() → i32
    //                   → llvm.zext i32 → i64  (index = i64 in LLVM)
    //   gpu.thread_id y/z → llvm.mlir.constant(0 : i64)
    //   gpu.block_dim x  → llvm.mlir.constant(totalThreads : i64)
    //   gpu.block_dim y/z → llvm.mlir.constant(1 : i64)
    //
    // The `index` type maps to i64 in the LLVM type system (index-bitwidth=0
    // means native pointer width = 64-bit on Apple Silicon).
    auto i64Ty = IntegerType::get(ctx, 64);

    mod.walk([&](Operation *op) {
      rewriter.setInsertionPoint(op);
      auto loc = op->getLoc();

      if (auto tidOp = dyn_cast<mlir::gpu::ThreadIdOp>(op)) {
        Value replacement;
        if (tidOp.getDimension() == mlir::gpu::Dimension::x) {
          Value tidStruct =
              LLVM::CallOp::create(rewriter, loc, tidFn, ValueRange{})
                  .getResult();
          Value i32val = LLVM::ExtractValueOp::create(
              rewriter, loc, i32Ty, tidStruct, ArrayRef<int64_t>{0});
          // Extend i32 → i64 to match `index` type (pointer width on Apple
          // Silicon). Use SExt to produce a single instruction without an
          // intermediate SSA: wrap in a struct-free zext inline by using the
          // i64 directly. Actually: emit only ExtractValue (i32) then trunc or
          // zext as needed. The users of gpu.thread_id have already been
          // lowered to expect i64 (via index_to_llvm). Emit zext to match. The
          // extra SSA is OK since _add_air_metadata's renumbering now correctly
          // handles it.
          replacement = LLVM::ZExtOp::create(rewriter, loc, i64Ty, i32val);
        } else {
          replacement = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                 rewriter.getI64IntegerAttr(0));
        }
        rewriter.replaceOp(op, replacement);
        return;
      }

      if (auto bdOp = dyn_cast<mlir::gpu::BlockDimOp>(op)) {
        int64_t val =
            (bdOp.getDimension() == mlir::gpu::Dimension::x) ? totalThreads : 1;
        Value replacement = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(val));
        rewriter.replaceOp(op, replacement);
        return;
      }

      if (isa<mlir::gpu::BarrierOp>(op)) {
        auto voidTy = LLVMVoidType::get(ctx);
        auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
        LLVMFuncOp barrFn;
        if (auto existing = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
          barrFn = existing;
        else {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(mod.getBody());
          barrFn = LLVMFuncOp::create(rewriter, mod.getLoc(), "air.wg.barrier",
                                      barrFnTy, Linkage::External);
        }
        Value flags = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(2));
        Value scope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(1));
        LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flags, scope});
        rewriter.eraseOp(op);
        return;
      }
    });
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createConvertTritonAppleGPUToLLVMPass() {
  return std::make_unique<ConvertTritonAppleGPUToLLVMPass>();
}

std::unique_ptr<mlir::Pass> createLowerGPUToAirPass() {
  return std::make_unique<LowerGPUToAirPass>();
}

void registerTritonAppleGPUToLLVMPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConvertTritonAppleGPUToLLVMPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<LowerGPUToAirPass>();
  });
}

} // namespace mlir::triton::applegpu
