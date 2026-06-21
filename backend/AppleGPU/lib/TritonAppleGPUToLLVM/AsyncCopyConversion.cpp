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

// ── Pipeliner async copy lowering ────────────────────────────────────────
// Triton's pipeliner emits async_copy/commit/wait for multi-buffered
// load-compute overlap. When possible we emit true async DMA via
// air.simdgroup_async_copy_2d (needs an extractable row stride, no mask, 2D
// tile); otherwise we fall back to a sync per-element copy via lowerLocalLdSt.

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

// Fresh single-event alloca in the function entry block, one per async copy,
// threaded out as the op's !ttg.async.token so async_wait waits on exactly this
// copy. LANDMINE: air.wait_simdgroup_events wants a thread-local (addrspace 0)
// pointer; a TG (as3) global crashes the GPU compiler. Slots stay scalar (Metal
// v1 bitcode mishandles arrays of typed pointers). Zero-initialized so a token
// reaching a wait with no preceding store holds a complete event (no-op).
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

// air.simdgroup_async_copy_2d is a simdgroup-cooperative DMA. When
// warpsPerCTA[outerDim] > 1 every warp consumes the staged buffer and must
// issue and wait its OWN event. LANDMINE: a warp-0-only form races - the
// post-wait TG barrier fences regular TG stores, not the DMA engine, so a
// sibling can read before warp 0's copy lands (nondeterministic corruption).
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

// Predicate selecting the first simdgroup (tid < 32). Used by the affine
// cross-warp copy, which fires its single large-tile DMA from one warp.
static Value emitWarp0Pred(ttg::AsyncCopyGlobalToLocalOp op,
                           ConversionPatternRewriter &rewriter, Location loc) {
  auto *ctx = op.getContext();
  auto mod = op->getParentOfType<ModuleOp>();
  auto i32Ty = IntegerType::get(ctx, 32);
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
  return LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult, tid,
                              c32);
}

// Runtime simdgroup (warp) index within the threadgroup: tid.x / 32.
static Value emitWarpId(Operation *op, ConversionPatternRewriter &rewriter,
                        Location loc) {
  auto *ctx = op->getContext();
  auto mod = op->getParentOfType<ModuleOp>();
  auto i32Ty = IntegerType::get(ctx, 32);
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
  return LLVM::UDivOp::create(rewriter, loc, i32Ty, tid, c32);
}

// Total simdgroups in the threadgroup = product(warpsPerCTA). The cross-warp
// async copy partitions the row dimension across this many warps.
static int64_t totalWarps(ttg::AsyncCopyGlobalToLocalOp op) {
  auto enc = op.getSrc().getType().getEncoding();
  auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(enc);
  if (!blocked)
    return 1;
  int64_t n = 1;
  for (auto w : blocked.getWarpsPerCTA())
    n *= w;
  return n;
}

// A modulo (inductor's `(pid*BLOCK+arange) % M` bounds-wrap) makes the per-row
// stride non-constant, so the affine DMA reconstruction below is invalid and
// the caller must fall back to the modulo-safe sync copy.
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

// IR-based proof (AxisInfo is null pre-conversion) that every boundary-wrap
// modulo is a no-op over this tile, so the affine DMA form is exact. Inductor
// annotates each rem with tt.contiguity = dense<C>; C >= tile extent means no
// element wraps inside the tile. Returns true only if >=1 modulo AND all
// block-aligned.
static bool allModuloBlockAligned(Value v, unsigned budget = 128) {
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
      // No-op iff contiguous across the full extent of the indexed dim, i.e.
      // contig >= numElements(modulo result).
      int64_t extent = 1;
      if (auto rt = dyn_cast<RankedTensorType>(def->getResult(0).getType()))
        extent = rt.getNumElements();
      if (contig < extent)
        return false;
      continue;
    }
    if (isa<triton::SplatOp, triton::MakeRangeOp>(def))
      continue;
    for (Value in : def->getOperands())
      worklist.push_back(in);
  }
  return sawModulo;
}

// Function-level safety gate for the boundary-wrap async path. The pipeliner
// hoists the modulo into the prologue, so a per-op walk can't see an unaligned
// live wrap. At function scope any rem not proven block-aligned means a live
// wrap, so async is unsafe for EVERY copy in the kernel. Returns true if safe.
static bool functionModuloIsSafe(Operation *op) {
  auto func = op->getParentOfType<FunctionOpInterface>();
  if (!func)
    return false;
  bool safe = true;
  func.walk([&](Operation *m) {
    if (!isa<arith::RemUIOp, arith::RemSIOp>(m))
      return;
    // Only per-element INDEX wraps threaten the affine form; a scalar modulo
    // (extent 1) is the GROUP-M pid swizzle and never breaks intra-tile access.
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

// Extract base/stride/origin from the flattened-index pattern:
// addptr(splat(base), addi(rowTerm, colTerm)). The nested-addptr shape is
// handled in extractAsyncCopyPtrInfo.
static bool extractFlatAsyncCopyPtrInfo(triton::AddPtrOp addptrOp,
                                        AsyncCopyPtrInfo &info,
                                        bool allowModulo) {
  auto *baseSplatOp = addptrOp->getOperand(0).getDefiningOp();
  if (!baseSplatOp || !isa<triton::SplatOp>(baseSplatOp)) {
    return false;
  }
  Value combinedIdx = addptrOp->getOperand(1);
  // allowModulo: caller proved the boundary-wrap is a no-op over this tile.
  if (!allowModulo && defChainContainsModulo(combinedIdx)) {
    return false;
  }
  auto *addiOp = combinedIdx.getDefiningOp();
  if (!addiOp || !isa<arith::AddIOp>(addiOp)) {
    return false;
  }

  auto peelBroadcast = [](Value v) -> Value {
    if (auto *bc = v.getDefiningOp())
      if (isa<triton::BroadcastOp>(bc))
        return bc->getOperand(0);
    return v;
  };
  // Which logical dim an index term varies along (0=row, 1=col, -1=unknown).
  // The row-major DMA is valid only when the strided term indexes the outer dim
  // and the unit term the inner; a transposed operand swaps these and the DMA
  // reads the tile transposed -> silent miscompile.
  auto termVaryingDim = [&](Value term) -> int {
    Value inner = peelBroadcast(term);
    if (auto *muli = inner.getDefiningOp())
      if (isa<arith::MulIOp>(muli))
        for (unsigned i = 0; i < 2; i++)
          if (auto *e = muli->getOperand(i).getDefiningOp())
            if (isa<triton::ExpandDimsOp>(e))
              inner = muli->getOperand(i);
    auto exp = inner.getDefiningOp<triton::ExpandDimsOp>();
    if (!exp)
      return -1;
    // expand_dims axis A inserts size-1 at A; index varies along the other dim.
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
        // inductor stride = constant dense<C>: record C as a scalar (an op here
        // would leave an unconvertible arith.constant post-legalization).
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

  // Strided term must index OUTER, unit term INNER; swapped (transposed
  // operand) would make the DMA read the tile transposed -> bail to sync copy.
  {
    Value stridedTerm = (colTerm == rhs) ? lhs : rhs;
    int stridedDim = termVaryingDim(stridedTerm);
    int colDim = termVaryingDim(colTerm);
    if (stridedDim != 0 || colDim != 1)
      return false;
  }

  info.basePtr = baseSplatOp->getOperand(0);
  info.stride = strideVal;
  // Peel a proven-no-op boundary wrap (allowModulo) so the pid-dependent row
  // origin survives; else the DMA reads row 0 for every program.
  info.rowStart = extractFirstElemScalar(rowRange, allowModulo);
  if (!info.rowStart)
    extractFirstElemConst(rowRange, info.rowStartConst, allowModulo);

  if (!allowModulo && defChainContainsModulo(colTerm))
    return false;
  Value colInner = peelBroadcast(colTerm);
  // Peel the column wrap under allowModulo (as for rows) so the pid_n*BLOCK_N
  // column origin survives; else colStart drops to 0 and every N-block reads B
  // from column 0 (only pid_n == 0 correct).
  info.colStart = extractFirstElemScalar(colInner, allowModulo);
  if (!info.colStart)
    extractFirstElemConst(colInner, info.colStartConst, allowModulo);
  return true;
}

// Conservative proof that the inner (dim-1) index of a 2D source pointer is
// unit-stride (contiguous columns). A non-unit inner stride (transposed/
// strided-view operand) reads the tile transposed -> silent miscompile.
// Returns true only on a positive proof; any unrecognized shape -> sync copy.
static bool innerDimIsUnitStride(Value ptrTensor) {
  auto *addptrOp = ptrTensor.getDefiningOp();
  if (!addptrOp || !isa<triton::AddPtrOp>(addptrOp))
    return false;

  // Collect the additive index terms (flat and nested addptr shapes).
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

  // The inner-dim term (expand_dims axis 0) must be a bare expand_dims, not
  // wrapped in a muli by a non-unit stride.
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

static bool rectStrideIs64B(ttg::AsyncCopyGlobalToLocalOp op) {
  return op->hasAttr("applegpu.rect_stride_64b");
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

    // Data is already in shared memory, so the matching async_wait is a no-op:
    // return a zero-initialized (completed) event slot.
    Value evSlot = createCompletedEventSlot(op, rewriter);
    rewriter.replaceOp(op, evSlot);
    return success();
  }

  // A tt.splat of a scalar i1 is a uniform mask (all-true/all-false) we can
  // gate the whole DMA on. Returns the scalar, or nullptr if not a splat.
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

  // Rectangular boundary mask: every and-tree leaf is a uniform splat(i1) or a
  // row/col bound `cmp slt idx, splat(extent)`. The DMA honors it exactly by
  // clamping the source tile to the remaining rows/cols and zero-filling the
  // rest (clamp mode 0), keeping other=0 masked loads on the DMA path.
  struct RectMaskInfo {
    SmallVector<Value, 2> uniforms; // scalar i1 leaves, all must be true
    Value rowBound, rowFirst;       // rows valid while row < rowBound
    Value colBound, colFirst;       // cols valid while col < colBound
    bool hasRow = false, hasCol = false;
    int64_t rowFirstConst = 0, colFirstConst = 0; // when scalar absent
  };

  static bool matchRectMask(Value mask, RectMaskInfo &info) {
    auto *def = mask.getDefiningOp();
    if (!def)
      return false;
    if (auto andOp = dyn_cast<arith::AndIOp>(def))
      return matchRectMask(andOp.getLhs(), info) &&
             matchRectMask(andOp.getRhs(), info);
    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      Value s = splat.getSrc();
      if (!s.getType().isInteger(1))
        return false;
      info.uniforms.push_back(s);
      return true;
    }
    if (auto bc = dyn_cast<triton::BroadcastOp>(def))
      return matchRectMask(bc.getSrc(), info);
    auto cmp = dyn_cast<arith::CmpIOp>(def);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      return false;
    auto resTy = dyn_cast<RankedTensorType>(cmp.getType());
    if (!resTy || resTy.getRank() != 2)
      return false;
    // Which logical dim does this bound? A (1 x N) row vector bounds the
    // inner (col) dim; an (N x 1) column vector bounds the outer (row) dim.
    bool boundsCols;
    if (resTy.getDimSize(0) == 1 && resTy.getDimSize(1) > 1)
      boundsCols = true;
    else if (resTy.getDimSize(1) == 1 && resTy.getDimSize(0) > 1)
      boundsCols = false;
    else
      return false;
    // Extent must be a uniform scalar.
    Value bound = extractFirstElemScalar(cmp.getRhs());
    if (!bound) {
      int64_t c;
      if (!extractFirstElemConst(cmp.getRhs(), c))
        return false;
      // Constant extents do not appear in the GEMM shapes this targets.
      return false;
    }
    // First index of the bounded range (smallest element).
    Value first = extractFirstElemScalar(cmp.getLhs());
    int64_t firstConst = 0;
    if (!first && !extractFirstElemConst(cmp.getLhs(), firstConst))
      return false;
    if (boundsCols) {
      if (info.hasCol)
        return false;
      info.hasCol = true;
      info.colBound = bound;
      info.colFirst = first;
      info.colFirstConst = firstConst;
    } else {
      if (info.hasRow)
        return false;
      info.hasRow = true;
      info.rowBound = bound;
      info.rowFirst = first;
      info.rowFirstConst = firstConst;
    }
    return true;
  }

  // The DMA writes only the src-tile intersection (no zero fill, measured on
  // M1), so the masked-out remainder of the dst slot must be zeroed explicitly
  // (cols >= srcCols or rows >= srcRows).
  static void emitResidualZero(ConversionPatternRewriter &rewriter,
                               Location loc, ModuleOp mod, Value dstBase,
                               int64_t rows, int64_t cols, unsigned elemBytes,
                               Value srcWBytes, Value srcHRows) {
    auto *ctx = rewriter.getContext();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i16Ty = IntegerType::get(ctx, 16);
    auto f32Ty = Float32Type::get(ctx);
    assert(llvm::isPowerOf2_64(cols) && llvm::isPowerOf2_32(elemBytes));
    int64_t colShift = llvm::Log2_64(cols);
    int64_t elemShift = llvm::Log2_32(elemBytes);

    Value elemShiftV = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemShift));
    Value srcCols = LLVM::LShrOp::create(rewriter, loc, srcWBytes, elemShiftV);

    // i32 result to match other lowerings' declaration of this intrinsic.
    auto i32T = IntegerType::get(ctx, 32);
    auto laneFn =
        getOrCreateFn(mod, rewriter, "air.thread_index_in_simdgroup", i32T, {});
    Value lane32 =
        LLVM::CallOp::create(rewriter, loc, laneFn, ValueRange{}).getResult();
    Value lane = LLVM::ZExtOp::create(rewriter, loc, i64Ty, lane32);
    (void)i16Ty;

    Value zeroF = LLVM::ConstantOp::create(rewriter, loc, f32Ty,
                                           rewriter.getF32FloatAttr(0.0f));
    Value colShiftV = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(colShift));
    Value colMask = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(cols - 1));
    Value colsV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(cols));
    Value rowsV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(rows));
    Value colPart = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::slt, srcCols, colsV);
    Value rowPart = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::slt, srcHRows, rowsV);
    Value partial = LLVM::OrOp::create(rewriter, loc, colPart, rowPart);
    auto *entry = rewriter.getInsertionBlock();
    auto entryPt = rewriter.getInsertionPoint();
    auto *fillB = rewriter.createBlock(entry->getParent(),
                                       std::next(Region::iterator(entry)));
    auto *doneB = rewriter.createBlock(entry->getParent(),
                                       std::next(Region::iterator(fillB)));
    doneB->getOperations().splice(doneB->begin(), entry->getOperations(),
                                  entryPt, entry->end());
    rewriter.setInsertionPointToEnd(entry);
    LLVM::CondBrOp::create(rewriter, loc, partial, fillB, doneB);
    rewriter.setInsertionPointToStart(fillB);
    int64_t total = rows * cols;
    Value totalV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                            rewriter.getI64IntegerAttr(total));
    Value c32 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                         rewriter.getI64IntegerAttr(32));
    auto *loopB = rewriter.createBlock(
        fillB->getParent(), std::next(Region::iterator(fillB)), {i64Ty}, {loc});
    rewriter.setInsertionPointToEnd(fillB);
    LLVM::BrOp::create(rewriter, loc, ValueRange{lane}, loopB);
    rewriter.setInsertionPointToStart(loopB);
    Value idx = loopB->getArgument(0);
    Value row = LLVM::LShrOp::create(rewriter, loc, idx, colShiftV);
    Value col = LLVM::AndOp::create(rewriter, loc, idx, colMask);
    Value colOOB = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge,
                                        col, srcCols);
    Value rowOOB = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge,
                                        row, srcHRows);
    Value oob = LLVM::OrOp::create(rewriter, loc, colOOB, rowOOB);
    auto *storeB = rewriter.createBlock(loopB->getParent(),
                                        std::next(Region::iterator(loopB)));
    auto *latchB = rewriter.createBlock(storeB->getParent(),
                                        std::next(Region::iterator(storeB)));
    rewriter.setInsertionPointToEnd(loopB);
    LLVM::CondBrOp::create(rewriter, loc, oob, storeB, latchB);
    rewriter.setInsertionPointToStart(storeB);
    Value p = LLVM::GEPOp::create(rewriter, loc, dstBase.getType(), f32Ty,
                                  dstBase, ArrayRef<LLVM::GEPArg>{idx});
    LLVM::StoreOp::create(rewriter, loc, zeroF, p);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, latchB);
    rewriter.setInsertionPointToStart(latchB);
    Value next = LLVM::AddOp::create(rewriter, loc, i64Ty, idx, c32);
    Value done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge,
                                      next, totalV);
    LLVM::CondBrOp::create(rewriter, loc, done, doneB, ValueRange{}, loopB,
                           ValueRange{next});
    rewriter.setInsertionPointToStart(doneB);
  }

  // remaining = clamp(bound - first, 0, cap) as i64, emitted from remapped
  // scalars. Returns null when the scalars cannot be remapped.
  static Value emitRemaining(ConversionPatternRewriter &rewriter, Location loc,
                             Value bound, Value first, int64_t firstConst,
                             int64_t cap) {
    auto i64Ty = rewriter.getI64Type();
    Value b = rewriter.getRemappedValue(bound);
    if (!b)
      return nullptr;
    if (b.getType() != i64Ty)
      b = LLVM::SExtOp::create(rewriter, loc, i64Ty, b);
    Value f;
    if (first) {
      f = rewriter.getRemappedValue(first);
      if (!f)
        return nullptr;
      if (f.getType() != i64Ty)
        f = LLVM::SExtOp::create(rewriter, loc, i64Ty, f);
    } else {
      f = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                   rewriter.getI64IntegerAttr(firstConst));
    }
    Value rem = LLVM::SubOp::create(rewriter, loc, i64Ty, b, f);
    Value zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(0));
    Value capV = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(cap));
    rem = LLVM::SMaxOp::create(rewriter, loc, rem, zero);
    return LLVM::SMinOp::create(rewriter, loc, rem, capV);
  }

  // Interior-tile fullness guard: a uniform i1 true exactly when every bounded
  // axis is fully in range (`first + extent <= bound`, plus uniform leaves), so
  // the clamp + residual zero are no-ops and the maskless full-tile DMA can
  // fire. Returns null if any scalar can't be remapped.
  static Value emitRectFullGuard(ConversionPatternRewriter &rewriter,
                                 Location loc, const RectMaskInfo &rectMask,
                                 int64_t tileRows, int64_t tileCols) {
    auto i64Ty = rewriter.getI64Type();
    auto axisGuard = [&](Value boundV, Value firstV, int64_t firstConst,
                         int64_t extent) -> Value {
      Value b = rewriter.getRemappedValue(boundV);
      if (!b)
        return nullptr;
      if (b.getType() != i64Ty)
        b = LLVM::SExtOp::create(rewriter, loc, i64Ty, b);
      Value f;
      if (firstV) {
        f = rewriter.getRemappedValue(firstV);
        if (!f)
          return nullptr;
        if (f.getType() != i64Ty)
          f = LLVM::SExtOp::create(rewriter, loc, i64Ty, f);
      } else {
        f = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                     rewriter.getI64IntegerAttr(firstConst));
      }
      Value ext = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(extent));
      Value hi = LLVM::AddOp::create(rewriter, loc, i64Ty, f, ext);
      return LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sle, hi,
                                  b);
    };

    Value guard;
    if (rectMask.hasRow) {
      guard = axisGuard(rectMask.rowBound, rectMask.rowFirst,
                        rectMask.rowFirstConst, tileRows);
      if (!guard)
        return nullptr;
    }
    if (rectMask.hasCol) {
      Value g = axisGuard(rectMask.colBound, rectMask.colFirst,
                          rectMask.colFirstConst, tileCols);
      if (!g)
        return nullptr;
      guard = guard ? (Value)LLVM::AndOp::create(rewriter, loc, guard, g) : g;
    }
    if (!guard)
      return nullptr;
    for (Value u : rectMask.uniforms) {
      Value ru = rewriter.getRemappedValue(u);
      if (!ru)
        return nullptr;
      guard = LLVM::AndOp::create(rewriter, loc, guard, ru);
    }
    return guard;
  }

  // Runtime tile-origin + row-stride fallback when the syntactic IR walk fails.
  // Derives the uniform tile-origin pointer and src row-stride from the
  // materialized per-element pointers; exact for ANY affine access.
  //   origin = ptrtoint(srcElems[k]) - (row_k*rowStrideBytes + col_k*elemBytes)
  // is uniform (per-lane part cancels). rowStrideBytes =
  // (ptr(k1)-ptr(k0))/rowGap for a row-adjacent register pair; no such pair ->
  // sync copy.
  LogicalResult lowerAsyncFromRuntimePtrs(
      ttg::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, bool useRectClamp = false,
      const RectMaskInfo *rectMask = nullptr, Value llvmGuard = nullptr,
      bool warp0Fire = false) const {
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

    // Register pair (k0,k1) of THIS thread sharing a column on different rows.
    // The row gap dr need not be 1; divide the byte delta by dr. Pick the
    // smallest positive dr.
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

    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    // Apply the memdesc slice/index offset (rotating staging slot); bare
    // getBase() stages every slot to slot 0.
    Value dstBase = smemObj.getShmemAffineBase(loc, rewriter, dstTy);

    Value dstStrideBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));

    // Cross-warp: partition the row dim across warps so each DMAs an exclusive
    // band and waits its OWN event (the warp-0-only form races). Rect-clamped
    // edge tiles keep the full-tile copy (residual-zero assumes the whole
    // tile).
    int64_t nWarps = warp0Fire ? totalWarps(op) : 1;
    bool partition =
        warp0Fire && !useRectClamp && nWarps > 1 && tileRows % nWarps == 0;
    int64_t bandRows = partition ? tileRows / nWarps : tileRows;

    if (partition) {
      Value wId = emitWarpId(op, rewriter, loc);
      Value wId64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, wId);
      Value bandRowsVal = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bandRows));
      Value bandStartRow =
          LLVM::MulOp::create(rewriter, loc, i64Ty, wId64, bandRowsVal);
      Value srcBandOff = LLVM::MulOp::create(rewriter, loc, i64Ty, bandStartRow,
                                             srcStrideBytes);
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(),
                                    IntegerType::get(ctx, 8), srcBase,
                                    ArrayRef<LLVM::GEPArg>{srcBandOff});
      Value dstBandOff = LLVM::MulOp::create(rewriter, loc, i64Ty, bandStartRow,
                                             dstStrideBytes);
      dstBase = LLVM::GEPOp::create(rewriter, loc, dstBase.getType(),
                                    IntegerType::get(ctx, 8), dstBase,
                                    ArrayRef<LLVM::GEPArg>{dstBandOff});
    }

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

    // Optionally-guarded async copy for a source tile. clampTile runs the rect
    // clamp + residual zero-fill; the full-tile arm skips both.
    auto emitCopy = [&](bool clampTile, Value baseGuard) {
      Value srcTileVec = tileVec;
      Value guard = baseGuard;
      if (clampTile && useRectClamp && rectMask) {
        Value srcW = widthVal, srcH = heightVal;
        if (rectMask->hasCol) {
          Value colRem = emitRemaining(rewriter, loc, rectMask->colBound,
                                       rectMask->colFirst,
                                       rectMask->colFirstConst, tileCols);
          srcW =
              LLVM::MulOp::create(rewriter, loc, i64Ty, colRem, elemBytesVal);
        }
        if (rectMask->hasRow) {
          srcH = emitRemaining(rewriter, loc, rectMask->rowBound,
                               rectMask->rowFirst, rectMask->rowFirstConst,
                               tileRows);
        }
        srcTileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcW, idx0);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcH, idx1);
        Value srcHForZero = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileRows));
        emitResidualZero(rewriter, loc, mod, dstBase, tileRows, tileCols,
                         elemBytes, srcW, srcHForZero);
        Value z = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(0));
        Value wPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcW, z);
        Value hPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcH, z);
        Value nonEmpty = LLVM::AndOp::create(rewriter, loc, wPos, hPos);
        guard = guard
                    ? (Value)LLVM::AndOp::create(rewriter, loc, guard, nonEmpty)
                    : nonEmpty;
      }
      if (guard) {
        auto *parentBlock = rewriter.getInsertionBlock();
        auto insertPt = rewriter.getInsertionPoint();
        auto *thenBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
        auto *afterBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(thenBlock)));
        afterBlock->getOperations().splice(afterBlock->begin(),
                                           parentBlock->getOperations(),
                                           insertPt, parentBlock->end());
        rewriter.setInsertionPointToEnd(parentBlock);
        LLVM::CondBrOp::create(rewriter, loc, guard, thenBlock, afterBlock);
        rewriter.setInsertionPointToStart(thenBlock);
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);
        rewriter.setInsertionPointToStart(afterBlock);
      } else {
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
      }
    };

    // Interior-tile fast path: when the rect mask proves the tile fully in
    // bounds the clamp + residual zero are no-ops, so fire the maskless
    // full-tile DMA; edge tiles keep the clamped DMA.
    Value fullGuard =
        (useRectClamp && rectMask)
            ? emitRectFullGuard(rewriter, loc, *rectMask, tileRows, tileCols)
            : nullptr;
    if (fullGuard) {
      auto *parentBlock = rewriter.getInsertionBlock();
      auto insertPt = rewriter.getInsertionPoint();
      auto *fullBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
      auto *edgeBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(fullBlock)));
      auto *contBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(edgeBlock)));
      contBlock->getOperations().splice(contBlock->begin(),
                                        parentBlock->getOperations(), insertPt,
                                        parentBlock->end());
      rewriter.setInsertionPointToEnd(parentBlock);
      LLVM::CondBrOp::create(rewriter, loc, fullGuard, fullBlock, edgeBlock);

      rewriter.setInsertionPointToEnd(fullBlock);
      emitCopy(/*clampTile=*/false, llvmGuard);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToEnd(edgeBlock);
      emitCopy(/*clampTile=*/true, llvmGuard);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToStart(contBlock);
    } else {
      emitCopy(/*clampTile=*/true, llvmGuard);
    }

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

    auto shape = srcTy.getShape();
    bool canAsyncDMA = (shape.size() == 2);

    // An async DMA into a single-slot staging buffer races across pipelined
    // K-steps (membar orders the sync scatter but not the in-flight DMA), so
    // force the membar-ordered sync copy when the dest is one slot.
    bool singleSlotLoopStaging = false;
    {
      Value dst = op.getResult();
      bool viaIndex = false;
      while (auto idx = dst.getDefiningOp<ttg::MemDescIndexOp>()) {
        dst = idx.getSrc();
        viaIndex = true;
      }
      if (viaIndex)
        if (auto alloc = dst.getDefiningOp<ttg::LocalAllocOp>()) {
          auto ty = cast<ttg::MemDescType>(alloc.getType());
          if (ty.getRank() >= 1 && ty.getShape().front() == 1)
            singleSlotLoopStaging = true;
        }
    }
    if (singleSlotLoopStaging)
      canAsyncDMA = false;

    // A multi-warp-outer staged copy (warpsPerCTA[outer]>1) can't fire from
    // every warp: identical whole-tile copies write-write race. Handled below.
    bool outerCrossWarp = asyncCopyOuterDimCrossWarp(op);

    // The async copy reads the full tile, so dropping a non-uniform mask is
    // safe only when the tile is proven fully in-bounds. Also gates the modulo
    // below.
    bool tileFullyInBounds =
        allModuloBlockAligned(op.getSrc()) && functionModuloIsSafe(op);

    // Async DMA takes no mask or a UNIFORM scalar gating the whole copy. A
    // non-uniform mask is dropped only when the tile is proven in-bounds; else
    // fall back to the per-element sync copy (honors the mask exactly).
    Value mlirMaskScalar;
    Value llvmMaskScalar;
    RectMaskInfo rectMask;
    bool useRectClamp = false;
    if (canAsyncDMA && llMask) {
      mlirMaskScalar = extractScalarMask(op.getMask());
      if (mlirMaskScalar) {
        llvmMaskScalar = rewriter.getRemappedValue(mlirMaskScalar);
      } else if (!tileFullyInBounds) {
        // Rect row/col bound mask with other=0: honored exactly by clamping the
        // source tile + zero-fill (clamp mode 0).
        RectMaskInfo probe;
        bool otherIsZero = true;
        if (op.getOther())
          if (auto cst = op.getOther().getDefiningOp<arith::ConstantOp>())
            if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
              otherIsZero =
                  dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
        auto remappable = [&](Value v) {
          return !v || rewriter.getRemappedValue(v) != nullptr;
        };
        // Residual zeroing stores f32; other element widths keep sync copy.
        if (!elemTy.isF32())
          otherIsZero = false;
        // LANDMINE: the DMA engine mis-copies when the source row stride is not
        // 64B-aligned (K=72/1000 corrupt, 80/96/1024 exact). The proof is
        // stamped pre-conversion; without it refuse rect and use sync copy.
        if (otherIsZero && !rectStrideIs64B(op))
          otherIsZero = false;
        if (otherIsZero && matchRectMask(op.getMask(), probe) &&
            remappable(probe.rowBound) && remappable(probe.rowFirst) &&
            remappable(probe.colBound) && remappable(probe.colFirst)) {
          rectMask = probe;
          useRectClamp = true;
          // Uniform leaves (pipeliner K-guards) gate the whole copy.
          for (Value u : rectMask.uniforms) {
            Value lu = rewriter.getRemappedValue(u);
            if (!lu) {
              useRectClamp = false;
              break;
            }
            llvmMaskScalar =
                llvmMaskScalar ? (Value)LLVM::AndOp::create(rewriter, loc,
                                                            llvmMaskScalar, lu)
                               : lu;
          }
          if (useRectClamp && !op->hasAttr("applegpu.rect_dma"))
            useRectClamp = false;
        }
        if (!useRectClamp) {
          // Non-uniform mask on a possibly-partial tile: use the sync copy.
          canAsyncDMA = false;
          llvmMaskScalar = nullptr;
        }
      }
    }

    // AxisInfo: a boundary-wrap modulo is a no-op when the source is contiguous
    // in its inner dim (wrap never crosses a tile edge), so async is exact.
    bool allowModulo = false;
    if (canAsyncDMA && axisInfo) {
      AxisInfo *ai = axisInfo->getAxisInfo(op.getSrc());
      if (ai) {
        int innerDim = shape.size() - 1;
        if (ai->getContiguity(innerDim) >= shape[innerDim])
          allowModulo = true;
      }
    }
    // AxisInfo (pre-conversion) often returns null here; fall back to the
    // IR-based proof (every wrap annotated contiguous across the tile extent).
    if (canAsyncDMA && !allowModulo && allModuloBlockAligned(op.getSrc()))
      allowModulo = true;

    // Function-level live-wrap guard: the pipeliner hides the modulo in the
    // prologue, so any non-block-aligned wrap in the kernel forces sync.
    bool funcModuloSafe = functionModuloIsSafe(op);
    if (!funcModuloSafe)
      allowModulo = false;

    AsyncCopyPtrInfo ptrInfo;
    Value llvmStride;
    if (canAsyncDMA && useRectClamp) {
      // LANDMINE: rect-clamped DMA needs a 64B-aligned source row stride
      // (engine mis-copies otherwise); the proof is the pre-conversion stamp.
      if (!rectStrideIs64B(op))
        canAsyncDMA = false;
    }
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
      // row-stride extraction from the materialized pointers (covers ANY affine
      // access). Gated on a 2D tile + funcModuloSafe: a live boundary wrap
      // breaks affinity, and the maskless full-tile read is correct only when
      // proven in-bounds (a rect-matched mask counts, it is honored exactly).
      bool noLiveBoundary =
          !llMask || mlirMaskScalar || tileFullyInBounds || useRectClamp;
      // The runtime DMA assumes contiguous columns; a transposed/strided-view
      // inner dim would read the tile transposed, so require a unit-stride
      // proof.
      bool innerUnit = innerDimIsUnitStride(op.getSrc());
      bool runtimeSafe = (shape.size() == 2) && funcModuloSafe &&
                         noLiveBoundary && innerUnit && !singleSlotLoopStaging;
      if (runtimeSafe) {
        if (succeeded(
                lowerAsyncFromRuntimePtrs(op, adaptor, rewriter, useRectClamp,
                                          useRectClamp ? &rectMask : nullptr,
                                          llvmMaskScalar, outerCrossWarp)))
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

    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;

    int64_t tileRows = shape[0];
    int64_t tileCols = shape[1];
    int64_t tileWidthBytes = tileCols * elemBytes;

    Value strideI64 = llvmStride;
    if (llvmStride.getType() != i64Ty) {
      strideI64 = LLVM::SExtOp::create(rewriter, loc, i64Ty, llvmStride);
    }
    Value elemBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemBytes));
    Value srcStrideBytes =
        LLVM::MulOp::create(rewriter, loc, i64Ty, strideI64, elemBytesVal);

    // Destination stride in bytes (TG is packed).
    Value dstStrideBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));

    // UNIFORM tile origin = basePtr + rowStart*stride + colStart (all scalar).
    // The intrinsic is cooperative: all threads must pass the SAME base, so we
    // can't use per-thread pointers from srcElems[0].
    Value srcBase = llvmBasePtr;
    if (llvmRowStart) {
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
      srcBase =
          LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy, srcBase,
                              ArrayRef<LLVM::GEPArg>{llvmColStart});
    } else if (ptrInfo.colStartConst != 0) {
      // Folded-constant first col (prefetched K-block offset). The
      // num_stages>=3 correctness fix.
      Value cc = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)ptrInfo.colStartConst));
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{cc});
    }

    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    // The DMA must write the SAME affine base the matching local_load reads, or
    // it stages every slot to slot 0, desyncing the rotating buffer.
    Value dstBase = smemObj.getShmemAffineBase(loc, rewriter, dstTy);

    // Cross-warp: partition the row dim across warps so each DMAs an exclusive
    // band and waits its OWN event (the warp-0-only form races). Only on an
    // even row split; otherwise fall back to warp-0-only firing.
    int64_t nWarps = outerCrossWarp ? totalWarps(op) : 1;
    bool partition =
        outerCrossWarp && !useRectClamp && nWarps > 1 && tileRows % nWarps == 0;
    int64_t bandRows = partition ? tileRows / nWarps : tileRows;

    if (partition) {
      Value wId = emitWarpId(op, rewriter, loc);
      Value wId64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, wId);
      Value bandRowsVal = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bandRows));
      Value bandStartRow =
          LLVM::MulOp::create(rewriter, loc, i64Ty, wId64, bandRowsVal);
      Value srcBandOff = LLVM::MulOp::create(rewriter, loc, i64Ty, bandStartRow,
                                             srcStrideBytes);
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(),
                                    IntegerType::get(ctx, 8), srcBase,
                                    ArrayRef<LLVM::GEPArg>{srcBandOff});
      Value dstBandOff = LLVM::MulOp::create(rewriter, loc, i64Ty, bandStartRow,
                                             dstStrideBytes);
      dstBase = LLVM::GEPOp::create(rewriter, loc, dstBase.getType(),
                                    IntegerType::get(ctx, 8), dstBase,
                                    ArrayRef<LLVM::GEPArg>{dstBandOff});
    } else if (outerCrossWarp) {
      Value w0 = emitWarp0Pred(op, rewriter, loc);
      llvmMaskScalar =
          llvmMaskScalar
              ? (Value)LLVM::AndOp::create(rewriter, loc, llvmMaskScalar, w0)
              : w0;
    }

    // height = bandRows (== tileRows when not partitioned).
    Value widthVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));
    Value heightVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(bandRows));

    Value tileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
    Value idx0 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(0));
    Value idx1 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                          rewriter.getI32IntegerAttr(1));
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

    // sizeof=1, alignof=1 (byte-granularity copy)
    Value one64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(1));

    Value clamp = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));

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

    // Optionally-guarded async copy for a source tile. clampTile runs the rect
    // clamp + residual zero-fill; the full-tile arm skips both.
    auto emitCopy = [&](bool clampTile, Value baseGuard) {
      Value srcTileVec = tileVec;
      Value guard = baseGuard;
      if (clampTile && useRectClamp) {
        Value srcW = widthVal, srcH = heightVal;
        if (rectMask.hasCol) {
          Value colRem =
              emitRemaining(rewriter, loc, rectMask.colBound, rectMask.colFirst,
                            rectMask.colFirstConst, tileCols);
          Value eb = LLVM::ConstantOp::create(
              rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(elemBytes));
          srcW = LLVM::MulOp::create(rewriter, loc, i64Ty, colRem, eb);
        }
        if (rectMask.hasRow) {
          srcH =
              emitRemaining(rewriter, loc, rectMask.rowBound, rectMask.rowFirst,
                            rectMask.rowFirstConst, tileRows);
        }
        srcTileVec = LLVM::UndefOp::create(rewriter, loc, vec2i64Ty);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcW, idx0);
        srcTileVec = LLVM::InsertElementOp::create(rewriter, loc, vec2i64Ty,
                                                   srcTileVec, srcH, idx1);
        Value srcHForZero = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileRows));
        emitResidualZero(rewriter, loc, mod, dstBase, tileRows, tileCols,
                         elemBytes, srcW, srcHForZero);
        Value z = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(0));
        Value wPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcW, z);
        Value hPos = LLVM::ICmpOp::create(rewriter, loc,
                                          LLVM::ICmpPredicate::sgt, srcH, z);
        Value nonEmpty = LLVM::AndOp::create(rewriter, loc, wPos, hPos);
        guard = guard
                    ? (Value)LLVM::AndOp::create(rewriter, loc, guard, nonEmpty)
                    : nonEmpty;
      }
      if (guard) {
        auto *parentBlock = rewriter.getInsertionBlock();
        auto insertPt = rewriter.getInsertionPoint();
        auto *thenBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
        auto *afterBlock = rewriter.createBlock(
            parentBlock->getParent(), std::next(Region::iterator(thenBlock)));
        afterBlock->getOperations().splice(afterBlock->begin(),
                                           parentBlock->getOperations(),
                                           insertPt, parentBlock->end());
        rewriter.setInsertionPointToEnd(parentBlock);
        LLVM::CondBrOp::create(rewriter, loc, guard, thenBlock, afterBlock);
        rewriter.setInsertionPointToStart(thenBlock);
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);
        rewriter.setInsertionPointToStart(afterBlock);
      } else {
        Value event = LLVM::CallOp::create(
                          rewriter, loc, asyncCopyFn,
                          ValueRange{one64, one64, dstBase, dstStrideBytes,
                                     one64, tileVec, srcBase, srcStrideBytes,
                                     one64, srcTileVec, offsetVec, clamp})
                          .getResult();
        LLVM::StoreOp::create(rewriter, loc, event, evAlloca);
      }
    };

    // Interior-tile fast path: when the rect mask proves the tile fully in
    // bounds (incl. a non-partial K-tile) the clamp + residual zero are no-ops,
    // so fire the maskless full-tile DMA; edge tiles keep the clamped DMA.
    Value fullGuard = useRectClamp ? emitRectFullGuard(rewriter, loc, rectMask,
                                                       tileRows, tileCols)
                                   : nullptr;
    if (fullGuard) {
      auto *parentBlock = rewriter.getInsertionBlock();
      auto insertPt = rewriter.getInsertionPoint();
      auto *fullBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(parentBlock)));
      auto *edgeBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(fullBlock)));
      auto *contBlock = rewriter.createBlock(
          parentBlock->getParent(), std::next(Region::iterator(edgeBlock)));
      contBlock->getOperations().splice(contBlock->begin(),
                                        parentBlock->getOperations(), insertPt,
                                        parentBlock->end());
      rewriter.setInsertionPointToEnd(parentBlock);
      LLVM::CondBrOp::create(rewriter, loc, fullGuard, fullBlock, edgeBlock);

      rewriter.setInsertionPointToEnd(fullBlock);
      emitCopy(/*clampTile=*/false, llvmMaskScalar);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToEnd(edgeBlock);
      emitCopy(/*clampTile=*/true, llvmMaskScalar);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, contBlock);

      rewriter.setInsertionPointToStart(contBlock);
    } else {
      emitCopy(/*clampTile=*/true, llvmMaskScalar);
    }

    // The token IS this copy's event slot; as a scf.for iter_arg it selects the
    // correct alternating buffer each iteration for the matching async_wait.
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
    // Metal events are per-copy, so a group just threads its member copy's slot
    // onward (one copy per commit_group). Empty group -> completed slot
    // (no-op).
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

    // Wait on exactly the copies whose tokens this wait consumes. One
    // wait_simdgroup_events(1, slot) per token keeps the slot scalar (Metal v1
    // bitcode mishandles arrays of typed pointers); zero-init slots make a
    // sync/skip token a no-op. Waits are unconditional: gating on the DMA decl
    // is conversion-order dependent, and a dropped wait reads in-flight slots.
    auto waitFn = getOrCreateFn(mod, rewriter, "air.wait_simdgroup_events",
                                voidTy, {i32Ty, ptrTy0});
    Value oneI32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(1));
    for (Value evSlot : adaptor.getAsyncToken())
      LLVM::CallOp::create(rewriter, loc, waitFn, ValueRange{oneI32, evSlot});

    // TG barrier for shared-memory visibility (sync and async paths). LANDMINE:
    // flag 2 (TG fence) suffices ONLY because async copies reaching a consumer
    // are single-simdgroup or per-warp-exclusive. A cross-warp-consumed band
    // would need flag 3, but those are routed to sync copy (see runtimeSafe).
    auto barrFn =
        getOrCreateFn(mod, rewriter, "air.wg.barrier", voidTy, {i32Ty, i32Ty});
    Value barrFlag = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                              rewriter.getI32IntegerAttr(2));
    Value barrScope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               rewriter.getI32IntegerAttr(1));
    LLVM::CallOp::create(rewriter, loc, barrFn,
                         ValueRange{barrFlag, barrScope});

    // After the wait all input copies are complete; return a completed slot.
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

} // anonymous namespace

// Defined out of the anonymous namespace: the pass driver also reconstructs
// async-copy pointer info to tag rect_stride_64b before lowering.
bool extractAsyncCopyPtrInfo(Value ptrTensor, AsyncCopyPtrInfo &info,
                             bool allowModulo) {
  auto *addptrOp = ptrTensor.getDefiningOp();
  if (!addptrOp || !isa<triton::AddPtrOp>(addptrOp))
    return false;

  // Outer addptr operand 0 is broadcast(inner_addptr), or a splat (flattened
  // combined-index GEMM) for the flat matcher.
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

  auto *baseSplatOp = innerOp->getOperand(0).getDefiningOp();
  if (!baseSplatOp || !isa<triton::SplatOp>(baseSplatOp))
    return false;
  info.basePtr = baseSplatOp->getOperand(0);

  // Row offset = muli(expand_dims(row_range), splat(stride)).
  Value rowOffset = innerOp->getOperand(1);
  auto *muliOp = rowOffset.getDefiningOp();
  if (!muliOp || !isa<arith::MulIOp>(muliOp))
    return false;

  // A modulo-indexed row offset is non-affine; bail to the sync copy.
  if (defChainContainsModulo(rowOffset))
    return false;

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

  // The strided row offset must index the OUTER dim (expand_dims axis 1). A
  // transposed operand puts the stride on the inner dim and the DMA reads the
  // tile transposed -> silent miscompile; bail to the sync copy.
  if (expandDimsVal)
    if (auto exp = expandDimsVal.getDefiningOp<triton::ExpandDimsOp>())
      if (exp.getAxis() != 1)
        return false;

  if (expandDimsVal) {
    info.rowStart = extractFirstElemScalar(expandDimsVal, allowModulo);
    if (!info.rowStart)
      extractFirstElemConst(expandDimsVal, info.rowStartConst, allowModulo);
  }
  // nullptr (and rowStartConst 0) means first row = 0

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

bool extractAffineMmaPtrInfo(Value ptrTensor, AffineMmaPtrInfo &info) {
  auto *addptrOp = ptrTensor.getDefiningOp();
  if (!addptrOp || !isa<triton::AddPtrOp>(addptrOp))
    return false;

  // Collect additive index terms across the flat and nested-addptr shapes.
  SmallVector<Value> terms, work;
  work.push_back(addptrOp->getOperand(1));
  work.push_back(addptrOp->getOperand(0));
  unsigned budget = 256;
  while (!work.empty() && budget--) {
    Value v = work.pop_back_val();
    if (auto *d = v.getDefiningOp()) {
      if (isa<triton::BroadcastOp>(d) || isa<triton::SplatOp>(d)) {
        work.push_back(d->getOperand(0));
        continue;
      }
      if (auto add = dyn_cast<arith::AddIOp>(d)) {
        work.push_back(add.getLhs());
        work.push_back(add.getRhs());
        continue;
      }
      if (auto inAddptr = dyn_cast<triton::AddPtrOp>(d)) {
        work.push_back(inAddptr->getOperand(0));
        work.push_back(inAddptr->getOperand(1));
        continue;
      }
    }
    terms.push_back(v);
  }
  if (budget == 0)
    return false;

  auto peelBroadcast = [](Value v) -> Value {
    while (auto *bc = v.getDefiningOp()) {
      if (isa<triton::BroadcastOp>(bc)) {
        v = bc->getOperand(0);
        continue;
      }
      break;
    }
    return v;
  };

  // expand_dims axis 1 varies row (dim 0), axis 0 varies col; bare = unit
  // stride, muli(expand_dims, s) = stride s.
  Value strideVal[2];
  int64_t strideConst[2] = {INT64_MIN, INT64_MIN};
  bool sawUnit[2] = {false, false};
  auto record = [&](int dim, Value s, int64_t sc) {
    if (s)
      strideVal[dim] = s;
    if (sc != INT64_MIN)
      strideConst[dim] = sc;
    if (!s && sc == INT64_MIN)
      sawUnit[dim] = true;
  };

  for (Value t : terms) {
    if (defChainContainsModulo(t))
      return false;
    Value inner = peelBroadcast(t);
    if (auto muli = inner.getDefiningOp<arith::MulIOp>()) {
      triton::ExpandDimsOp exp;
      Value strideSSA;
      int64_t strideC = INT64_MIN;
      for (unsigned i = 0; i < 2; i++) {
        Value opnd = muli->getOperand(i);
        Value p = peelBroadcast(opnd);
        if (auto e = p.getDefiningOp<triton::ExpandDimsOp>())
          exp = e;
        else if (auto sp = opnd.getDefiningOp<triton::SplatOp>())
          strideSSA = sp->getOperand(0);
        else if (auto cst = p.getDefiningOp<arith::ConstantOp>()) {
          if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
            if (dense.isSplat())
              strideC = dense.getSplatValue<APInt>().getSExtValue();
        }
      }
      if (!exp || (!strideSSA && strideC == INT64_MIN))
        return false;
      int dim = exp.getAxis() == 1 ? 0 : 1;
      record(dim, strideSSA, strideC);
      continue;
    }
    if (auto exp = inner.getDefiningOp<triton::ExpandDimsOp>()) {
      int dim = exp.getAxis() == 1 ? 0 : 1;
      record(dim, Value(), INT64_MIN);
      continue;
    }
  }

  auto haveDim = [&](int dim) {
    return strideVal[dim] || strideConst[dim] != INT64_MIN || sawUnit[dim];
  };
  if (!haveDim(0) || !haveDim(1))
    return false;

  auto setStride = [&](int dim, Value &outV, int64_t &outC) {
    if (strideVal[dim]) {
      outV = strideVal[dim];
      outC = INT64_MIN;
    } else if (strideConst[dim] != INT64_MIN) {
      outC = strideConst[dim];
    } else {
      outC = 1;
    }
  };
  setStride(0, info.rowStride, info.rowStrideConst);
  setStride(1, info.colStride, info.colStrideConst);
  bool rowUnit = !info.rowStride && info.rowStrideConst == 1;
  bool colUnit = !info.colStride && info.colStrideConst == 1;
  // Exactly one dim must be unit-stride for the SG tile load; transpose = the
  // inner (col) dim carries the leading stride.
  if (rowUnit == colUnit)
    return false;
  info.transposed = rowUnit;
  return true;
}

void populateAsyncCopyPatterns(LLVMTypeConverter &typeConverter,
                               RewritePatternSet &patterns,
                               ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  patterns.add<AppleBarrierOpConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
  patterns.add<AsyncCopyGlobalToLocalOpAppleConversion>(
      typeConverter, &axisInfoAnalysis,
      PatternBenefit(patternBenefitDefault + 10));
  patterns.add<AsyncCommitGroupOpAppleConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
  patterns.add<AsyncWaitOpAppleConversion>(
      typeConverter, PatternBenefit(patternBenefitDefault + 10));
}

} // namespace mlir::triton::applegpu
