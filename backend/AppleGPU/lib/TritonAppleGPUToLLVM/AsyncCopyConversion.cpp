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
#include "ConvertCommon.h"

namespace mlir::triton::applegpu {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

namespace {

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

// Predicate selecting the first simdgroup (tid < 32). A multi-warp-outer copy
// fires from this one warp only so the redundant whole-tile copies do not
// write-write race on the shared region; the staging barrier publishes.
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

  // Rectangular boundary mask: every leaf of the and-tree is either a uniform
  // splat(i1) or a row/col bound `cmp slt idx, splat(extent)`. The async DMA
  // honors such a mask exactly by clamping the source tile to
  // (extent - firstIdx) remaining rows/cols and zero-filling the rest
  // (clamp mode 0), so other=0 masked loads can stay on the DMA path.
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
    // The 2D DMA mis-copies when the source row stride is not 64B-aligned
    // (measured: K=72/1000 corrupt, K=80/96/1024 exact). Strides are the
    // matrix extents here, so require every bound to be a kernel argument
    // Triton proved 16-element divisible.
    // First index of the bounded range (its smallest element: indices ascend).
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
  // M1), so the masked-out remainder of the dst slot must be zeroed
  // explicitly: every lane of the issuing warp stores 0 to the dst elements
  // with col >= srcCols or row >= srcRows. Runs after the fire (disjoint
  // bytes), skipped per element when the tile is full.
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

    // Other lowerings declare this intrinsic with i32 result; match them so
    // a single declaration serves every call site.
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

  // Interior-tile fullness guard for the rect-clamped DMA load: a uniform i1
  // that is true exactly when every bounded axis is fully in range, so the
  // source clamp + residual zero-fill are no-ops and we can fire the maskless
  // full-tile DMA (byte-identical to the --even path).
  //
  // Per axis the guard is `first + extent <= bound`. The K axis is bounded too
  // (A's mask is `(om<M)&(kk<K)`, B's is `(kk<K)&(on<N)`), so the LAST partial
  // K-tile — where `kk + BK > K` on a ragged-K shape — makes the K-axis guard
  // false and correctly keeps the clamped path. Uniform leaves (pipeliner
  // K-guards) must also be true. Returns null if any scalar cannot be remapped,
  // in which case the caller keeps the unconditional clamped DMA.
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

    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    Value dstBase = smemObj.getBase();

    Value dstStrideBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(tileWidthBytes));

    // Cross-warp copies fire from one simdgroup only: tid<32 issues the whole
    // tile, the zero-init event slot makes other warps' waits no-ops and the
    // staging barrier publishes.
    int64_t bandRows = tileRows;
    if (warp0Fire) {
      Value w0 = emitWarp0Pred(op, rewriter, loc);
      llvmGuard = llvmGuard
                      ? (Value)LLVM::AndOp::create(rewriter, loc, llvmGuard, w0)
                      : w0;
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

    // Build the (residual-zero + optionally-guarded) async copy for a given
    // source tile. `clampTile` runs the rect clamp + residual zero-fill; the
    // full-tile arm passes the compile-const tileVec and skips both, emitting
    // exactly what the maskless --even path does.
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

    // Interior-tile fast path: when the rect mask proves the tile is fully in
    // bounds (incl. a non-partial K-tile) the clamp + residual zero are no-ops,
    // so fire the maskless full-tile DMA; edge tiles keep the clamped DMA.
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

    auto shape = srcTy.getShape();
    bool canAsyncDMA = (shape.size() == 2);
    // A multi-warp-outer staged copy (warpsPerCTA[outer]>1) cannot fire from
    // every warp: identical whole-tile copies write-write race on the same
    // bytes. Fire from one simdgroup only (tid<32), the staging barrier
    // publishes.
    bool outerCrossWarp = asyncCopyOuterDimCrossWarp(op);

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
    RectMaskInfo rectMask;
    bool useRectClamp = false;
    if (canAsyncDMA && llMask) {
      mlirMaskScalar = extractScalarMask(op.getMask());
      if (mlirMaskScalar) {
        llvmMaskScalar = rewriter.getRemappedValue(mlirMaskScalar);
      } else if (!tileFullyInBounds) {
        // Rectangular row/col bound mask with other=0: the DMA honors it
        // exactly by clamping the source tile and zero-filling (clamp mode 0).
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
        // The DMA engine mis-copies when the source row stride is not 64B
        // aligned (measured: K=72/1000 corrupt, 80/96/1024 exact). The proof
        // is stamped pre-conversion (constant stride or a tt.divisibility>=16
        // kernel-arg stride); without it the sync copy is the only sound
        // fallback, so refuse rect.
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
          // Non-uniform boundary mask on a possibly-partial tile: async cannot
          // honor per-element bounds, so use the sync copy.
          canAsyncDMA = false;
          llvmMaskScalar = nullptr;
        }
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
    if (canAsyncDMA && useRectClamp) {
      // Rect-clamped DMA needs a 64B-aligned source row stride (the engine
      // mis-copies otherwise); the proof is the pre-conversion stamp.
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
      // A rect-matched boundary mask is honored exactly by the clamped src
      // tile + zero fill, so it does not count as a live boundary.
      bool noLiveBoundary =
          !llMask || mlirMaskScalar || tileFullyInBounds || useRectClamp;
      // The runtime DMA assumes contiguous columns (inner unit stride). For a
      // transposed/strided-view operand the inner dim is strided, so the
      // contiguous-row copy reads the tile transposed. Require a positive
      // unit-inner-stride proof; otherwise use the layout-exact sync copy.
      bool innerUnit = innerDimIsUnitStride(op.getSrc());
      bool runtimeSafe =
          (shape.size() == 2) && funcModuloSafe && noLiveBoundary && innerUnit;
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
      // Folded-constant first col (the prefetched K-block offset): GEP by
      // colStartConst elements. THIS is the num_stages>=3 correctness fix.
      Value cc = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)ptrInfo.colStartConst));
      srcBase = LLVM::GEPOp::create(rewriter, loc, srcBase.getType(), elemTy,
                                    srcBase, ArrayRef<LLVM::GEPArg>{cc});
    }

    Value llDst = adaptor.getResult();
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, elemTy, rewriter);
    Value dstBase = smemObj.getBase();

    // Cross-warp copies fire from one simdgroup only (tid<32); zero-init
    // event slots keep other warps' waits no-ops, the staging barrier
    // publishes.
    int64_t bandRows = tileRows;
    if (outerCrossWarp) {
      Value w0 = emitWarp0Pred(op, rewriter, loc);
      llvmMaskScalar =
          llvmMaskScalar
              ? (Value)LLVM::AndOp::create(rewriter, loc, llvmMaskScalar, w0)
              : w0;
    }

    // height = bandRows so each warp copies only its own band (== tileRows when
    // not partitioned).
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

    // Build the (rect clamp + residual zero) optionally-guarded async copy for
    // a source tile. The full-tile arm passes the compile-const tileVec and
    // skips the clamp + residual zero, emitting exactly the maskless --even
    // DMA.
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

    // Wait on EXACTLY the copies whose tokens this wait consumes. Each token
    // (adaptor operand) is that copy's event slot (ptr addrspace(0)); see the
    // AsyncToken type conversion and the async_copy lowering. The token is a
    // scf.for iter_arg, so the loop-carried value selects the correct
    // alternating (double/triple-buffered) buffer each iteration: this is the
    // num_stages>=3 correctness fix. A separate wait_simdgroup_events(1,
    // slot) per token keeps the slot scalar (Metal v1 bitcode handles arrays
    // of typed pointers poorly). Each slot is zero-initialized in the entry
    // block, so a token from a sync/skip path holds a complete event and the
    // wait is a real no-op, never a read of an uninitialized pointer. The
    // waits are emitted UNCONDITIONALLY: gating them on the module already
    // containing the DMA declaration is conversion-order dependent (a loop
    // wait lowers before the loop's copies; sync prologue copies declare
    // nothing) and dropping the wait while any DMA is live reads in-flight
    // slots.
    auto waitFn = getOrCreateFn(mod, rewriter, "air.wait_simdgroup_events",
                                voidTy, {i32Ty, ptrTy0});
    Value oneI32 = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(1));
    for (Value evSlot : adaptor.getAsyncToken())
      LLVM::CallOp::create(rewriter, loc, waitFn, ValueRange{oneI32, evSlot});

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

} // anonymous namespace

// Defined out of the anonymous namespace: the pass driver also reconstructs
// async-copy pointer info to tag rect_stride_64b before lowering.
bool extractAsyncCopyPtrInfo(Value ptrTensor, AsyncCopyPtrInfo &info,
                                    bool allowModulo) {
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

  auto *baseSplatOp = innerOp->getOperand(0).getDefiningOp();
  if (!baseSplatOp || !isa<triton::SplatOp>(baseSplatOp))
    return false;
  info.basePtr = baseSplatOp->getOperand(0);

  // Row offset = muli(expand_dims(row_range), splat(stride)).
  Value rowOffset = innerOp->getOperand(1);
  auto *muliOp = rowOffset.getDefiningOp();
  if (!muliOp || !isa<arith::MulIOp>(muliOp))
    return false;

  // Non-affine (modulo-indexed) row offset defeats the constant-stride DMA
  // reconstruction below. Bail so the caller uses the sync per-element copy.
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

  // The strided (row) offset must index the OUTER dim (expand_dims axis 1 ->
  // Nx1 column vector). A transposed operand instead puts the explicit stride
  // on the INNER dim (expand_dims axis 0), and the contiguous-row DMA would
  // then read the tile transposed -> silent miscompile. Bail to the sync copy.
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
