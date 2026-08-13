// EmitMSLDotDma.cpp - device-direct operand staging (threadgroup DMA).
//
// Eligibility, slot/transpose bookkeeping and the AST builders that stage a dot
// operand device->threadgroup through air.simdgroup_async_copy_2d instead of
// materialising it into registers and scattering it. Split out of
// EmitMSLDot.cpp unchanged.
//
// The pointer-tensor recognisers these consult (peelBroadcast,
// matchTilePointer, matchDirectStage) stay in EmitMSLDot.cpp, which matches on
// them too.
//
// This path ships OFF (TRITON_MSL_DMA_STAGE); see CLAUDE.local.md.
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.

#include "MSLConstants.h"
#include "MSLEmitter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {
using B = msl::BinOp;

template <typename T> T definingOp(Value v) {
  return dyn_cast_or_null<T>(v.getDefiningOp());
}
} // namespace

Value peelBroadcast(Value v);
std::optional<DirectStage> matchTilePointer(Value ptr, int64_t rows,
                                            int64_t cols);
std::optional<DirectStage> matchDirectStage(Value operand, int64_t rows,
                                            int64_t cols);

// The DMA form of an async copy: the pointer tensor must denote a contiguous
// device tile whose destination is a plain unpadded row-major threadgroup
// buffer, which is exactly what the pipeliner allocates.

// Shared by the copy itself and the sibling walk, so the two cannot disagree.
#define DMA_NO(why)                                                            \
  do {                                                                         \
    if (getenv("TRITON_MSL_DMA_PROBE"))                                        \
      llvm::errs() << "[dma-elig] no: " << (why) << "\n";                      \
    return false;                                                              \
  } while (0)

bool MSLEmitter::dmaCopyEligible(ttg::AsyncCopyGlobalToLocalOp ac) {
  // The intrinsic takes no predicate, so a per-element mask would read past a
  // ragged operand; a uniform one just guards the whole call.
  if (Value m = ac.getMask())
    if (!definingOp<tt::SplatOp>(peelBroadcast(m)))
      DMA_NO("masked");
  auto srcTy = dyn_cast<RankedTensorType>(ac.getSrc().getType());
  auto mt = dyn_cast<ttg::MemDescType>(ac.getResult().getType());
  if (!srcTy || !mt || srcTy.getRank() != 2 || mt.getRank() != 2)
    DMA_NO("rank");
  Type e = mt.getElementType();
  if (!(e.isF32() || e.isF16() || e.isBF16()))
    DMA_NO("dtype");
  // A non-row-major destination cannot be expressed as (dstStride, dims).
  if (!memdescStrides(mt).empty())
    DMA_NO("dst-not-rowmajor");
  // Asynchronous, so one buffer lets trip N+1's request overwrite the tile trip
  // N is still reading -- the register path survives that only by being
  // synchronous and barrier-ordered.
  Value buf = ac.getResult();
  while (auto idx = buf.getDefiningOp<ttg::MemDescIndexOp>())
    buf = idx.getSrc();
  auto bt = dyn_cast<ttg::MemDescType>(buf.getType());
  if (!bt || bt.getRank() < 3 || bt.getShape()[0] < 2)
    DMA_NO("single-buffered");
  int64_t rows = mt.getShape()[0], cols = mt.getShape()[1];
  if (!matchTilePointer(ac.getSrc(), rows, cols).has_value())
    DMA_NO("tile-pointer");
  return true;
}

int64_t moduleNumWarps(Operation *o) {
  if (auto mod = o->getParentOfType<ModuleOp>())
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      return a.getInt();
  return 1;
}

tt::DotOp MSLEmitter::dmaCopyConsumerB(ttg::AsyncCopyGlobalToLocalOp ac) {
  auto mt = dyn_cast<ttg::MemDescType>(ac.getResult().getType());
  if (!mt || mt.getRank() != 2)
    return nullptr;
  Value buf = ac.getResult();
  while (auto idx = buf.getDefiningOp<ttg::MemDescIndexOp>())
    buf = idx.getSrc();
  Operation *scope = buf.getDefiningOp();
  if (!scope)
    return nullptr;
  scope = scope->getParentOp();
  if (!scope)
    return nullptr;
  int64_t K = mt.getShape()[0], N = mt.getShape()[1];
  tt::DotOp found = nullptr;
  bool ambiguous = false;
  scope->walk([&](tt::DotOp dot) {
    if (ambiguous)
      return;
    auto ll = dotOperandLocalLoad(dot.getB(), K, N);
    if (!ll)
      return;
    Value src = ll.getSrc();
    while (auto idx = src.getDefiningOp<ttg::MemDescIndexOp>())
      src = idx.getSrc();
    if (src != buf)
      return;
    if (found)
      ambiguous = true;
    found = dot;
  });
  return ambiguous ? nullptr : found;
}

ttg::AsyncCopyGlobalToLocalOp MSLEmitter::dotBFillCopy(tt::DotOp op, int64_t K,
                                                       int64_t N) {
  auto ll = dotOperandLocalLoad(op.getB(), K, N);
  if (!ll)
    return nullptr;
  Value buf = ll.getSrc();
  while (auto idx = buf.getDefiningOp<ttg::MemDescIndexOp>())
    buf = idx.getSrc();
  Operation *scope = buf.getDefiningOp();
  if (!scope)
    return nullptr;
  scope = scope->getParentOp();
  if (!scope)
    return nullptr;
  ttg::AsyncCopyGlobalToLocalOp found = nullptr;
  bool ambiguous = false;
  scope->walk([&](ttg::AsyncCopyGlobalToLocalOp ac) {
    if (ambiguous)
      return;
    Value dst = ac.getResult();
    while (auto idx = dst.getDefiningOp<ttg::MemDescIndexOp>())
      dst = idx.getSrc();
    if (dst != buf)
      return;
    auto mt = dyn_cast<ttg::MemDescType>(ac.getResult().getType());
    if (!mt || mt.getRank() != 2 || mt.getShape()[0] != K ||
        mt.getShape()[1] != N)
      ambiguous = true;
    if (!found)
      found = ac;
  });
  return ambiguous ? nullptr : found;
}

bool MSLEmitter::dmaStoreTransposed(ttg::AsyncCopyGlobalToLocalOp ac) {
  Value buf = ac.getResult();
  while (auto idx = buf.getDefiningOp<ttg::MemDescIndexOp>())
    buf = idx.getSrc();
  Operation *key = buf.getDefiningOp();
  if (!key)
    return false;
  auto cached = dmaStoreTrCache.find(key);
  if (cached != dmaStoreTrCache.end())
    return cached->second;
  bool r = computeDmaStoreTransposed(ac);
  dmaStoreTrCache[key] = r;
  return r;
}

bool MSLEmitter::computeDmaStoreTransposed(ttg::AsyncCopyGlobalToLocalOp ac) {
  auto mt = dyn_cast<ttg::MemDescType>(ac.getResult().getType());
  if (!mt || mt.getRank() != 2)
    return false;
  int64_t K = mt.getShape()[0], N = mt.getShape()[1];
  if (!(K % 8 == 0 && N % 8 == 0 && moduleNumWarps(ac) > 0 &&
        N % moduleNumWarps(ac) == 0))
    return false;

  Value buf = ac.getResult();
  while (auto idx = buf.getDefiningOp<ttg::MemDescIndexOp>())
    buf = idx.getSrc();
  Operation *scope = buf.getDefiningOp();
  if (!scope || !scope->getParentOp())
    return false;

  tt::DotOp dot = nullptr;
  bool ok = true;
  // This decides the buffer's shape, so it must answer the same way whether it
  // is asked while planning the pool or while emitting the copy. Whether a
  // shift already has an MSL name is a property of emission order, not of the
  // tile, so the binding test is not applied here.
  scope->getParentOp()->walk([&](ttg::AsyncCopyGlobalToLocalOp other) {
    if (!ok)
      return;
    Value dst = other.getResult();
    while (auto idx = dst.getDefiningOp<ttg::MemDescIndexOp>())
      dst = idx.getSrc();
    if (dst != buf)
      return;
    auto ds = asyncCopyDma(other, /*requireBound=*/false);
    if (!ds || !ds->srcTransposed) {
      ok = false;
      return;
    }
    tt::DotOp d = dmaCopyConsumerB(other);
    if (!d || (dot && d != dot)) {
      ok = false;
      return;
    }
    dot = d;
  });
  return ok && dot != nullptr;
}

// The pipeline stage a copy fills, as the index of the loop-carried token it
// ultimately feeds. A peeled prologue copy commits into a for-loop init
// operand; an in-loop copy commits into a yield operand. Copies sharing an
// index are the same stage and share one event token; -1 is "no loop", which
// makes every such copy its own stage.
int MSLEmitter::dmaStageSlot(ttg::AsyncCopyGlobalToLocalOp ac) {
  Value tok;
  for (Operation *u : ac.getToken().getUsers())
    if (auto commit = dyn_cast<ttg::AsyncCommitGroupOp>(u))
      tok = commit.getAsyncToken();
  if (!tok)
    return -1;
  for (OpOperand &use : tok.getUses()) {
    Operation *owner = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
      int idx =
          (int)use.getOperandNumber() - (int)forOp.getNumControlOperands();
      if (idx >= 0)
        return idx;
    }
    if (isa<scf::YieldOp>(owner))
      return (int)use.getOperandNumber();
  }
  return -1;
}

std::optional<DirectStage>
MSLEmitter::asyncCopyDma(ttg::AsyncCopyGlobalToLocalOp ac, bool requireBound) {
  if (!dmaStagingEnabled() || !dmaCopyEligible(ac))
    return std::nullopt;
  auto mt = cast<ttg::MemDescType>(ac.getResult().getType());
  auto ds = matchTilePointer(ac.getSrc(), mt.getShape()[0], mt.getShape()[1]);
  if (!ds)
    return std::nullopt;
  // All-or-nothing per loop: staging one operand by DMA and the other through
  // registers mixes two layouts for the same MMA block and computes garbage.
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(ac->getParentOp())) {
    bool allOk = true;
    forOp.walk([&](ttg::AsyncCopyGlobalToLocalOp other) {
      if (other != ac && allOk && !dmaCopyEligible(other))
        allOk = false;
    });
    if (!allOk)
      return std::nullopt;
  }
  auto bound = [&](Value v) {
    if (!v)
      return true;
    auto it = valMap.find(v);
    return it != valMap.end() && it->second.size() == 1;
  };
  if (requireBound &&
      (!bound(ds->basePtr) || !(ds->rowStrideLit || bound(ds->rowStride)) ||
       !bound(ds->rowShift) || !bound(ds->colShift)))
    return std::nullopt;
  return ds;
}

// True when device-direct operand staging is enabled. Off unless the
// environment asks for it: the DMA path changes how operands reach the MMA and
// async copy has historically produced silent wrong answers rather than
// crashes, so it must be opt-in until proven on a wider body of kernels.
bool MSLEmitter::dmaStagingEnabled() {
  static const bool on = getenv("TRITON_MSL_DMA_STAGE") != nullptr;
  return on;
}

// The single decision point for device-direct staging, so the preamble scan and
// the dot emitter cannot disagree about whether the shim is referenced.
// Only the B operand is considered: add_prefetch_dot_operand has already
// rotated A into a loop-carried value with no load left to match.
// Shape/element/pointer test for device-direct B staging. Deliberately free of
// any planDot call: planDot itself needs this to size the pool, so the two
// would otherwise recurse.
std::optional<DirectStage> MSLEmitter::bDmaCandidate(tt::DotOp op,
                                                     bool requireBound) {
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
  if (!aTy || !bTy || aTy.getRank() != 2 || bTy.getRank() != 2)
    return std::nullopt;
  Type be = bTy.getElementType();
  if (!(be.isF32() || be.isF16() || be.isBF16()))
    return std::nullopt;
  Value stage = op.getB();
  if (Value s = dotOperandConvertSource(op, op.getB()))
    stage = s;
  auto ds = matchDirectStage(stage, bTy.getShape()[0], bTy.getShape()[1]);
  if (!ds)
    return std::nullopt;
  if (!rowBoundNeverRagged(*ds, bTy.getShape()[0]))
    return std::nullopt;
  // The copy spells the base pointer and the strides as MSL scalars, so every
  // one of them must already be bound. A shift defined inside the K-loop (or
  // otherwise not yet emitted) has no name here.
  auto bound = [&](Value v) {
    if (!v)
      return true;
    auto it = valMap.find(v);
    return it != valMap.end() && it->second.size() == 1;
  };
  if (requireBound &&
      (!bound(ds->basePtr) || !(ds->rowStrideLit || bound(ds->rowStride)) ||
       !bound(ds->rowShift) || !bound(ds->colShift)))
    return std::nullopt;
  return ds;
}

bool MSLEmitter::aDirectEnabled() { return true; }

// A produced in #mma already sits in simdgroup_matrix storage: register 2*f+e
// is fragment f's thread_elements()[e]. Past mT == numWarps the layout
// replicates instead of partitioning, and the missing rows are in no lane's
// registers at all.
bool MSLEmitter::aFragEligible(tt::DotOp op, const DotPlan &plan) {
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  if (!aTy || aTy.getRank() != 2 || !aTy.getElementType().isF32())
    return false;
  // Every test below passes for a transposed A, which would then be read in the
  // untransposed order. B peels its transpose (bStageTransposed) and
  // aDirectCandidate declines one; this path had neither.
  if (dotOperandTransposed(op, op.getA()))
    return false;
  auto cTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!cTy || aTy.getEncoding() != cTy.getEncoding())
    return false;
  if (plan.K % 8 || plan.M != aTy.getShape()[0] || plan.K != aTy.getShape()[1])
    return false;
  if (plan.mT != plan.numWarps)
    return false;
  auto ll = ttg::toLinearLayout(aTy);
  auto kReg = StringAttr::get(op.getContext(), "register");
  if (ll.getInDimSize(kReg) != 2 * (plan.K / 8))
    return false;
  auto outs = llvm::to_vector(ll.getOutDimNames());
  return ll.getBasis(kReg, 0, outs[0]) == 0 &&
         ll.getBasis(kReg, 0, outs[1]) == 1;
}

// A's fragments can be simdgroup_load-ed straight off device memory when the
// operand denotes an unmasked row-major device tile and the warp partition
// splits along M alone. The second condition is what makes A warp-private: warp
// w touches only rows [w*(M/numWarps), (w+1)*(M/numWarps)), so no warp ever
// reads a row another warp would have had to publish, and the tile needs
// neither staging nor a rendezvous.
//
// A masked load is rejected inside matchDirectStage: a ragged M tile would read
// off the end of A, which is a silent out-of-bounds read rather than a fault,
// so boundary tiles keep the staged path.
//
// The transposed form is declined: it computes correctly, but a column-major A
// has no reuse along the row axis and is slower than staging it.
std::optional<DirectStage> MSLEmitter::aDirectCandidate(tt::DotOp op, int64_t M,
                                                        int64_t K,
                                                        int64_t numWarps,
                                                        bool requireBound) {
  if (!aDirectEnabled())
    return std::nullopt;
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  if (!aTy || aTy.getRank() != 2)
    return std::nullopt;
  Type ae = aTy.getElementType();
  if (!(ae.isF32() || ae.isF16() || ae.isBF16()))
    return std::nullopt;
  // Warps must tile M alone, in equal whole-fragment bands.
  if (numWarps < 1 || M % (numWarps * 8))
    return std::nullopt;
  if (dotOperandTransposed(op, op.getA()))
    return std::nullopt;
  Value stage = op.getA();
  if (Value s = dotOperandConvertSource(op, op.getA()))
    stage = s;
  auto ds = matchDirectStage(stage, M, K);
  if (!ds || ds->srcTransposed)
    return std::nullopt;
  if (!rowBoundNeverRagged(*ds, M))
    return std::nullopt;
  auto bound = [&](Value v) {
    if (!v)
      return true;
    auto it = valMap.find(v);
    return it != valMap.end() && it->second.size() == 1;
  };
  if (requireBound &&
      (!bound(ds->basePtr) || !(ds->rowStrideLit || bound(ds->rowStride)) ||
       !bound(ds->rowShift) || !bound(ds->colShift)))
    return std::nullopt;
  return ds;
}

// The direct path addresses A by absolute fragment row, so every warp must own
// a whole number of fragment rows and read only those; a warp that spans the
// whole tile would re-read every row once per warp. The fused path partitions
// A the same way through planWarpTiling, which handles the rest.
std::optional<DirectStage> MSLEmitter::dotADirect(tt::DotOp op,
                                                  bool knownFusedAcc) {
  auto cTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
  if (!cTy || !aTy)
    return std::nullopt;
  int rk = cTy.getRank();
  int64_t M = cTy.getShape()[rk - 2], N = cTy.getShape()[rk - 1];
  int64_t K = aTy.getShape()[aTy.getRank() - 1];
  if (M % 8 || N % 8)
    return std::nullopt;
  tt::LinearLayout ll = ttg::toLinearLayout(cTy);
  auto wd = StringAttr::get(op.getContext(), "warp");
  int64_t nw = ll.hasInDim(wd) ? ll.getInDimSize(wd) : 1;
  int64_t mT = M / 8, nT = N / 8, nFrag = mT * nT;
  if (nw > nFrag)
    nw = nFrag;
  bool fusedAcc = knownFusedAcc || dotIsFusedGemmAcc(op);
  if (!fusedAcc && !(rk == 2 && nw > 0 && mT % nw == 0))
    return std::nullopt;
  // When a warp spans every ni, staging is the better arm as long as it can
  // hold A, B and C at once. Past the cap the staged path loses its warp
  // tiling and falls back to recomputing the whole tile in every warp, which
  // costs far more than the direct band's extra loads.
  if (!fusedAcc && nT > nw) {
    auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
    if (!bTy)
      return std::nullopt;
    int64_t aw = byteWidth(aTy.getElementType());
    int64_t bw = byteWidth(bTy.getElementType());
    int64_t staged = M * K * aw + K * N * bw + M * N * 4;
    if (staged <= kTGResidentBudgetBytes)
      return std::nullopt;
  }
  return aDirectCandidate(op, M, K, nw, /*requireBound=*/false);
}

msl::Stmt *MSLEmitter::bFragLoad(const DotEmitCtx &dc, int64_t ki,
                                 msl::Expr *niExpr, int64_t niLit, StringRef fb,
                                 int64_t ldb, msl::Expr *kiOff) {
  // Staged pre-transpose, B's tile is N x K: fragment (ki, ni) starts at row
  // ni, column ki, and the flag swaps the axes on the way into the fragment.
  if (dc.bStageTransposed) {
    int64_t ld = dc.bStageLd;
    msl::Expr *kTerm = kiOff ? kiOff : (msl::Expr *)ctx.i32lit(ki * 8);
    if (niExpr)
      return ctx.exprStmt(ctx.call(
          msl::builtin::sg::Load,
          {ctx.var(fb),
           ctx.binary(B::Add, ctx.var(dc.tgBCur),
                      ctx.paren(ctx.add(
                          kTerm, ctx.mul(niExpr, ctx.i32lit(8 * ld))))),
           ctx.i32lit(ld), ctx.raw("ulong2(0, 0)"), ctx.lit("true")}));
    if (kiOff)
      return ctx.exprStmt(ctx.call(
          msl::builtin::sg::Load,
          {ctx.var(fb),
           ctx.binary(B::Add, ctx.var(dc.tgBCur),
                      ctx.paren(ctx.add(kTerm, ctx.i32lit(niLit * 8 * ld)))),
           ctx.i32lit(ld), ctx.raw("ulong2(0, 0)"), ctx.lit("true")}));
    return sgLoad(fb, dc.tgBCur, niLit * 8 * ld + ki * 8, ld,
                  /*transpose=*/true);
  }
  msl::Expr *kTerm = kiOff ? (msl::Expr *)ctx.paren(
                                 ctx.mul(kiOff, ctx.i32lit(ldb)))
                           : (msl::Expr *)ctx.i32lit(ki * 8 * ldb);
  if (niExpr)
    return ctx.exprStmt(ctx.call(
        msl::builtin::sg::Load,
        {ctx.var(fb),
         ctx.binary(B::Add, ctx.var(dc.tgBCur),
                    ctx.paren(ctx.add(kTerm, ctx.mul(niExpr, ctx.i32lit(8))))),
         ctx.i32lit(ldb)}));
  if (kiOff)
    return ctx.exprStmt(ctx.call(
        msl::builtin::sg::Load,
        {ctx.var(fb),
         ctx.binary(B::Add, ctx.var(dc.tgBCur),
                    ctx.paren(ctx.add(kTerm, ctx.i32lit(niLit * 8)))),
         ctx.i32lit(ldb)}));
  return sgLoad(fb, dc.tgBCur, ki * 8 * ldb + niLit * 8, ldb);
}

std::optional<DirectStage> MSLEmitter::dotDmaStage(tt::DotOp op) {
  if (!dmaStagingEnabled())
    return std::nullopt;
  // Only the fused path emits the copy. The panel and direct paths still read
  // the operand's per-register names, which suppressing the load would leave
  // unbound.
  DotPlan p = planDot(op);
  if (p.kind != DotPlan::Kind::Fused || p.bInPlace)
    return std::nullopt;
  return bDmaCandidate(op);
}

// The K-loop induction variable, as an MSL name, when this dot sits directly in
// one. The recognised operand pointer advances by `ptrDelta` per trip, and the
// loop IV counts K elements per trip, so the origin's offset is
// (iv / step) * ptrDelta. Empty when the dot is not in a loop, in which case
// the tile origin has no per-trip term.
StringRef MSLEmitter::dotDmaTripVar(Operation *op) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(op->getParentOp());
  if (!forOp)
    return {};
  auto it = valMap.find(forOp.getInductionVar());
  if (it == valMap.end() || it->second.empty())
    return {};
  return it->second[0];
}

// The row pitch as an expression, whichever form the matcher recovered. Every
// site that spells the stride goes through here so the two forms cannot
// diverge.
msl::Expr *MSLEmitter::dmaRowStride(const DirectStage &ds) {
  if (ds.rowStrideLit)
    return ctx.i32lit(*ds.rowStrideLit);
  return ctx.var(scalarName(ds.rowStride));
}

// The tile origin: base + (rowShift*rowStride + colShift) elements, plus the
// per-trip delta accumulated by the K-loop. Every term is a uniform scalar, so
// the result is threadgroup-uniform -- which the DMA requires.
msl::Expr *MSLEmitter::dmaTileOrigin(const DirectStage &ds, StringRef tripVar) {
  msl::Expr *off = nullptr;
  auto add = [&](msl::Expr *e) { off = off ? ctx.binary(B::Add, off, e) : e; };
  // rowStride is the pitch of whichever axis is strided; the other axis is
  // contiguous and shifts by one element. Transposing swaps which is which.
  Value pitched = ds.srcTransposed ? ds.colShift : ds.rowShift;
  Value unitAxis = ds.srcTransposed ? ds.rowShift : ds.colShift;
  auto pitchedLit = ds.srcTransposed ? ds.colShiftLit : ds.rowShiftLit;
  auto unitLit = ds.srcTransposed ? ds.rowShiftLit : ds.colShiftLit;
  if (pitched)
    add(ctx.paren(
        ctx.binary(B::Mul, ctx.var(scalarName(pitched)), dmaRowStride(ds))));
  if (pitchedLit && *pitchedLit)
    add(ctx.paren(
        ctx.binary(B::Mul, ctx.i32lit(*pitchedLit), dmaRowStride(ds))));
  if (unitAxis)
    add(ctx.var(scalarName(unitAxis)));
  if (unitLit && *unitLit)
    add(ctx.i32lit(*unitLit));
  // The K-loop IV counts K elements, and the operand advances one row per K
  // element, so the per-trip origin offset is exactly iv*rowStride -- no need
  // to divide the IV by the loop step and rescale by ptrDelta.
  // The IV counts K elements and the tile advances `delta` elements per trip of
  // `rows` (B) or `cols` (A) K-elements, so the origin term is
  // iv * delta / step. A scalar delta is B's row stride and the step is `rows`,
  // giving iv*stride; a literal delta is A's K-block and divides out exactly.
  if (!tripVar.empty()) {
    // `iv` counts K elements; a copy running `aheadSteps` blocks ahead adds
    // that many K-blocks on top.
    if (ds.ptrDeltaLit) {
      int64_t step = ds.cols; // advances along its columns
      if (step > 0 && *ds.ptrDeltaLit % step == 0) {
        int64_t perK = *ds.ptrDeltaLit / step;
        msl::Expr *t = ctx.var(tripVar);
        if (ds.aheadSteps)
          t = ctx.paren(
              ctx.binary(B::Add, t, ctx.i32lit(ds.aheadSteps * ds.cols)));
        if (perK != 1)
          t = ctx.paren(ctx.binary(B::Mul, t, ctx.i32lit(perK)));
        add(ctx.paren(t));
      }
    } else if (ds.ptrDelta) {
      msl::Expr *t = ctx.var(tripVar);
      if (ds.aheadSteps)
        t = ctx.paren(
            ctx.binary(B::Add, t, ctx.i32lit(ds.aheadSteps * ds.rows)));
      // The IV steps down the tile's rows, which are contiguous when the source
      // is transposed.
      add(ds.srcTransposed
              ? ctx.paren(t)
              : ctx.paren(ctx.binary(B::Mul, t, dmaRowStride(ds))));
    }
  }
  msl::Expr *base = ctx.var(scalarName(ds.basePtr));
  return off ? ctx.binary(B::Add, base, ctx.paren(off)) : base;
}

// The shim entry point for an element size, in the orientation the source tile
// needs. One entry point per element size because the intrinsic silently copies
// garbage when its element size is not a literal constant; the `_tr` forms swap
// the source strides so a column-major tile still lands row-major.
std::string MSLEmitter::dmaCallee(int64_t elemBytes, bool transposed) {
  return "__triton_tg_async_copy_begin_" + std::to_string(elemBytes) +
         (transposed ? "_tr" : "");
}

// `ulong h = __triton_tg_async_copy_begin_<n>(tg, pitch, src, stride, rows,
// cols);` -- one entry point per element size, because the intrinsic silently
// copies garbage when its element size is not a literal constant.
msl::Stmt *MSLEmitter::dmaBegin(StringRef handle, StringRef tgBuf,
                                int64_t pitch, msl::Expr *src,
                                const DirectStage &ds, int64_t elemBytes) {
  std::string callee = dmaCallee(elemBytes, ds.srcTransposed);
  msl::Expr *c = ctx.call(callee, {ctx.var(tgBuf), ctx.i32lit(pitch), src,
                                   dmaRowStride(ds), ctx.i32lit(ds.rows),
                                   ctx.i32lit(ds.cols)});
  return ctx.declStmt(ctx.named("ulong"), handle, c);
}

msl::Stmt *MSLEmitter::dmaWait(StringRef handle) {
  return ctx.exprStmt(
      ctx.call("__triton_tg_async_copy_wait", {ctx.var(handle)}));
}

// `h = begin(tgBbase + parity*stagedB, ...)` for the trip after the current
// one. The destination is the tile the current MMAs are not reading, selected
// by the parity flag the caller has already flipped.
msl::Stmt *MSLEmitter::dmaBeginInto(StringRef handle, const DotPlan &plan,
                                    const DotEmitCtx &dc, const DirectStage &ds,
                                    RankedTensorType bStageTy, int64_t ldb,
                                    StringRef tripVar, bool nextTrip) {
  int64_t eb = byteWidth(bStageTy.getElementType());
  // The parity names the tile this trip READS. The priming copy fills that
  // tile; the in-loop copy fills the other one, for the next trip to read
  // after the top-of-trip flip.
  msl::Expr *slot = nextTrip
                        ? ctx.paren(ctx.binary(B::Sub, ctx.i32lit(1),
                                               ctx.var(fusedDot.dmaParity)))
                        : static_cast<msl::Expr *>(ctx.var(fusedDot.dmaParity));
  // air.simdgroup_async_copy_2d is issued per simdgroup, so letting every warp
  // request the whole tile would move it numWarps times over. Split it instead:
  // warp w takes the band [w*band, (w+1)*band).
  //
  // `_tr` swaps the source strides, so it also swaps which axis the device
  // walks contiguously. Banding the contiguous axis chops every warp's run into
  // a scalar strided gather, so a transposed source must be banded on the
  // opposite axis.
  int64_t nw = plan.numWarps > 0 ? plan.numWarps : 1;
  bool bandCols = ds.srcTransposed;
  int64_t axis = bandCols ? ds.cols : ds.rows;
  int64_t band = axis / nw;
  bool split = band > 0 && axis % nw == 0;
  if (!split) {
    band = axis;
    nw = 1;
  }
  msl::Expr *dst = ctx.paren(ctx.binary(
      B::Add, ctx.var(dc.tgB),
      ctx.paren(ctx.binary(B::Mul, slot, ctx.i32lit(plan.stagedB / eb)))));
  // The next trip's tile starts one K-block further down B's rows.
  msl::Expr *src = dmaTileOrigin(ds, tripVar);
  if (nextTrip)
    src = ctx.binary(B::Add, src,
                     ds.srcTransposed
                         ? static_cast<msl::Expr *>(ctx.i32lit(ds.rows))
                         : ctx.paren(ctx.binary(B::Mul, ctx.i32lit(ds.rows),
                                                dmaRowStride(ds))));
  if (split) {
    msl::Expr *wOff =
        ctx.paren(ctx.binary(B::Mul, ctx.var(warpId), ctx.i32lit(band)));
    dst = ctx.paren(ctx.binary(
        B::Add, dst,
        bandCols ? wOff
                 : ctx.paren(ctx.binary(B::Mul, wOff, ctx.i32lit(ldb)))));
    src = ctx.binary(B::Add, src,
                     ctx.paren(ctx.binary(B::Mul, wOff, dmaRowStride(ds))));
  }
  msl::Expr *c = ctx.call(dmaCallee(eb, ds.srcTransposed),
                          {dst, ctx.i32lit(ldb), src, dmaRowStride(ds),
                           ctx.i32lit(bandCols ? ds.rows : band),
                           ctx.i32lit(bandCols ? band : ds.cols)});
  return ctx.assignStmt(ctx.var(handle), c);
}

} // namespace mlir::triton::applegpu
