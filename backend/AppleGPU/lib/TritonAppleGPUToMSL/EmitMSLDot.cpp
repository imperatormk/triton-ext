// EmitMSLDot.cpp - dot/GEMM simdgroup-matrix + fp-narrowing lowering.
//
// AST builders for the simdgroup-matrix fragment MMA (emitDot /
// emitDotScalar / emitFusedGemm), plus the fp_to_fp narrowing helpers.
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.
// The fp-narrowing bit-twiddling body is the design-sanctioned Raw escape
// hatch: the builder wraps that block in one RawStmt and
// keeps only the outer `sc h = as_type<sc>(bits);` as real nodes. The fp
// helpers must never touch nextId/indent, so they reconstruct their block
// through a private id/indent local seeded from the emitter's current state
// without mutating it.

#include "MSLConstants.h"
#include "MSLEmitter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {
using B = msl::BinOp;
using CS = msl::Cast::Style;

template <typename T> T definingOp(Value v) {
  return dyn_cast_or_null<T>(v.getDefiningOp());
}

// Per-warp ownership of the (mT x nT) accumulator grid.
//
// The 1D form hands warp w the fragments w, w+numWarps, ... in row-major order,
// so at numWarps==nT a warp owns one B column and fragsPerWarp distinct A rows:
// fragsPerWarp+1 simdgroup_loads per k-step. The 2D form instead gives the warp
// a (miCount x niCount) subblock, costing miCount+niCount loads for the same
// fragsPerWarp MMAs.
//
// Slot order within a warp is row-major over that subblock, and every consumer
// of accNames[j] must agree on it - MMA and both readback paths derive (mi, ni)
// from this one place.
struct WarpTiling {
  int64_t miCount = 1, niCount = 1;  // subblock shape owned by one warp
  int64_t wGridN = 1;                // warps across the N axis
  bool twoD = false;

  int64_t slots() const { return miCount * niCount; }

  // Fragment coordinates of warp w's slot j.
  void frag(int64_t w, int64_t j, int64_t nT, int64_t numWarps, int64_t &mi,
            int64_t &ni) const {
    if (!twoD) {
      int64_t f = w + j * numWarps;
      mi = f / nT;
      ni = f % nT;
      return;
    }
    mi = (w / wGridN) * miCount + j / niCount;
    ni = (w % wGridN) * niCount + j % niCount;
  }
};

// Factor fragsPerWarp into a near-square (miCount x niCount) that tiles the
// (mT x nT) grid exactly with numWarps warps. Returns a 1D tiling when no such
// split exists (prime fragsPerWarp, ragged grid, partial last warp).
WarpTiling planWarpTiling(int64_t mT, int64_t nT, int64_t numWarps,
                          int64_t nFrag, int64_t fragsPerWarp) {
  WarpTiling t;
  if (numWarps < 1 || fragsPerWarp < 2)
    return t;
  // Every warp must be full and the grid exactly covered.
  if (mT * nT != nFrag || numWarps * fragsPerWarp != nFrag)
    return t;

  int64_t best = 0;
  for (int64_t ni = 1; ni <= fragsPerWarp; ++ni) {
    if (fragsPerWarp % ni)
      continue;
    int64_t mi = fragsPerWarp / ni;
    if (mT % mi || nT % ni)
      continue;
    if ((mT / mi) * (nT / ni) != numWarps)
      continue;
    // Prefer the split minimising loads (mi+ni), tie-break toward wider ni.
    if (!best || mi + ni < best) {
      best = mi + ni;
      t.miCount = mi;
      t.niCount = ni;
      t.wGridN = nT / ni;
      t.twoD = true;
    }
  }
  return t;
}
} // namespace

// Stage one dot operand's registers into its threadgroup pool:
// `tg[sliceFlatOffset] = name[r];` per register, optionally batch-guarded.
// `skip` short-circuits (operand is already in-place). `guard(r)` returns the
// per-register batch condition or nullptr (unguarded).
void MSLEmitter::stageOperand(msl::Block &body, StringRef tgName, Value stage,
                              RankedTensorType stageTy,
                              ArrayRef<std::string> names, bool skip,
                              llvm::function_ref<msl::Expr *(int)> guard,
                              int64_t rowPad) {
  if (skip)
    return;
  for (int r = 0, n = regCount(stage); r < n; ++r) {
    msl::Stmt *asn = ctx.assignStmt(
        ctx.subscript(ctx.var(tgName),
                      layout.sliceFlatOffset(stageTy, r, rowPad)),
        ctx.var(names[r]));
    msl::Expr *g = guard ? guard(r) : nullptr;
    body.push_back(g ? ctx.compactIf(g, asn) : asn);
  }
}

// Strategy + budget arithmetic for one tt.dot. Pure: reads the op, the layout
// and the emitter's pool/memdesc state, and touches nothing else. Kind is
// decided in the same order emitDot used to fall through its guards:
// integer operands -> Scalar, unsupported shape/type -> Unsupported, A+B alone
// over budget -> Panel, an active fused-GEMM phase -> Fused, else Direct.
DotPlan MSLEmitter::planDot(tt::DotOp op) {
  DotPlan p;
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  Type aElem = aTy.getElementType();
  Type bElem = bTy.getElementType();
  Type cElem = cTy.getElementType();
  if (isa<IntegerType>(aElem) || isa<IntegerType>(bElem) ||
      isa<IntegerType>(cElem)) {
    p.kind = DotPlan::Kind::Scalar;
    return p;
  }
  p.rank = aTy.getRank();
  if ((p.rank != 2 && p.rank != 3) || !isDotOperandElem(aElem) ||
      !isDotOperandElem(bElem) || aElem != bElem ||
      !(cElem.isF32() || cElem.isF16()))
    return p;
  p.Bd = p.rank == 3 ? cTy.getShape()[0] : 1;
  p.M = cTy.getShape()[p.rank - 2];
  p.N = cTy.getShape()[p.rank - 1];
  p.K = aTy.getShape()[p.rank - 1];
  if (p.M % 8 || p.N % 8 || p.K % 8)
    return p;

  int64_t aBytes = p.M * p.K * byteWidth(aElem);
  int64_t bBytes = p.N * p.K * byteWidth(bElem);
  int64_t accBytes = 4;

  // In-place aliasing is only offered when the whole A+B tile is resident, so
  // this tests the unstaged footprint against the hard 32KB cap, not the
  // (smaller) live pool budget that sizes the staged regions below.
  if (p.rank == 2 && aBytes + bBytes <= kTGResidentBudgetBytes) {
    p.aInPlace = dotOperandInPlaceBuf(op.getA(), p.M, p.K);
    p.bInPlace = dotOperandInPlaceBuf(op.getB(), p.K, p.N);
  }
  p.aStage = op.getA();
  p.bStage = op.getB();
  if (!p.aInPlace)
    if (Value s = dotOperandConvertSource(op, op.getA()))
      p.aStage = s;
  if (!p.bInPlace)
    if (Value s = dotOperandConvertSource(op, op.getB()))
      p.bStage = s;

  dotStageRowPads(p.M, p.N, p.K, byteWidth(aElem), byteWidth(bElem), p.aPad,
                  p.bPad);
  if (p.aInPlace)
    p.aPad = 0;
  if (p.bInPlace)
    p.bPad = 0;
  p.stagedA = p.aInPlace ? 0 : p.M * (p.K + p.aPad) * byteWidth(aElem);
  p.stagedB = p.bInPlace ? 0 : p.K * (p.N + p.bPad) * byteWidth(bElem);
  p.stagedAB = p.stagedA + p.stagedB;
  p.phase = fusedDot.phase;
  // The fused epilogue writes C only after the K-loop, behind a barrier, so
  // its accumulators can reuse the (dead) A/B staging instead of claiming a
  // disjoint region. Keeping C disjoint there doubles the threadgroup
  // footprint and costs residency.
  bool fusedEpilogueC = p.phase != FusedDotPhase::None;
  p.disjointC = !fusedEpilogueC &&
                p.stagedAB + p.M * p.N * accBytes <= poolBudget();
  p.bandRows = p.M;
  if (!p.disjointC && !fusedEpilogueC)
    p.bandRows = dotCBandRows(p.M, p.N, poolBudget(), accBytes);

  p.needAB = p.phase == FusedDotPhase::None || p.phase == FusedDotPhase::MMA;
  p.needC = p.phase != FusedDotPhase::Decl;

  p.mT = p.M / 8;
  p.nT = p.N / 8;
  p.kT = p.K / 8;
  p.nFrag = p.mT * p.nT;
  tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
  auto kWarpDim = StringAttr::get(op.getContext(), "warp");
  p.numWarps = cLL.hasInDim(kWarpDim) ? cLL.getInDimSize(kWarpDim) : 1;
  if (p.numWarps > p.nFrag)
    p.numWarps = p.nFrag;

  p.kind = dotNeedsPanel(p.M, p.N, p.K, byteWidth(aElem), accBytes)
               ? DotPlan::Kind::Panel
           : p.phase != FusedDotPhase::None ? DotPlan::Kind::Fused
                                            : DotPlan::Kind::Direct;
  return p;
}

// tt.dot -> simdgroup-matrix fragment MMA: panel / fused (Decl/MMA/Readback +
// direct-store) / per-dot (disjoint / aliased-banded) paths, dispatched over
// planDot's DotPlan. Guard/offset compound exprs are ctx.raw leaves; the
// per-fragment MMA + readback are the shared sg* builders.
bool MSLEmitter::emitDot(tt::DotOp op, msl::Block &body) {
  DotPlan plan = planDot(op);
  if (plan.kind == DotPlan::Kind::Scalar)
    return emitDotScalar(op, body);
  if (plan.kind == DotPlan::Kind::Unsupported)
    return false;

  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  Type aElem = aTy.getElementType();
  const int rank = plan.rank;

  DotEmitCtx dc;
  dc.cInit = names(op.getC());
  dc.opScalar = sgOperandScalar(aElem);
  dc.opFrag = sgFragType(aElem);
  dc.accFragTy = ctx.matrix(msl::MatrixType::Elem::Float);

  dc.aStage = plan.aStage;
  dc.bStage = plan.bStage;
  dc.aNames = names(dc.aStage);
  dc.bNames = names(dc.bStage);

  dotPoolPtrs(body, plan, dc);
  dc.ids = dotResultIds(body, op, plan);

  auto outNames = llvm::to_vector(ttg::toLinearLayout(cTy).getOutDimNames());
  dc.rowDim = outNames[rank - 2];
  dc.colDim = outNames[rank - 1];

  if (plan.kind == DotPlan::Kind::Panel) {
    if (!emitDotPanel(op, body, plan, dc))
      return false;
    bindRegs(op.getResult(), dc.ids);
    return true;
  }

  if (plan.kind == DotPlan::Kind::Fused) {
    auto readbackInto = [&](msl::Block &tgt, int64_t bi, int64_t r0,
                            int64_t r1) {
      dotReadback(tgt, cTy, dc.tgC, dc.ids, fusedDot.baseNames, rank, plan.N,
                  bi, r0, r1, dc.rowDim, dc.colDim);
    };
    if (!emitDotFused(op, body, plan, dc, readbackInto))
      return false;
    bindRegs(op.getResult(), dc.ids);
    return true;
  }

  if (!emitDotDirect(op, body, plan, dc))
    return false;
  bindRegs(op.getResult(), dc.ids);
  return true;
}

// Declare the three threadgroup pool pointers A/B/C. All three ids are minted
// unconditionally even when a phase suppresses the matching decl - the fused
// phases share one id numbering, so skipping a fresh() here would renumber
// every later name.
void MSLEmitter::dotPoolPtrs(msl::Block &body, const DotPlan &plan,
                             DotEmitCtx &dc) {
  auto tgPtr = [&](StringRef scalar) {
    return ctx.ptr(ctx.named(scalar), msl::AddrSpace::Threadgroup);
  };
  dc.tgA = fresh();
  dc.tgB = fresh();
  dc.tgC = fresh();
  if (plan.needAB) {
    body.push_back(ctx.declStmt(tgPtr(dc.opScalar), dc.tgA,
                                plan.aInPlace ? inPlaceBase(*plan.aInPlace)
                                              : poolRegion(0, dc.opScalar)));
    body.push_back(ctx.declStmt(tgPtr(dc.opScalar), dc.tgB,
                                plan.bInPlace
                                    ? inPlaceBase(*plan.bInPlace)
                                    : poolRegion(plan.stagedA, dc.opScalar)));
  }
  if (plan.needC)
    body.push_back(
        ctx.declStmt(tgPtr("float"), dc.tgC,
                     poolRegion(plan.disjointC ? plan.stagedAB : 0, "float")));
}

// The per-register C result names: the fused path reuses the ids minted at its
// Decl phase, every other path declares fresh zeroed accumulators.
SmallVector<std::string>
MSLEmitter::dotResultIds(msl::Block &body, tt::DotOp op, const DotPlan &plan) {
  if (plan.phase != FusedDotPhase::None)
    return fusedDot.ids;
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  msl::Type *accScalarTy = scalarType(cTy.getElementType());
  SmallVector<std::string> ids(regCount(op.getResult()));
  for (auto &id : ids) {
    id = fresh();
    body.push_back(ctx.declStmt(
        accScalarTy, id, ctx.cast(CS::CStyle, accScalarTy, ctx.lit("0"))));
  }
  return ids;
}

// Per-dot readback of one C row band: `if (guard) ids[r] = tgC[bandOff] +
// base[r];`, batch-guarded at rank 3. `base` is the C-init name list (per-dot)
// or the fused path's saved base names; a single-element list broadcasts.
void MSLEmitter::dotReadback(msl::Block &tgt, RankedTensorType cTy,
                             StringRef tgC, ArrayRef<std::string> ids,
                             ArrayRef<std::string> base, int rank, int64_t N,
                             int64_t bi, int64_t r0, int64_t r1,
                             StringAttr rowDim, StringAttr colDim) {
  for (int r = 0, nRes = ids.size(); r < nRes; ++r) {
    std::string b = base[base.size() == 1 ? 0 : r];
    // ((rowExpr - r0) * N + colExpr)
    msl::Expr *bandOff = ctx.paren(ctx.add(
        ctx.mul(
            ctx.paren(ctx.binary(B::Sub, layout.layoutCoordExpr(cTy, r, rowDim),
                                 ctx.i32lit(r0))),
            ctx.i32lit(N)),
        layout.layoutCoordExpr(cTy, r, colDim)));
    // (rowExpr >= r0 && rowExpr < r1). The row coord is an xor of layout bases,
    // so it always lands inside the row dim: a band spanning the whole dim
    // makes the guard a tautology.
    bool wholeBand = r0 == 0 && r1 >= cTy.getShape()[cTy.getRank() - 2];
    msl::Expr *guard = nullptr;
    if (!wholeBand)
      guard = ctx.paren(
          ctx.binary(B::LAnd,
                     ctx.binary(B::Ge, layout.layoutCoordExpr(cTy, r, rowDim),
                                ctx.i32lit(r0)),
                     ctx.binary(B::Lt, layout.layoutCoordExpr(cTy, r, rowDim),
                                ctx.i32lit(r1))));
    if (rank == 3) {
      msl::Expr *batchEq = ctx.binary(
          B::Eq, layout.batchCoordExpr(cTy, r), ctx.i32lit(bi));
      guard = guard ? ctx.paren(ctx.binary(B::LAnd, batchEq, guard)) : batchEq;
    }
    msl::Stmt *assign =
        ctx.assignStmt(ctx.var(ids[r]), readbackValue(tgC, bandOff, b));
    tgt.push_back(guard ? ctx.compactIfBare(guard, assign) : assign);
  }
}

// Per-dot (non-fused, non-panel) simdgroup-matrix path: per batch slice, stage
// A/B into the pool, run the fragment MMA grid, then round-trip C through tgC.
// When C is disjoint from the staged A/B the accumulators are stored and read
// back in one pass; otherwise tgC aliases A/B, so the accumulators stay live
// across a per-band store/readback ladder.
bool MSLEmitter::emitDotDirect(tt::DotOp op, msl::Block &body,
                               const DotPlan &plan, const DotEmitCtx &dc) {
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  auto aStageTy = cast<RankedTensorType>(dc.aStage.getType());
  auto bStageTy = cast<RankedTensorType>(dc.bStage.getType());
  const int rank = plan.rank;
  const int64_t Bd = plan.Bd, M = plan.M, N = plan.N, K = plan.K;
  const int64_t nT = plan.nT, kT = plan.kT;
  const int64_t nFrag = plan.nFrag, numWarps = plan.numWarps;

  auto barrier = [&] { body.push_back(ctx.hardBarrier(false)); };

  // Wrap `inner` in `if (warpId == w) { ... }`.
  auto warpIf = [&](int64_t w, msl::Block inner) {
    body.push_back(ctx.ifScope(
        ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
        std::move(inner)));
  };

  auto readback = [&](int64_t bi, int64_t r0, int64_t r1) {
    dotReadback(body, cTy, dc.tgC, dc.ids, dc.cInit, rank, N, bi, r0, r1,
                dc.rowDim, dc.colDim);
  };

  const int64_t lda = K + plan.aPad, ldb = N + plan.bPad;
  auto fragMMA = [&](int64_t mi, int64_t ni, StringRef acc, msl::Block &into) {
    for (int64_t ki = 0; ki < kT; ++ki) {
      std::string fa = fresh(), fb = fresh();
      into.push_back(fragDecl(dc.opFrag, fa));
      into.push_back(sgLoad(fa, dc.tgA, mi * 8 * lda + ki * 8, lda));
      into.push_back(fragDecl(dc.opFrag, fb));
      into.push_back(sgLoad(fb, dc.tgB, ki * 8 * ldb + ni * 8, ldb));
      into.push_back(sgMultiplyAccumulate(acc, fa, fb));
    }
  };

  // Batch guard condition `<batchCoord> == bi` (rank-3 only).
  auto batchCond = [&](RankedTensorType rt, int reg,
                       int64_t bi) -> msl::Expr * {
    return ctx.binary(B::Eq, layout.batchCoordExpr(rt, reg),
                      ctx.lit(std::to_string(bi)));
  };
  for (int64_t bi = 0; bi < Bd; ++bi) {
    barrier();
    auto guardA = [&](int r) { return batchCond(aStageTy, r, bi); };
    auto guardB = [&](int r) { return batchCond(bStageTy, r, bi); };
    stageOperand(
        body, dc.tgA, dc.aStage, aStageTy, dc.aNames, (bool)plan.aInPlace,
        rank == 3 ? llvm::function_ref<msl::Expr *(int)>(guardA) : nullptr,
        plan.aPad);
    stageOperand(
        body, dc.tgB, dc.bStage, bStageTy, dc.bNames, (bool)plan.bInPlace,
        rank == 3 ? llvm::function_ref<msl::Expr *(int)>(guardB) : nullptr,
        plan.bPad);
    barrier();

    if (plan.disjointC) {
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        for (int64_t f = w; f < nFrag; f += numWarps) {
          int64_t mi = f / nT, ni = f % nT;
          std::string acc = fresh();
          inner.push_back(accFragDecl(dc.accFragTy, acc));
          fragMMA(mi, ni, acc, inner);
          inner.push_back(sgStore(acc, dc.tgC, mi * 8 * N + ni * 8, N));
        }
        warpIf(w, std::move(inner));
      }
      barrier();
      readback(bi, 0, M);
      continue;
    }

    std::string accBase = fresh() + "_" + std::to_string(bi) + "_";
    auto accName = [&](int64_t mi, int64_t ni) {
      return accBase + std::to_string(mi) + "_" + std::to_string(ni);
    };
    for (int64_t f = 0; f < nFrag; ++f)
      body.push_back(accFragDecl(dc.accFragTy, accName(f / nT, f % nT)));
    for (int64_t w = 0; w < numWarps; ++w) {
      msl::Block inner;
      for (int64_t f = w; f < nFrag; f += numWarps)
        fragMMA(f / nT, f % nT, accName(f / nT, f % nT), inner);
      warpIf(w, std::move(inner));
    }
    for (int64_t r0 = 0; r0 < M; r0 += plan.bandRows) {
      int64_t r1 = std::min<int64_t>(r0 + plan.bandRows, M);
      barrier();
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        for (int64_t f = w; f < nFrag; f += numWarps) {
          int64_t mi = f / nT, ni = f % nT;
          if (mi * 8 < r0 || mi * 8 >= r1)
            continue;
          inner.push_back(
              sgStore(accName(mi, ni), dc.tgC, (mi * 8 - r0) * N + ni * 8, N));
        }
        warpIf(w, std::move(inner));
      }
      barrier();
      readback(bi, r0, r1);
    }
  }
  barrier();
  return true;
}

// Fused GEMM dot phases: Decl (declare/zero persistent frags), MMA (stage A/B +
// accumulate, branchless or per-warp ladder), Readback (store frags + gather,
// with optional direct-store to device C).
bool MSLEmitter::emitDotFused(
    tt::DotOp op, msl::Block &body, const DotPlan &plan, const DotEmitCtx &dc,
    llvm::function_ref<void(msl::Block &, int64_t, int64_t, int64_t)>
        readbackInto) {
  auto aStageTy = cast<RankedTensorType>(dc.aStage.getType());
  auto bStageTy = cast<RankedTensorType>(dc.bStage.getType());
  const int64_t M = plan.M, N = plan.N, K = plan.K;
  const int64_t nT = plan.nT, kT = plan.kT;
  const int64_t nFrag = plan.nFrag, numWarps = plan.numWarps;
  const int64_t lda = K + plan.aPad, ldb = N + plan.bPad;
  int64_t fragsPerWarp = (nFrag + numWarps - 1) / numWarps;
  WarpTiling wt = planWarpTiling(plan.mT, nT, numWarps, nFrag, fragsPerWarp);
  auto barrier = [&] { body.push_back(ctx.hardBarrier(false)); };

  if (fusedDot.phase == FusedDotPhase::Decl) {
    fusedDot.accNames.assign(fragsPerWarp, "");
    for (int64_t j = 0; j < fragsPerWarp; ++j) {
      std::string acc = fresh();
      fusedDot.accNames[j] = acc;
      body.push_back(accFragDecl(dc.accFragTy, acc));
    }
    return true;
  }

  if (fusedDot.phase == FusedDotPhase::MMA) {
    bool stagesHere = !plan.aInPlace || !plan.bInPlace;
    if (stagesHere)
      barrier();
    stageOperand(body, dc.tgA, dc.aStage, aStageTy, dc.aNames,
                 (bool)plan.aInPlace, nullptr, plan.aPad);
    stageOperand(body, dc.tgB, dc.bStage, bStageTy, dc.bNames,
                 (bool)plan.bInPlace, nullptr, plan.bPad);
    barrier();
    bool branchless = (numWarps == nT);
    // slots: (mi, niKey, niExpr); niKey dedups bFrag, niExpr is the typed
    // index.
    struct Slot {
      int64_t mi;
      std::string niKey;
      msl::Expr *niExpr;
    };
    auto emitSlots = [&](ArrayRef<Slot> slots, msl::Block &into) {
      for (int64_t ki = 0; ki < kT; ++ki) {
        DenseMap<int64_t, std::string> aFrag;
        std::map<std::string, std::string> bFrag;
        for (auto &pr : slots) {
          int64_t mi = pr.mi;
          const std::string &niKey = pr.niKey;
          if (!aFrag.count(mi)) {
            std::string fa = fresh();
            aFrag[mi] = fa;
            into.push_back(fragDecl(dc.opFrag, fa));
            into.push_back(sgLoad(fa, dc.tgA, mi * 8 * lda + ki * 8, lda));
          }
          if (!bFrag.count(niKey)) {
            std::string fb = fresh();
            bFrag[niKey] = fb;
            into.push_back(fragDecl(dc.opFrag, fb));
            // simdgroup_load(fb, tgB + (ki*8*ldb + niExpr * 8), ldb);
            msl::Expr *off = ctx.paren(ctx.add(
                ctx.i32lit(ki * 8 * ldb), ctx.mul(pr.niExpr, ctx.i32lit(8))));
            into.push_back(ctx.exprStmt(
                ctx.call(msl::builtin::sg::Load,
                         {ctx.var(fb), ctx.binary(B::Add, ctx.var(dc.tgB), off),
                          ctx.i32lit(ldb)})));
          }
        }
        for (auto [j, mn] : llvm::enumerate(slots)) {
          const std::string &acc = fusedDot.accNames[j];
          into.push_back(
              sgMultiplyAccumulate(acc, aFrag[mn.mi], bFrag[mn.niKey]));
        }
      }
    };
    if (wt.twoD) {
      // mi is a compile-time constant only within a warp row, so the 2D form
      // branches per warp-row and keeps ni as a warpId expression.
      int64_t wGridM = numWarps / wt.wGridN;
      // ni = (warpId % wGridN) * niCount + c
      auto niBase = [&]() -> msl::Expr * {
        msl::Expr *col =
            ctx.paren(ctx.binary(B::Rem, ctx.var(warpId), ctx.i32lit(wt.wGridN)));
        return ctx.paren(ctx.mul(col, ctx.i32lit(wt.niCount)));
      };
      std::string colKey =
          "(" + warpId + " % " + std::to_string(wt.wGridN) + ")*" +
          std::to_string(wt.niCount);
      auto emitRow = [&](int64_t wr, msl::Block &into) {
        SmallVector<Slot> slots;
        for (int64_t r = 0; r < wt.miCount; ++r)
          for (int64_t c = 0; c < wt.niCount; ++c) {
            msl::Expr *ni =
                c ? ctx.paren(ctx.add(niBase(), ctx.i32lit(c))) : niBase();
            slots.push_back({wr * wt.miCount + r,
                             colKey + "+" + std::to_string(c), ni});
          }
        emitSlots(slots, into);
      };
      if (wGridM == 1) {
        emitRow(0, body);
      } else {
        for (int64_t wr = 0; wr < wGridM; ++wr) {
          msl::Block inner;
          emitRow(wr, inner);
          body.push_back(ctx.ifScope(
              ctx.binary(B::Eq,
                         ctx.paren(ctx.binary(B::Div, ctx.var(warpId),
                                              ctx.i32lit(wt.wGridN))),
                         ctx.i32lit(wr)),
              std::move(inner)));
        }
      }
    } else if (branchless) {
      std::string niKey = "(" + warpId + " % " + std::to_string(nT) + ")";
      msl::Expr *niExpr =
          ctx.paren(ctx.binary(B::Rem, ctx.var(warpId), ctx.i32lit(nT)));
      SmallVector<Slot> slots;
      for (int64_t j = 0; j * numWarps < nFrag; ++j)
        slots.push_back({(j * numWarps) / nT, niKey, niExpr});
      emitSlots(slots, body);
    } else {
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        SmallVector<Slot> slots;
        for (int64_t f = w; f < nFrag; f += numWarps)
          slots.push_back({f / nT, std::to_string(f % nT), ctx.i32lit(f % nT)});
        emitSlots(slots, inner);
        body.push_back(ctx.ifScope(
            ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
            std::move(inner)));
      }
    }
    if (!stagesHere)
      barrier();
    bindRegs(op.getResult(), dc.ids);
    return true;
  }

  // Readback.
  if (fusedDot.direct) {
    const DirectStore &d = *fusedDot.direct;
    std::string base = scalarName(d.basePtr).str(),
                ldc = scalarName(d.ldc).str();
    std::string rowB = scalarName(d.rowBase).str(),
                colB = scalarName(d.colBase).str();
    msl::Block ifBody;
    for (int64_t w = 0; w < numWarps; ++w) {
      msl::Block inner;
      for (int64_t j = 0; j < fragsPerWarp; ++j) {
        int64_t mi, ni;
        wt.frag(w, j, nT, numWarps, mi, ni);
        if (mi * nT + ni >= nFrag)
          continue;
        // simdgroup_store(acc, base + (rowB + mi*8)*ldc + (colB + ni*8), ldc);
        msl::Expr *off = ctx.addChain(
            {ctx.var(base),
             ctx.mul(ctx.paren(ctx.add(ctx.var(rowB), ctx.i32lit(mi * 8))),
                     ctx.var(ldc)),
             ctx.paren(ctx.add(ctx.var(colB), ctx.i32lit(ni * 8)))});
        inner.push_back(ctx.exprStmt(
            ctx.call(msl::builtin::sg::Store,
                     {ctx.var(fusedDot.accNames[j]), off, ctx.var(ldc)})));
      }
      ifBody.push_back(ctx.ifScope(
          ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
          std::move(inner)));
    }
    // if (fullTileVar) { <ifBody> } else { <pool store + gather> }
    msl::Block elseBody;
    // Save `body` size to splice the pool path into elseBody instead.
    // Build pool store + gather into elseBody:
    {
      msl::Block &tgt = elseBody;
      tgt.push_back(ctx.hardBarrier(false));
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        for (int64_t j = 0; j < fragsPerWarp; ++j) {
          int64_t mi, ni;
          wt.frag(w, j, nT, numWarps, mi, ni);
          if (mi * nT + ni >= nFrag)
            continue;
          inner.push_back(
              sgStore(fusedDot.accNames[j], dc.tgC, mi * 8 * N + ni * 8, N));
        }
        tgt.push_back(ctx.ifScope(
            ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
            std::move(inner)));
      }
      tgt.push_back(ctx.hardBarrier(false));
      readbackInto(tgt, 0, 0, M);
      tgt.push_back(ctx.hardBarrier(false));
    }
    body.push_back(ctx.ifElseScope(ctx.var(d.fullTileVar), std::move(ifBody),
                                   std::move(elseBody)));
    return true;
  }

  // Non-direct readback: pool store + gather.
  barrier();
  for (int64_t w = 0; w < numWarps; ++w) {
    msl::Block inner;
    for (int64_t j = 0; j < fragsPerWarp; ++j) {
      int64_t mi, ni;
      wt.frag(w, j, nT, numWarps, mi, ni);
      if (mi * nT + ni >= nFrag)
        continue;
      inner.push_back(
          sgStore(fusedDot.accNames[j], dc.tgC, mi * 8 * N + ni * 8, N));
    }
    body.push_back(ctx.ifScope(
        ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
        std::move(inner)));
  }
  barrier();
  readbackInto(body, 0, 0, M);
  barrier();
  return true;
}

// Panel-tiled dot: A/B panels staged into the pool, MMA per (m0,n0) panel into
// pC, then a guarded readback.
bool MSLEmitter::emitDotPanel(tt::DotOp op, msl::Block &body,
                              const DotPlan &plan, const DotEmitCtx &dc) {
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  Type aElem = aTy.getElementType();
  auto aStageTy = cast<RankedTensorType>(dc.aStage.getType());
  auto bStageTy = cast<RankedTensorType>(dc.bStage.getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  const int rank = plan.rank;
  const int64_t Bd = plan.Bd, M = plan.M, N = plan.N, K = plan.K;
  const int64_t numWarps = plan.numWarps;
  int64_t elemBytes = byteWidth(aElem), accBytes = 4;
  int64_t mp, np;
  dotPanelDims(M, N, K, elemBytes, accBytes, mp, np);
  int64_t aPanelBytes = mp * K * elemBytes, bPanelBytes = K * np * elemBytes;
  auto aOut = llvm::to_vector(ttg::toLinearLayout(aStageTy).getOutDimNames());
  StringAttr aRowDim = aOut[rank - 2], aColDim = aOut[rank - 1];
  auto bOut = llvm::to_vector(ttg::toLinearLayout(bStageTy).getOutDimNames());
  StringAttr bColDim = bOut[rank - 1], bRowDim = bOut[rank - 2];
  int nRes = regCount(op.getResult());

  auto tgPtr = [&](StringRef s) {
    return ctx.ptr(ctx.named(s), msl::AddrSpace::Threadgroup);
  };
  std::string pA = fresh(), pB = fresh(), pC = fresh();
  body.push_back(
      ctx.declStmt(tgPtr(dc.opScalar), pA, poolRegion(0, dc.opScalar)));
  body.push_back(ctx.declStmt(tgPtr(dc.opScalar), pB,
                              poolRegion(aPanelBytes, dc.opScalar)));
  body.push_back(ctx.declStmt(tgPtr("float"), pC,
                              poolRegion(aPanelBytes + bPanelBytes, "float")));
  int nARegs = regCount(dc.aStage), nBRegs = regCount(dc.bStage);
  auto barrier = [&] { body.push_back(ctx.hardBarrier(false)); };

  for (int64_t bi = 0; bi < Bd; ++bi) {
    for (int64_t m0 = 0; m0 < M; m0 += mp) {
      int64_t m1 = std::min<int64_t>(m0 + mp, M), mpCur = m1 - m0;
      barrier();
      for (int r = 0; r < nARegs; ++r) {
        // (row >= m0 && row < m1)
        msl::Expr *guard = ctx.paren(ctx.binary(
            B::LAnd,
            ctx.binary(B::Ge, layout.layoutCoordExpr(aStageTy, r, aRowDim),
                       ctx.i32lit(m0)),
            ctx.binary(B::Lt, layout.layoutCoordExpr(aStageTy, r, aRowDim),
                       ctx.i32lit(m1))));
        if (rank == 3)
          guard = ctx.paren(
              ctx.binary(B::LAnd,
                         ctx.binary(B::Eq, layout.batchCoordExpr(aStageTy, r),
                                    ctx.i32lit(bi)),
                         guard));
        // ((row - m0) * K + col)
        msl::Expr *off = ctx.paren(ctx.add(
            ctx.mul(ctx.paren(ctx.binary(
                        B::Sub, layout.layoutCoordExpr(aStageTy, r, aRowDim),
                        ctx.i32lit(m0))),
                    ctx.i32lit(K)),
            layout.layoutCoordExpr(aStageTy, r, aColDim)));
        body.push_back(ctx.compactIfBare(
            guard, ctx.assignStmt(ctx.subscript(ctx.var(pA), off),
                                  ctx.var(dc.aNames[r]))));
      }
      for (int64_t n0 = 0; n0 < N; n0 += np) {
        int64_t n1 = std::min<int64_t>(n0 + np, N), npCur = n1 - n0;
        int64_t pmT = mpCur / 8, pnT = npCur / 8;
        barrier();
        for (int r = 0; r < nBRegs; ++r) {
          // (col >= n0 && col < n1)
          msl::Expr *guard = ctx.paren(ctx.binary(
              B::LAnd,
              ctx.binary(B::Ge, layout.layoutCoordExpr(bStageTy, r, bColDim),
                         ctx.i32lit(n0)),
              ctx.binary(B::Lt, layout.layoutCoordExpr(bStageTy, r, bColDim),
                         ctx.i32lit(n1))));
          if (rank == 3)
            guard = ctx.paren(
                ctx.binary(B::LAnd,
                           ctx.binary(B::Eq, layout.batchCoordExpr(bStageTy, r),
                                      ctx.i32lit(bi)),
                           guard));
          // (row * npCur + (col - n0))
          msl::Expr *off = ctx.paren(ctx.add(
              ctx.mul(layout.layoutCoordExpr(bStageTy, r, bRowDim),
                      ctx.i32lit(npCur)),
              ctx.paren(ctx.binary(B::Sub,
                                   layout.layoutCoordExpr(bStageTy, r, bColDim),
                                   ctx.i32lit(n0)))));
          body.push_back(ctx.compactIfBare(
              guard, ctx.assignStmt(ctx.subscript(ctx.var(pB), off),
                                    ctx.var(dc.bNames[r]))));
        }
        barrier();
        int64_t pnFrag = pmT * pnT;
        int64_t pWarps = numWarps > pnFrag ? pnFrag : numWarps;
        for (int64_t w = 0; w < pWarps; ++w) {
          msl::Block inner;
          for (int64_t f = w; f < pnFrag; f += pWarps) {
            int64_t mi = f / pnT, ni = f % pnT;
            std::string acc = fresh();
            inner.push_back(
                accFragDecl(ctx.matrix(msl::MatrixType::Elem::Float), acc));
            for (int64_t ki = 0; ki < (K / 8); ++ki) {
              std::string fa = fresh(), fb = fresh();
              inner.push_back(fragDecl(dc.opFrag, fa));
              inner.push_back(sgLoad(fa, pA, mi * 8 * K + ki * 8, K));
              inner.push_back(fragDecl(dc.opFrag, fb));
              inner.push_back(sgLoad(fb, pB, ki * 8 * npCur + ni * 8, npCur));
              inner.push_back(sgMultiplyAccumulate(acc, fa, fb));
            }
            inner.push_back(sgStore(acc, pC, mi * 8 * npCur + ni * 8, npCur));
          }
          body.push_back(ctx.ifScope(
              ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
              std::move(inner)));
        }
        barrier();
        for (int r = 0; r < nRes; ++r) {
          std::string base = dc.cInit[dc.cInit.size() == 1 ? 0 : r];
          // ((rowExpr - m0) * npCur + (colExpr - n0))
          msl::Expr *off = ctx.paren(ctx.add(
              ctx.mul(ctx.paren(ctx.binary(
                          B::Sub, layout.layoutCoordExpr(cTy, r, dc.rowDim),
                          ctx.i32lit(m0))),
                      ctx.i32lit(npCur)),
              ctx.paren(ctx.binary(B::Sub,
                                   layout.layoutCoordExpr(cTy, r, dc.colDim),
                                   ctx.i32lit(n0)))));
          // (rowExpr >= m0 && rowExpr < m1 && colExpr >= n0 && colExpr < n1)
          msl::Expr *guard = ctx.paren(ctx.chain(
              B::LAnd,
              {ctx.binary(B::Ge, layout.layoutCoordExpr(cTy, r, dc.rowDim),
                          ctx.i32lit(m0)),
               ctx.binary(B::Lt, layout.layoutCoordExpr(cTy, r, dc.rowDim),
                          ctx.i32lit(m1)),
               ctx.binary(B::Ge, layout.layoutCoordExpr(cTy, r, dc.colDim),
                          ctx.i32lit(n0)),
               ctx.binary(B::Lt, layout.layoutCoordExpr(cTy, r, dc.colDim),
                          ctx.i32lit(n1))}));
          if (rank == 3)
            guard = ctx.paren(
                ctx.binary(B::LAnd,
                           ctx.binary(B::Eq, layout.batchCoordExpr(cTy, r),
                                      ctx.i32lit(bi)),
                           guard));
          body.push_back(ctx.compactIfBare(
              guard,
              ctx.assignStmt(ctx.var(dc.ids[r]),
                             ctx.binary(B::Add, ctx.subscript(ctx.var(pC), off),
                                        ctx.var(base)))));
        }
      }
    }
  }
  body.push_back(ctx.hardBarrier(false));
  return true;
}

// Integer/scalar tt.dot: stage A/B into the pool, then each thread runs a
// scalar K-loop for its owned C registers.
bool MSLEmitter::emitDotScalar(tt::DotOp op, msl::Block &body) {
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  Type aElem = aTy.getElementType();
  Type bElem = bTy.getElementType();
  Type cElem = cTy.getElementType();
  int rank = aTy.getRank();
  if (rank != 2 && rank != 3)
    return false;
  int64_t Bd = rank == 3 ? cTy.getShape()[0] : 1;
  int64_t K = aTy.getShape()[rank - 1];
  int64_t N = cTy.getShape()[rank - 1];

  std::string aScalar = mslScalarType(aElem);
  std::string bScalar = mslScalarType(bElem);
  std::string accScalar = mslScalarType(cElem);
  msl::Type *accTy = scalarType(cElem);

  Value aStage = op.getA(), bStage = op.getB();
  if (rank == 2) {
    if (Value s = dotOperandConvertSource(op, op.getA()))
      aStage = s;
    if (Value s = dotOperandConvertSource(op, op.getB()))
      bStage = s;
  }
  auto aStageTy = cast<RankedTensorType>(aStage.getType());
  auto bStageTy = cast<RankedTensorType>(bStage.getType());
  auto &aNames = names(aStage);
  auto &bNames = names(bStage);
  auto &cInit = names(op.getC());
  int64_t aBytes = cTy.getShape()[rank - 2] * K * byteWidth(aElem);

  auto tgPtr = [&](StringRef s) {
    return ctx.ptr(ctx.named(s), msl::AddrSpace::Threadgroup);
  };
  std::string tgA = fresh(), tgB = fresh();
  body.push_back(ctx.declStmt(tgPtr(aScalar), tgA, poolRegion(0, aScalar)));
  body.push_back(
      ctx.declStmt(tgPtr(bScalar), tgB, poolRegion(aBytes, bScalar)));

  tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
  auto outNames = llvm::to_vector(cLL.getOutDimNames());
  StringAttr dRow = outNames[rank - 2], dCol = outNames[rank - 1];
  auto batchCond = [&](RankedTensorType rt, int reg, int64_t bi) {
    return ctx.binary(B::Eq, layout.batchCoordExpr(rt, reg),
                      ctx.lit(std::to_string(bi)));
  };

  int nRes = regCount(op.getResult());
  SmallVector<std::string> ids(nRes);
  for (int r = 0; r < nRes; ++r) {
    ids[r] = fresh();
    body.push_back(
        ctx.declStmt(accTy, ids[r], ctx.var(cInit[cInit.size() == 1 ? 0 : r])));
  }

  for (int64_t bi = 0; bi < Bd; ++bi) {
    body.push_back(ctx.hardBarrier(false));
    auto guardA = [&](int r) { return batchCond(aStageTy, r, bi); };
    auto guardB = [&](int r) { return batchCond(bStageTy, r, bi); };
    stageOperand(body, tgA, aStage, aStageTy, aNames, false,
                 rank == 3 ? llvm::function_ref<msl::Expr *(int)>(guardA)
                           : nullptr);
    stageOperand(body, tgB, bStage, bStageTy, bNames, false,
                 rank == 3 ? llvm::function_ref<msl::Expr *(int)>(guardB)
                           : nullptr);
    body.push_back(ctx.hardBarrier(false));

    for (int r = 0; r < nRes; ++r) {
      std::string mrow = fresh(), ncol = fresh(), acc = fresh(), kv = fresh();
      body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), mrow,
                                  layout.layoutCoordExpr(cTy, r, dRow)));
      body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), ncol,
                                  layout.layoutCoordExpr(cTy, r, dCol)));
      body.push_back(
          ctx.declStmt(accTy, acc, ctx.cast(CS::CStyle, accTy, ctx.lit("0"))));
      // for (int kv = 0; kv < K; ++kv) { acc += (acc)tgA[mrow*K+kv] *
      // (acc)tgB[kv*N+ncol]; }
      msl::Stmt *init =
          ctx.declStmt(ctx.scalar(msl::Scalar::I32), kv, ctx.lit("0"));
      msl::Expr *cond =
          ctx.binary(B::Lt, ctx.var(kv), ctx.lit(std::to_string(K)));
      msl::Stmt *step = ctx.exprStmt(ctx.raw("++" + kv));
      msl::Expr *aElemE =
          ctx.cast(CS::CStyle, accTy,
                   ctx.subscript(ctx.var(tgA),
                                 ctx.add(ctx.mul(ctx.var(mrow), ctx.i32lit(K)),
                                         ctx.var(kv))));
      msl::Expr *bElemE =
          ctx.cast(CS::CStyle, accTy,
                   ctx.subscript(ctx.var(tgB),
                                 ctx.add(ctx.mul(ctx.var(kv), ctx.i32lit(N)),
                                         ctx.var(ncol))));
      msl::Block loop;
      loop.push_back(
          ctx.addAssignStmt(ctx.var(acc), ctx.binary(B::Mul, aElemE, bElemE)));
      body.push_back(ctx.forScope(init, cond, step, std::move(loop)));
      msl::Stmt *accum = ctx.addAssignStmt(ctx.var(ids[r]), ctx.var(acc));
      body.push_back(rank == 3 ? ctx.compactIf(batchCond(cTy, r, bi), accum)
                               : accum);
    }
  }
  body.push_back(ctx.hardBarrier(false));
  bindRegs(op.getResult(), ids);
  return true;
}

//===----------------------------------------------------------------------===//
// simdgroup-matrix fragment MMA sub-builders
//===----------------------------------------------------------------------===//

msl::MatrixType *MSLEmitter::sgFragType(Type t) {
  if (t.isF16())
    return ctx.matrix(msl::MatrixType::Elem::Half);
  if (t.isBF16())
    return ctx.matrix(msl::MatrixType::Elem::Bfloat);
  return ctx.matrix(msl::MatrixType::Elem::Float);
}

// `base + off` (no outer paren - emitted bare inside the call).
msl::Expr *MSLEmitter::fragAddr(StringRef base, int64_t off) {
  return ctx.binary(msl::BinOp::Add, ctx.var(base),
                    ctx.lit(std::to_string(off)));
}

// `frag name;` - uninitialized operand fragment decl.
msl::Stmt *MSLEmitter::fragDecl(msl::MatrixType *frag, StringRef name) {
  return ctx.declStmt(frag, name);
}

// `frag name = frag(0.0f);` - zeroed accumulator fragment. The type name
// doubles as the ctor callee (simdgroup_float8x8(0.0f)); read the printed name
// back off the MatrixType so the call callee matches the decl type exactly.
msl::Stmt *MSLEmitter::accFragDecl(msl::MatrixType *frag, StringRef name) {
  StringRef ctorName = frag->elem == msl::MatrixType::Elem::Half
                           ? msl::builtin::sg::Half8x8
                       : frag->elem == msl::MatrixType::Elem::Bfloat
                           ? msl::builtin::sg::Bfloat8x8
                           : msl::builtin::sg::Float8x8;
  return ctx.declStmt(frag, name, ctx.call(ctorName, {ctx.lit("0.0f")}));
}

// `simdgroup_load(frag, base + off, ld);`
//
// The trailing `transpose` parameter is deliberately never passed:
// stageOperand always writes operands into threadgroup memory row-major, so
// the fragment layout is canonical by construction. A column-major B is
// transposed on the store into TG, not by flagging the load - and note Metal's
// flag means "memory is transposed relative to the matrix's canonical layout",
// so row_major maps to transpose=false.
msl::Stmt *MSLEmitter::sgLoad(StringRef frag, StringRef base, int64_t off,
                              int64_t ld) {
  return ctx.exprStmt(
      ctx.call(msl::builtin::sg::Load, {ctx.var(frag), fragAddr(base, off),
                                        ctx.lit(std::to_string(ld))}));
}

// `simdgroup_store(acc, base + off, ld);`
msl::Stmt *MSLEmitter::sgStore(StringRef acc, StringRef base, int64_t off,
                               int64_t ld) {
  return ctx.exprStmt(
      ctx.call(msl::builtin::sg::Store, {ctx.var(acc), fragAddr(base, off),
                                         ctx.lit(std::to_string(ld))}));
}

// `simdgroup_multiply_accumulate(acc, a, b, acc);`
msl::Stmt *MSLEmitter::sgMultiplyAccumulate(StringRef acc, StringRef a,
                                            StringRef b) {
  return ctx.exprStmt(
      ctx.call(msl::builtin::sg::MultiplyAccumulate,
               {ctx.var(acc), ctx.var(a), ctx.var(b), ctx.var(acc)}));
}

// C readback value: `buf[off] + base` (off already carries its outer parens).
msl::Expr *MSLEmitter::readbackValue(StringRef buf, msl::Expr *off,
                                     StringRef base) {
  return ctx.binary(msl::BinOp::Add, ctx.subscript(ctx.var(buf), off),
                    ctx.var(base));
}

// Walk back through addptr/splat/bitcast to the underlying pointer; returns the
// base BlockArgument, or null when the chain leaves kernel-arg territory.
Value MSLEmitter::traceToKernelArg(Value v) {
  while (v) {
    if (isa<BlockArgument>(v))
      return v;
    Operation *def = v.getDefiningOp();
    if (auto ap = dyn_cast_or_null<tt::AddPtrOp>(def)) {
      v = ap.getPtr();
      continue;
    }
    if (auto sp = dyn_cast_or_null<tt::SplatOp>(def)) {
      v = sp.getSrc();
      continue;
    }
    if (auto bc = dyn_cast_or_null<tt::BitcastOp>(def)) {
      v = bc.getSrc();
      continue;
    }
    return Value();
  }
  return Value();
}

bool MSLEmitter::tracesToKernelArg(Value v) {
  return static_cast<bool>(traceToKernelArg(v));
}

Value MSLEmitter::peelBroadcastExpand(Value v) {
  while (v) {
    Operation *def = v.getDefiningOp();
    if (auto b = dyn_cast_or_null<tt::BroadcastOp>(def)) {
      v = b.getSrc();
      continue;
    }
    if (auto e = dyn_cast_or_null<tt::ExpandDimsOp>(def)) {
      v = e.getSrc();
      continue;
    }
    return v;
  }
  return v;
}

Value MSLEmitter::matchTileIndex(Value v) {
  v = peelBroadcastExpand(v);
  auto add = definingOp<arith::AddIOp>(v);
  if (!add)
    return Value();
  Value a = peelBroadcastExpand(add.getLhs());
  Value b = peelBroadcastExpand(add.getRhs());
  auto isIota = [](Value x) {
    auto sp = definingOp<tt::SplatOp>(x);
    Value base = sp ? sp.getSrc() : x;
    auto mr = dyn_cast_or_null<tt::MakeRangeOp>(
        peelBroadcastExpand(base).getDefiningOp());
    return mr && mr.getStart() == 0;
  };
  auto splatScalar = [](Value x) -> Value {
    auto sp = definingOp<tt::SplatOp>(x);
    return sp ? sp.getSrc() : Value();
  };
  if (isIota(b))
    if (Value s = splatScalar(a))
      return s;
  if (isIota(a))
    if (Value s = splatScalar(b))
      return s;
  return Value();
}

bool MSLEmitter::matchRowMajorOffset(Value off, Value &rowBase, Value &ldc,
                                     Value &colBase) {
  auto add = definingOp<arith::AddIOp>(off);
  if (!add)
    return false;
  auto tryTerm = [&](Value rowT, Value colT) {
    Value cb = matchTileIndex(colT);
    if (!cb)
      return false;
    Value rowScaled = peelBroadcastExpand(rowT);
    auto mul = definingOp<arith::MulIOp>(rowScaled);
    if (!mul)
      return false;
    auto scalarSrc = [](Value x) -> Value {
      auto sp =
          dyn_cast_or_null<tt::SplatOp>(peelBroadcastExpand(x).getDefiningOp());
      return sp ? sp.getSrc() : Value();
    };
    Value rIdxA = mul.getLhs(), rIdxB = mul.getRhs();
    Value rb = matchTileIndex(rIdxA);
    Value stride = scalarSrc(rIdxB);
    if (!rb || !stride) {
      rb = matchTileIndex(rIdxB);
      stride = scalarSrc(rIdxA);
    }
    if (!rb || !stride)
      return false;
    rowBase = rb;
    ldc = stride;
    colBase = cb;
    return true;
  };
  return tryTerm(add.getLhs(), add.getRhs()) ||
         tryTerm(add.getRhs(), add.getLhs());
}

bool MSLEmitter::matchBoundaryMask(Value m, Value &boundM, Value &boundN) {
  auto conj = definingOp<arith::AndIOp>(m);
  if (!conj)
    return false;
  auto cmpBound = [&](Value side, bool wantRow) -> Value {
    Value c = peelBroadcastExpand(side);
    auto cmp = definingOp<arith::CmpIOp>(c);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      return Value();
    if (!matchTileIndex(cmp.getLhs()))
      return Value();
    auto sp = dyn_cast_or_null<tt::SplatOp>(
        peelBroadcastExpand(cmp.getRhs()).getDefiningOp());
    return sp ? sp.getSrc() : Value();
  };
  if (Value bm = cmpBound(conj.getLhs(), true))
    if (Value bn = cmpBound(conj.getRhs(), false)) {
      boundM = bm;
      boundN = bn;
      return true;
    }
  if (Value bm = cmpBound(conj.getRhs(), true))
    if (Value bn = cmpBound(conj.getLhs(), false)) {
      boundM = bm;
      boundN = bn;
      return true;
    }
  return false;
}

std::optional<DirectStore> MSLEmitter::matchDirectStore(Value forResult) {
  if (!forResult.hasOneUse())
    return std::nullopt;
  auto cvt = dyn_cast<ttg::ConvertLayoutOp>(*forResult.user_begin());
  if (!cvt || !cvt.getResult().hasOneUse())
    return std::nullopt;
  auto store = dyn_cast<tt::StoreOp>(*cvt.getResult().user_begin());
  if (!store || store.getValue() != cvt.getResult())
    return std::nullopt;
  auto cTy = dyn_cast<RankedTensorType>(cvt.getResult().getType());
  if (!cTy || !cTy.getElementType().isF32())
    return std::nullopt;
  auto ptr = definingOp<tt::AddPtrOp>(store.getPtr());
  if (!ptr)
    return std::nullopt;
  auto splat = definingOp<tt::SplatOp>(ptr.getPtr());
  if (!splat || !isa<BlockArgument>(splat.getSrc()))
    return std::nullopt;
  Value rowBase, ldc, colBase;
  if (!matchRowMajorOffset(ptr.getOffset(), rowBase, ldc, colBase))
    return std::nullopt;
  Value boundM, boundN;
  if (store.getMask()) {
    if (!matchBoundaryMask(store.getMask(), boundM, boundN))
      return std::nullopt;
  }
  DirectStore ds;
  ds.store = store;
  ds.basePtr = splat.getSrc();
  ds.ldc = ldc;
  ds.rowBase = rowBase;
  ds.colBase = colBase;
  ds.boundM = boundM;
  ds.boundN = boundN;
  return ds;
}

bool MSLEmitter::dotIsFusedGemmAcc(tt::DotOp d) {
  auto forOp = dyn_cast<scf::ForOp>(d->getParentOp());
  if (!forOp)
    return false;
  auto m = matchGemmDotLoop(forOp);
  return m && m->first == d;
}

std::optional<std::pair<tt::DotOp, unsigned>>
MSLEmitter::matchGemmDotLoop(scf::ForOp op) {
  Block *body = op.getBody();
  auto yield = cast<scf::YieldOp>(body->getTerminator());
  tt::DotOp found;
  int nDots = 0;
  for (Operation &o : body->without_terminator())
    if (auto d = dyn_cast<tt::DotOp>(&o)) {
      found = d;
      ++nDots;
    }
  if (nDots != 1 || !found)
    return std::nullopt;

  auto cArg = dyn_cast<BlockArgument>(found.getC());
  if (!cArg || cArg.getOwner() != body)
    return std::nullopt;
  unsigned idx = cArg.getArgNumber();
  if (idx == 0)
    return std::nullopt; // arg 0 is the induction var
  unsigned iterIdx = idx - 1;
  if (yield.getOperand(iterIdx) != found.getResult())
    return std::nullopt;
  // The iter-arg feeds the dot's C and nothing else; the dot result feeds the
  // yield and nothing else. This keeps the accumulator purely register-carried.
  for (Operation *u : cArg.getUsers())
    if (u != found.getOperation())
      return std::nullopt;
  for (Operation *u : found.getResult().getUsers())
    if (u != yield.getOperation())
      return std::nullopt;

  auto cTy = dyn_cast<RankedTensorType>(found.getResult().getType());
  if (!cTy || cTy.getRank() != 2)
    return std::nullopt;
  Type aElem = cast<RankedTensorType>(found.getA().getType()).getElementType();
  Type cElem = cTy.getElementType();
  if (isa<IntegerType>(aElem) || !(cElem.isF32() || cElem.isF16()))
    return std::nullopt;
  int64_t M = cTy.getShape()[0], N = cTy.getShape()[1];
  int64_t K = cast<RankedTensorType>(found.getA().getType()).getShape()[1];
  if (M % 8 || N % 8 || K % 8)
    return std::nullopt;

  // Gate: the fused path needs a warp-tile of <= 32 simdgroup_float8x8
  // accumulators per warp AND a pool that holds the staged operands.
  // Anything larger falls back to the per-dot path. Staging bytes mirror the
  // dot path: an operand already resident in a threadgroup buffer (in-place)
  // stages 0.
  int64_t aBytes = M * K * byteWidth(aElem);
  int64_t bBytes = N * K * byteWidth(aElem);
  int64_t cFull = M * N * 4;
  bool wholeTileFits = aBytes + bBytes <= kTGResidentBudgetBytes;
  // A/B that structurally resolve to a local_alloc buffer are loaded in place
  // by emitDot (stage 0). The precise in-place base lives in memdescMap,
  // which is only populated once the enclosing memdesc_index is emitted inside
  // the loop; here (pre-loop) the structural walk is the reliable signal.
  int64_t stagedA = aBytes, stagedB = bBytes;
  if (wholeTileFits) {
    if (dotOperandLocalLoad(found.getA(), M, K))
      stagedA = 0;
    if (dotOperandLocalLoad(found.getB(), K, N))
      stagedB = 0;
  }
  // The fused epilogue overlays C's accumulators on the dead A/B staging, so
  // the pool only has to hold whichever of the two is larger.
  if (std::max(stagedA + stagedB, cFull) > poolBudget())
    return std::nullopt;
  tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
  auto kWarpDim = StringAttr::get(op.getContext(), "warp");
  int64_t numWarps = cLL.hasInDim(kWarpDim) ? cLL.getInDimSize(kWarpDim) : 1;
  int64_t nFrag = (M / 8) * (N / 8);
  if (numWarps > nFrag)
    numWarps = nFrag;
  int64_t fragsPerWarp = (nFrag + numWarps - 1) / numWarps;
  if (fragsPerWarp > 32)
    return std::nullopt;

  return std::make_pair(found, iterIdx);
}

unsigned MSLEmitter::reduceMask(const tt::LinearLayout &ll, StringAttr inDim,
                                StringAttr outDim) {
  unsigned mask = 0;
  if (!ll.hasInDim(inDim))
    return 0;
  for (int b = 0, n = ll.getInDimSizeLog2(inDim); b < n; ++b)
    if (ll.getBasis(inDim, b, outDim) != 0)
      mask |= (1u << b);
  return mask;
}

SmallVector<int> MSLEmitter::subsetsOf(unsigned mask, int numWarps) {
  SmallVector<int> bits;
  for (int b = 0; b < 16; ++b)
    if (mask & (1u << b))
      bits.push_back(b);
  SmallVector<int> vals;
  for (int s = 0; s < (1 << bits.size()); ++s) {
    int v = 0;
    for (int i = 0; i < (int)bits.size(); ++i)
      if (s & (1 << i))
        v |= (1 << bits[i]);
    if (v < numWarps)
      vals.push_back(v);
  }
  return vals;
}

SmallVector<std::pair<int, int32_t>>
MSLEmitter::axisBits(const tt::LinearLayout &ll, StringAttr inDim,
                     StringAttr outDim) {
  SmallVector<std::pair<int, int32_t>> bits;
  if (!ll.hasInDim(inDim))
    return bits;
  for (int b = 0, n = ll.getInDimSizeLog2(inDim); b < n; ++b) {
    int32_t basis = ll.getBasis(inDim, b, outDim);
    if (basis != 0)
      bits.push_back({b, basis});
  }
  llvm::sort(bits, [](auto &a, auto &c) { return a.second < c.second; });
  return bits;
}

ttg::LocalLoadOp MSLEmitter::dotOperandLocalLoad(Value operand, int64_t rows,
                                                 int64_t cols) {
  Value v = operand;
  while (auto cvt = definingOp<ttg::ConvertLayoutOp>(v))
    v = cvt.getSrc();
  auto ll = definingOp<ttg::LocalLoadOp>(v);
  if (!ll)
    return nullptr;
  auto mt = cast<ttg::MemDescType>(ll.getSrc().getType());
  if (mt.getRank() != 2 || mt.getShape()[0] != rows || mt.getShape()[1] != cols)
    return nullptr;
  Value src = ll.getSrc();
  while (Operation *def = src.getDefiningOp()) {
    if (auto mi = dyn_cast<ttg::MemDescIndexOp>(def)) {
      src = mi.getSrc();
      continue;
    }
    if (isa<ttg::LocalAllocOp>(def))
      return ll;
    return nullptr;
  }
  return nullptr;
}

bool MSLEmitter::dotReadsOperandInPlace(tt::DotOp d, Value operand) {
  auto cTy = cast<RankedTensorType>(d.getResult().getType());
  if (cTy.getRank() != 2)
    return false;
  int64_t M = cTy.getShape()[0], N = cTy.getShape()[1];
  int64_t Kd = cast<RankedTensorType>(d.getA().getType()).getShape()[1];
  int64_t aBy =
      M * Kd *
      (bitsOf(cast<RankedTensorType>(d.getA().getType()).getElementType()) / 8);
  int64_t bBy =
      Kd * N *
      (bitsOf(cast<RankedTensorType>(d.getB().getType()).getElementType()) / 8);
  if (aBy + bBy > kTGResidentBudgetBytes)
    return false;
  if (operand == d.getA())
    return dotOperandLocalLoad(operand, M, Kd);
  if (operand == d.getB())
    return dotOperandLocalLoad(operand, Kd, N);
  return false;
}

bool MSLEmitter::convertLayoutIsDeadDotStage(ttg::ConvertLayoutOp c) {
  if (c.getResult().use_empty())
    return false;
  for (OpOperand &use : c.getResult().getUses()) {
    auto d = dyn_cast<tt::DotOp>(use.getOwner());
    if (!d || !dotReadsOperandInPlace(d, c.getResult()))
      return false;
  }
  return true;
}

bool MSLEmitter::localLoadIsDeadDotStage(ttg::LocalLoadOp ll) {
  if (ll.getResult().use_empty())
    return false;
  for (OpOperand &use : ll.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (auto c = dyn_cast<ttg::ConvertLayoutOp>(owner)) {
      if (!convertLayoutIsDeadDotStage(c))
        return false;
      continue;
    }
    auto d = dyn_cast<tt::DotOp>(owner);
    if (!d || !dotReadsOperandInPlace(d, ll.getResult()))
      return false;
  }
  return true;
}

std::optional<InPlaceOperand>
MSLEmitter::dotOperandInPlaceBuf(Value operand, int64_t rows, int64_t cols) {
  ttg::LocalLoadOp ll = dotOperandLocalLoad(operand, rows, cols);
  if (!ll)
    return std::nullopt;
  auto it = memdescMap.find(ll.getSrc());
  if (it == memdescMap.end() || !it->second.bufStrides.empty())
    return std::nullopt;
  return InPlaceOperand{it->second.buf, it->second.baseOffset};
}

Value MSLEmitter::dotOperandConvertSource(tt::DotOp d, Value operand) {
  auto cvt = definingOp<ttg::ConvertLayoutOp>(operand);
  if (!cvt)
    return nullptr;
  Value src = cvt.getSrc();
  auto st = dyn_cast<RankedTensorType>(src.getType());
  if (!st || st.getRank() != 2)
    return nullptr;
  if (dotOperandLocalLoad(operand, st.getShape()[0], st.getShape()[1]))
    return nullptr;
  auto cTy = cast<RankedTensorType>(d.getResult().getType());
  if (cTy.getRank() != 2)
    return nullptr;
  return src;
}

bool MSLEmitter::convertLayoutIsDeadDotStageSource(ttg::ConvertLayoutOp c) {
  if (c.getResult().use_empty())
    return false;
  for (OpOperand &use : c.getResult().getUses()) {
    auto d = dyn_cast<tt::DotOp>(use.getOwner());
    if (!d)
      return false;
    if (use.get() != d.getA() && use.get() != d.getB())
      return false;
    auto cTy = cast<RankedTensorType>(d.getResult().getType());
    if (cTy.getRank() != 2)
      return false;
    if (!dotOperandConvertSource(d, use.get()))
      return false;
  }
  return true;
}

void MSLEmitter::dotPanelDims(int64_t M, int64_t N, int64_t K,
                              int64_t elemBytes, int64_t accBytes, int64_t &mp,
                              int64_t &np) {
  mp = M;
  np = N;
  auto fits = [&](int64_t m, int64_t n) {
    return m * K * elemBytes + K * n * elemBytes + m * n * accBytes <=
           kTGResidentBudgetBytes;
  };
  while (!fits(mp, np)) {
    if (mp >= np && mp > 8)
      mp -= 8;
    else if (np > 8)
      np -= 8;
    else if (mp > 8)
      mp -= 8;
    else
      break;
  }
}

bool MSLEmitter::dotNeedsPanel(int64_t M, int64_t N, int64_t K,
                               int64_t elemBytes, int64_t accBytes) {
  return M * K * elemBytes + K * N * elemBytes > kTGResidentBudgetBytes;
}

int64_t MSLEmitter::dotCBandRows(int64_t M, int64_t N, int64_t cBudget,
                                 int64_t accBytes) {
  int64_t rowBytes = N * accBytes;
  int64_t band = cBudget / rowBytes;
  band -= band % 8;
  if (band < 8)
    band = 8;
  if (band > M)
    band = M;
  return band;
}

} // namespace mlir::triton::applegpu
