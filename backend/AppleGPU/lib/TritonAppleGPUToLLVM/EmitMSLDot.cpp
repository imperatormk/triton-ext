// EmitMSLDot.cpp - dot/GEMM simdgroup-matrix + fp-narrowing lowering.
//
// AST builders for the simdgroup-matrix fragment MMA (astEmitDot /
// astEmitDotScalar / astEmitFusedGemm), plus the fp_to_fp narrowing helpers.
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.
// The fp-narrowing bit-twiddling body is the design-sanctioned Raw escape
// hatch (MSL_AST_DESIGN.md): the builder wraps that block in one RawStmt and
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
} // namespace

// tt.dot -> simdgroup-matrix fragment MMA: panel / fused (Decl/MMA/Readback +
// direct-store) / per-dot (disjoint / aliased-banded) paths. Guard/offset
// compound exprs are ctx.raw leaves; the per-fragment MMA + readback are the
// shared astSg* builders.
bool MSLEmitter::astEmitDot(tt::DotOp op, msl::Block &body) {
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  Type aElem = aTy.getElementType();
  Type bElem = bTy.getElementType();
  Type cElem = cTy.getElementType();
  if (isa<IntegerType>(aElem) || isa<IntegerType>(bElem) ||
      isa<IntegerType>(cElem))
    return astEmitDotScalar(op, body);
  int rank = aTy.getRank();
  if ((rank != 2 && rank != 3) || !isDotOperandElem(aElem) ||
      !isDotOperandElem(bElem) || aElem != bElem ||
      !(cElem.isF32() || cElem.isF16()))
    return false;
  int64_t Bd = rank == 3 ? cTy.getShape()[0] : 1;
  int64_t M = cTy.getShape()[rank - 2];
  int64_t N = cTy.getShape()[rank - 1];
  int64_t K = aTy.getShape()[rank - 1];
  if (M % 8 || N % 8 || K % 8)
    return false;

  auto &cInit = names(op.getC());
  std::string opScalar = sgOperandScalar(aElem);
  msl::MatrixType *opFrag = astSgFragType(aElem);
  msl::MatrixType *accFragTy = ctx.matrix(msl::MatrixType::Elem::Float);
  msl::Type *accScalarTy = astScalarType(cElem);
  std::string accScalar = mslScalarType(cElem);

  int64_t aBytes = M * K * (bitsOf(aElem) / 8);
  int64_t bBytes = N * K * (bitsOf(bElem) / 8);
  int64_t accBytes = 4;
  int64_t cFull = M * N * accBytes;

  std::optional<InPlaceOperand> aInPlace, bInPlace;
  bool wholeTileFits = M * K * (bitsOf(aElem) / 8) + bBytes <= 32768;
  if (rank == 2 && wholeTileFits) {
    aInPlace = dotOperandInPlaceBuf(op.getA(), M, K);
    bInPlace = dotOperandInPlaceBuf(op.getB(), K, N);
  }
  Value aStage = op.getA(), bStage = op.getB();
  if (!aInPlace)
    if (Value s = dotOperandConvertSource(op, op.getA())) aStage = s;
  if (!bInPlace)
    if (Value s = dotOperandConvertSource(op, op.getB())) bStage = s;
  auto aStageTy = cast<RankedTensorType>(aStage.getType());
  auto bStageTy = cast<RankedTensorType>(bStage.getType());
  auto &aNames = names(aStage);
  auto &bNames = names(bStage);

  int64_t stagedA = aInPlace ? 0 : aBytes;
  int64_t stagedB = bInPlace ? 0 : bBytes;
  int64_t stagedAB = stagedA + stagedB;
  bool disjointC = stagedAB + cFull <= poolBudget();
  int64_t bandRows = disjointC ? M : dotCBandRows(M, N, poolBudget(), accBytes);

  FusedDotPhase phase = fusedDot.phase;
  bool needAB = phase == FusedDotPhase::None || phase == FusedDotPhase::MMA;
  bool needC = phase != FusedDotPhase::Decl;

  // `threadgroup <scalar>* name` decl type (star attached, matches string).
  auto tgPtr = [&](StringRef scalar) {
    return ctx.named(("threadgroup " + scalar.str() + "*"));
  };
  auto barrier = [&] { body.push_back(ctx.hardBarrier(false)); };

  std::string tgA = fresh(), tgB = fresh(), tgC = fresh();
  if (needAB) {
    body.push_back(ctx.declStmt(
        tgPtr(opScalar), tgA,
        aInPlace ? ctx.raw(inPlaceBase(*aInPlace))
                 : astPoolRegion(0, opScalar)));
    body.push_back(ctx.declStmt(
        tgPtr(opScalar), tgB,
        bInPlace ? ctx.raw(inPlaceBase(*bInPlace))
                 : astPoolRegion(stagedA, opScalar)));
  }
  if (needC)
    body.push_back(ctx.declStmt(
        tgPtr("float"), tgC,
        astPoolRegion(disjointC ? stagedAB : 0, "float")));

  int64_t mT = M / 8, nT = N / 8, kT = K / 8;
  tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
  auto kWarpDim = StringAttr::get(op.getContext(), "warp");
  int64_t numWarps = cLL.hasInDim(kWarpDim) ? cLL.getInDimSize(kWarpDim) : 1;
  int64_t nFrag = mT * nT;
  if (numWarps > nFrag) numWarps = nFrag;

  int nRes = regCount(op.getResult());
  bool fused = phase != FusedDotPhase::None;
  SmallVector<std::string> ids(nRes);
  if (fused) {
    ids = fusedDot.ids;
  } else {
    for (int r = 0; r < nRes; ++r) {
      ids[r] = fresh();
      body.push_back(ctx.declStmt(accScalarTy, ids[r],
                                  ctx.cast(CS::CStyle, accScalarTy, ctx.lit("0"))));
    }
  }

  auto outNames = llvm::to_vector(cLL.getOutDimNames());
  StringAttr rowDim = outNames[rank - 2], colDim = outNames[rank - 1];

  // Wrap `inner` in `if (warpId == w) { ... }`.
  auto warpIf = [&](int64_t w, msl::Block inner) {
    body.push_back(ctx.ifScope(
        ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
        std::move(inner)));
  };

  // Per-dot readback: `if (guard) ids[r] = tgC[bandOff] + base;`
  ArrayRef<std::string> rbBase = fused ? ArrayRef<std::string>(fusedDot.baseNames)
                                       : ArrayRef<std::string>(cInit);
  auto readbackInto = [&](msl::Block &tgt, int64_t bi, int64_t r0, int64_t r1) {
    for (int r = 0; r < nRes; ++r) {
      std::string base = rbBase[rbBase.size() == 1 ? 0 : r];
      // ((rowExpr - r0) * N + colExpr)
      msl::Expr *bandOff = ctx.paren(ctx.add(
          ctx.mul(ctx.paren(ctx.binary(B::Sub, astLayoutCoordExpr(cTy, r, rowDim),
                                       ctx.i32lit(r0))),
                  ctx.i32lit(N)),
          astLayoutCoordExpr(cTy, r, colDim)));
      // (rowExpr >= r0 && rowExpr < r1)
      msl::Expr *guard = ctx.paren(ctx.binary(
          B::LAnd,
          ctx.binary(B::Ge, astLayoutCoordExpr(cTy, r, rowDim), ctx.i32lit(r0)),
          ctx.binary(B::Lt, astLayoutCoordExpr(cTy, r, rowDim), ctx.i32lit(r1))));
      if (rank == 3)
        guard = ctx.paren(ctx.binary(
            B::LAnd,
            ctx.binary(B::Eq, astBatchCoordExpr(cTy, r), ctx.i32lit(bi)),
            guard));
      tgt.push_back(ctx.compactIfBare(
          guard, ctx.assignStmt(ctx.var(ids[r]),
                                astReadbackValue(tgC, bandOff, base))));
    }
  };
  auto readback = [&](int64_t bi, int64_t r0, int64_t r1) {
    readbackInto(body, bi, r0, r1);
  };

  // Shared per-fragment MMA into `acc` (disjoint/aliased/fused-Readback-less).
  auto fragMMA = [&](StringRef tgAn, StringRef tgBn, int64_t mi, int64_t ni,
                     StringRef acc, msl::Block &into) {
    for (int64_t ki = 0; ki < kT; ++ki) {
      std::string fa = fresh(), fb = fresh();
      into.push_back(astFragDecl(opFrag, fa));
      into.push_back(astSgLoad(fa, tgAn, mi * 8 * K + ki * 8, K));
      into.push_back(astFragDecl(opFrag, fb));
      into.push_back(astSgLoad(fb, tgBn, ki * 8 * N + ni * 8, N));
      into.push_back(astSgMultiplyAccumulate(acc, fa, fb));
    }
  };

  if (dotNeedsPanel(M, N, K, bitsOf(aElem) / 8, accBytes)) {
    if (!astEmitDotPanel(op, body, aStage, bStage, aNames, bNames, cInit, ids,
                         M, N, K, Bd, rank, opFrag, opScalar, numWarps, rowDim,
                         colDim))
      return false;
    valMap[op.getResult()] = ids;
    return true;
  }

  if (fused) {
    if (!astEmitDotFused(op, body, aStage, bStage, aNames, bNames, tgA, tgB, tgC,
                         ids, M, N, K, mT, nT, kT, nFrag, numWarps, aInPlace,
                         bInPlace, opFrag, accFragTy, readbackInto))
      return false;
    valMap[op.getResult()] = ids;
    return true;
  }

  // Per-dot path. Batch guard condition `<batchCoord> == bi` (rank-3 only).
  auto batchCond = [&](RankedTensorType rt, int reg, int64_t bi) -> msl::Expr * {
    return ctx.binary(B::Eq, astBatchCoordExpr(rt, reg),
                      ctx.lit(std::to_string(bi)));
  };
  for (int64_t bi = 0; bi < Bd; ++bi) {
    barrier();
    if (!aInPlace)
      for (int r = 0, n = regCount(aStage); r < n; ++r) {
        msl::Stmt *asn = ctx.assignStmt(
            ctx.subscript(ctx.var(tgA), astSliceFlatOffset(aStageTy, r)),
            ctx.var(aNames[r]));
        body.push_back(rank == 3 ? ctx.compactIf(batchCond(aStageTy, r, bi), asn)
                                 : asn);
      }
    if (!bInPlace)
      for (int r = 0, n = regCount(bStage); r < n; ++r) {
        msl::Stmt *asn = ctx.assignStmt(
            ctx.subscript(ctx.var(tgB), astSliceFlatOffset(bStageTy, r)),
            ctx.var(bNames[r]));
        body.push_back(rank == 3 ? ctx.compactIf(batchCond(bStageTy, r, bi), asn)
                                 : asn);
      }
    barrier();

    if (disjointC) {
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        for (int64_t f = w; f < nFrag; f += numWarps) {
          int64_t mi = f / nT, ni = f % nT;
          std::string acc = fresh();
          inner.push_back(astAccFragDecl(accFragTy, acc));
          fragMMA(tgA, tgB, mi, ni, acc, inner);
          inner.push_back(astSgStore(acc, tgC, mi * 8 * N + ni * 8, N));
        }
        warpIf(w, std::move(inner));
      }
      barrier();
      readback(bi, 0, M);
      continue;
    }

    std::string accBase = fresh() + "_" + std::to_string(bi) + "_";
    for (int64_t f = 0; f < nFrag; ++f) {
      int64_t mi = f / nT, ni = f % nT;
      std::string acc = accBase + std::to_string(mi) + "_" + std::to_string(ni);
      body.push_back(astAccFragDecl(accFragTy, acc));
    }
    for (int64_t w = 0; w < numWarps; ++w) {
      msl::Block inner;
      for (int64_t f = w; f < nFrag; f += numWarps) {
        int64_t mi = f / nT, ni = f % nT;
        std::string acc = accBase + std::to_string(mi) + "_" + std::to_string(ni);
        fragMMA(tgA, tgB, mi, ni, acc, inner);
      }
      warpIf(w, std::move(inner));
    }
    for (int64_t r0 = 0; r0 < M; r0 += bandRows) {
      int64_t r1 = std::min<int64_t>(r0 + bandRows, M);
      barrier();
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        for (int64_t f = w; f < nFrag; f += numWarps) {
          int64_t mi = f / nT, ni = f % nT;
          if (mi * 8 < r0 || mi * 8 >= r1) continue;
          std::string acc = accBase + std::to_string(mi) + "_" + std::to_string(ni);
          inner.push_back(astSgStore(acc, tgC, (mi * 8 - r0) * N + ni * 8, N));
        }
        if (!inner.empty())
          warpIf(w, std::move(inner));
        else
          warpIf(w, msl::Block{});
      }
      barrier();
      readback(bi, r0, r1);
    }
  }
  barrier();
  valMap[op.getResult()] = ids;
  return true;
}

// Fused GEMM dot phases: Decl (declare/zero persistent frags), MMA (stage A/B +
// accumulate, branchless or per-warp ladder), Readback (store frags + gather,
// with optional direct-store to device C).
bool MSLEmitter::astEmitDotFused(
    tt::DotOp op, msl::Block &body, Value aStage, Value bStage,
    ArrayRef<std::string> aNames, ArrayRef<std::string> bNames, StringRef tgA,
    StringRef tgB, StringRef tgC, ArrayRef<std::string> ids, int64_t M,
    int64_t N, int64_t K, int64_t mT, int64_t nT, int64_t kT, int64_t nFrag,
    int64_t numWarps, std::optional<InPlaceOperand> aInPlace,
    std::optional<InPlaceOperand> bInPlace, msl::MatrixType *opFrag,
    msl::MatrixType *accFragTy,
    llvm::function_ref<void(msl::Block &, int64_t, int64_t, int64_t)>
        readbackInto) {
  auto aStageTy = cast<RankedTensorType>(aStage.getType());
  auto bStageTy = cast<RankedTensorType>(bStage.getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  int64_t fragsPerWarp = (nFrag + numWarps - 1) / numWarps;
  auto barrier = [&] { body.push_back(ctx.hardBarrier(false)); };

  if (fusedDot.phase == FusedDotPhase::Decl) {
    fusedDot.accNames.assign(fragsPerWarp, "");
    for (int64_t j = 0; j < fragsPerWarp; ++j) {
      std::string acc = fresh();
      fusedDot.accNames[j] = acc;
      body.push_back(astAccFragDecl(accFragTy, acc));
    }
    return true;
  }

  if (fusedDot.phase == FusedDotPhase::MMA) {
    bool stagesHere = !aInPlace || !bInPlace;
    if (stagesHere)
      barrier();
    if (!aInPlace)
      for (int r = 0, n = regCount(aStage); r < n; ++r)
        body.push_back(ctx.assignStmt(
            ctx.subscript(ctx.var(tgA), astSliceFlatOffset(aStageTy, r)),
            ctx.var(aNames[r])));
    if (!bInPlace)
      for (int r = 0, n = regCount(bStage); r < n; ++r)
        body.push_back(ctx.assignStmt(
            ctx.subscript(ctx.var(tgB), astSliceFlatOffset(bStageTy, r)),
            ctx.var(bNames[r])));
    barrier();
    bool branchless = (numWarps == nT);
    // slots: (mi, niKey, niExpr); niKey dedups bFrag, niExpr is the typed index.
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
            into.push_back(astFragDecl(opFrag, fa));
            into.push_back(astSgLoad(fa, tgA, mi * 8 * K + ki * 8, K));
          }
          if (!bFrag.count(niKey)) {
            std::string fb = fresh();
            bFrag[niKey] = fb;
            into.push_back(astFragDecl(opFrag, fb));
            // simdgroup_load(fb, tgB + (ki*8*N + niExpr * 8), N);
            msl::Expr *off = ctx.paren(ctx.add(
                ctx.i32lit(ki * 8 * N), ctx.mul(pr.niExpr, ctx.i32lit(8))));
            into.push_back(ctx.exprStmt(ctx.call(
                msl::builtin::sg::Load,
                {ctx.var(fb), ctx.binary(B::Add, ctx.var(tgB), off),
                 ctx.i32lit(N)})));
          }
        }
        for (auto [j, mn] : llvm::enumerate(slots)) {
          const std::string &acc = fusedDot.accNames[j];
          into.push_back(
              astSgMultiplyAccumulate(acc, aFrag[mn.mi], bFrag[mn.niKey]));
        }
      }
    };
    if (branchless) {
      std::string niKey = "(" + warpId + " % " + std::to_string(nT) + ")";
      msl::Expr *niExpr = ctx.paren(ctx.binary(B::Rem, ctx.var(warpId),
                                               ctx.i32lit(nT)));
      SmallVector<Slot> slots;
      for (int64_t j = 0; j * numWarps < nFrag; ++j)
        slots.push_back({(j * numWarps) / nT, niKey, niExpr});
      emitSlots(slots, body);
    } else {
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        SmallVector<Slot> slots;
        for (int64_t f = w; f < nFrag; f += numWarps)
          slots.push_back(
              {f / nT, std::to_string(f % nT), ctx.i32lit(f % nT)});
        emitSlots(slots, inner);
        body.push_back(ctx.ifScope(
            ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
            std::move(inner)));
      }
    }
    if (!stagesHere)
      barrier();
    valMap[op.getResult()] = SmallVector<std::string>(ids.begin(), ids.end());
    return true;
  }

  // Readback.
  if (fusedDot.direct) {
    const DirectStore &d = *fusedDot.direct;
    std::string base = names(d.basePtr)[0], ldc = names(d.ldc)[0];
    std::string rowB = names(d.rowBase)[0], colB = names(d.colBase)[0];
    msl::Block ifBody;
    for (int64_t w = 0; w < numWarps; ++w) {
      msl::Block inner;
      for (int64_t f = w, j = 0; f < nFrag; f += numWarps, ++j) {
        int64_t mi = f / nT, ni = f % nT;
        // simdgroup_store(acc, base + (rowB + mi*8)*ldc + (colB + ni*8), ldc);
        msl::Expr *off = ctx.addChain(
            {ctx.var(base),
             ctx.mul(ctx.paren(ctx.add(ctx.var(rowB), ctx.i32lit(mi * 8))),
                     ctx.var(ldc)),
             ctx.paren(ctx.add(ctx.var(colB), ctx.i32lit(ni * 8)))});
        inner.push_back(ctx.exprStmt(ctx.call(
            msl::builtin::sg::Store,
            {ctx.var(fusedDot.accNames[j]), off,
             ctx.var(ldc)})));
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
        for (int64_t f = w, j = 0; f < nFrag; f += numWarps, ++j) {
          int64_t mi = f / nT, ni = f % nT;
          inner.push_back(astSgStore(fusedDot.accNames[j], tgC,
                                     mi * 8 * N + ni * 8, N));
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
    for (int64_t f = w, j = 0; f < nFrag; f += numWarps, ++j) {
      int64_t mi = f / nT, ni = f % nT;
      inner.push_back(astSgStore(fusedDot.accNames[j], tgC, mi * 8 * N + ni * 8, N));
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
bool MSLEmitter::astEmitDotPanel(tt::DotOp op, msl::Block &body, Value aStage,
                                 Value bStage, ArrayRef<std::string> aNames,
                                 ArrayRef<std::string> bNames,
                                 ArrayRef<std::string> cInit,
                                 ArrayRef<std::string> ids, int64_t M, int64_t N,
                                 int64_t K, int64_t Bd, int rank,
                                 msl::MatrixType *opFrag, StringRef opScalar,
                                 int64_t numWarps, StringAttr rowDim,
                                 StringAttr colDim) {
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  Type aElem = aTy.getElementType();
  auto aStageTy = cast<RankedTensorType>(aStage.getType());
  auto bStageTy = cast<RankedTensorType>(bStage.getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  int64_t elemBytes = bitsOf(aElem) / 8, accBytes = 4;
  int64_t mp, np;
  dotPanelDims(M, N, K, elemBytes, accBytes, mp, np);
  int64_t aPanelBytes = mp * K * elemBytes, bPanelBytes = K * np * elemBytes;
  auto aOut = llvm::to_vector(ttg::toLinearLayout(aStageTy).getOutDimNames());
  StringAttr aRowDim = aOut[rank - 2], aColDim = aOut[rank - 1];
  auto bOut = llvm::to_vector(ttg::toLinearLayout(bStageTy).getOutDimNames());
  StringAttr bColDim = bOut[rank - 1], bRowDim = bOut[rank - 2];
  int nRes = regCount(op.getResult());

  auto tgPtr = [&](StringRef s) { return ctx.named(("threadgroup " + s.str() + "*")); };
  std::string pA = fresh(), pB = fresh(), pC = fresh();
  body.push_back(ctx.declStmt(tgPtr(opScalar), pA, astPoolRegion(0, opScalar)));
  body.push_back(ctx.declStmt(tgPtr(opScalar), pB, astPoolRegion(aPanelBytes, opScalar)));
  body.push_back(ctx.declStmt(tgPtr("float"), pC,
                              astPoolRegion(aPanelBytes + bPanelBytes, "float")));
  int nARegs = regCount(aStage), nBRegs = regCount(bStage);
  auto barrier = [&] { body.push_back(ctx.hardBarrier(false)); };

  for (int64_t bi = 0; bi < Bd; ++bi) {
    for (int64_t m0 = 0; m0 < M; m0 += mp) {
      int64_t m1 = std::min<int64_t>(m0 + mp, M), mpCur = m1 - m0;
      barrier();
      for (int r = 0; r < nARegs; ++r) {
        // (row >= m0 && row < m1)
        msl::Expr *guard = ctx.paren(ctx.binary(
            B::LAnd,
            ctx.binary(B::Ge, astLayoutCoordExpr(aStageTy, r, aRowDim),
                       ctx.i32lit(m0)),
            ctx.binary(B::Lt, astLayoutCoordExpr(aStageTy, r, aRowDim),
                       ctx.i32lit(m1))));
        if (rank == 3)
          guard = ctx.paren(ctx.binary(
              B::LAnd,
              ctx.binary(B::Eq, astBatchCoordExpr(aStageTy, r), ctx.i32lit(bi)),
              guard));
        // ((row - m0) * K + col)
        msl::Expr *off = ctx.paren(ctx.add(
            ctx.mul(ctx.paren(ctx.binary(B::Sub,
                                         astLayoutCoordExpr(aStageTy, r, aRowDim),
                                         ctx.i32lit(m0))),
                    ctx.i32lit(K)),
            astLayoutCoordExpr(aStageTy, r, aColDim)));
        body.push_back(ctx.compactIfBare(
            guard, ctx.assignStmt(ctx.subscript(ctx.var(pA), off),
                                  ctx.var(aNames[r]))));
      }
      for (int64_t n0 = 0; n0 < N; n0 += np) {
        int64_t n1 = std::min<int64_t>(n0 + np, N), npCur = n1 - n0;
        int64_t pmT = mpCur / 8, pnT = npCur / 8;
        barrier();
        for (int r = 0; r < nBRegs; ++r) {
          // (col >= n0 && col < n1)
          msl::Expr *guard = ctx.paren(ctx.binary(
              B::LAnd,
              ctx.binary(B::Ge, astLayoutCoordExpr(bStageTy, r, bColDim),
                         ctx.i32lit(n0)),
              ctx.binary(B::Lt, astLayoutCoordExpr(bStageTy, r, bColDim),
                         ctx.i32lit(n1))));
          if (rank == 3)
            guard = ctx.paren(ctx.binary(
                B::LAnd,
                ctx.binary(B::Eq, astBatchCoordExpr(bStageTy, r), ctx.i32lit(bi)),
                guard));
          // (row * npCur + (col - n0))
          msl::Expr *off = ctx.paren(ctx.add(
              ctx.mul(astLayoutCoordExpr(bStageTy, r, bRowDim), ctx.i32lit(npCur)),
              ctx.paren(ctx.binary(B::Sub, astLayoutCoordExpr(bStageTy, r, bColDim),
                                   ctx.i32lit(n0)))));
          body.push_back(ctx.compactIfBare(
              guard, ctx.assignStmt(ctx.subscript(ctx.var(pB), off),
                                    ctx.var(bNames[r]))));
        }
        barrier();
        int64_t pnFrag = pmT * pnT;
        int64_t pWarps = numWarps > pnFrag ? pnFrag : numWarps;
        for (int64_t w = 0; w < pWarps; ++w) {
          msl::Block inner;
          for (int64_t f = w; f < pnFrag; f += pWarps) {
            int64_t mi = f / pnT, ni = f % pnT;
            std::string acc = fresh();
            inner.push_back(astAccFragDecl(ctx.matrix(msl::MatrixType::Elem::Float), acc));
            for (int64_t ki = 0; ki < (K / 8); ++ki) {
              std::string fa = fresh(), fb = fresh();
              inner.push_back(astFragDecl(opFrag, fa));
              inner.push_back(astSgLoad(fa, pA, mi * 8 * K + ki * 8, K));
              inner.push_back(astFragDecl(opFrag, fb));
              inner.push_back(astSgLoad(fb, pB, ki * 8 * npCur + ni * 8, npCur));
              inner.push_back(astSgMultiplyAccumulate(acc, fa, fb));
            }
            inner.push_back(astSgStore(acc, pC, mi * 8 * npCur + ni * 8, npCur));
          }
          body.push_back(ctx.ifScope(
              ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
              std::move(inner)));
        }
        barrier();
        for (int r = 0; r < nRes; ++r) {
          std::string base = cInit[cInit.size() == 1 ? 0 : r];
          // ((rowExpr - m0) * npCur + (colExpr - n0))
          msl::Expr *off = ctx.paren(ctx.add(
              ctx.mul(ctx.paren(ctx.binary(B::Sub,
                                           astLayoutCoordExpr(cTy, r, rowDim),
                                           ctx.i32lit(m0))),
                      ctx.i32lit(npCur)),
              ctx.paren(ctx.binary(B::Sub, astLayoutCoordExpr(cTy, r, colDim),
                                   ctx.i32lit(n0)))));
          // (rowExpr >= m0 && rowExpr < m1 && colExpr >= n0 && colExpr < n1)
          msl::Expr *guard = ctx.paren(ctx.chain(
              B::LAnd,
              {ctx.binary(B::Ge, astLayoutCoordExpr(cTy, r, rowDim),
                          ctx.i32lit(m0)),
               ctx.binary(B::Lt, astLayoutCoordExpr(cTy, r, rowDim),
                          ctx.i32lit(m1)),
               ctx.binary(B::Ge, astLayoutCoordExpr(cTy, r, colDim),
                          ctx.i32lit(n0)),
               ctx.binary(B::Lt, astLayoutCoordExpr(cTy, r, colDim),
                          ctx.i32lit(n1))}));
          if (rank == 3)
            guard = ctx.paren(ctx.binary(
                B::LAnd,
                ctx.binary(B::Eq, astBatchCoordExpr(cTy, r), ctx.i32lit(bi)),
                guard));
          body.push_back(ctx.compactIfBare(
              guard, ctx.assignStmt(
                         ctx.var(ids[r]),
                         ctx.binary(B::Add, ctx.subscript(ctx.var(pC), off),
                                    ctx.var(base)))));
        }
      }
    }
  }
  body.push_back(ctx.hardBarrier(false));
  return true;
}

// Integer/scalar tt.dot: stage A/B into the pool, then each thread runs a scalar
// K-loop for its owned C registers.
bool MSLEmitter::astEmitDotScalar(tt::DotOp op, msl::Block &body) {
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
  msl::Type *accTy = astScalarType(cElem);

  Value aStage = op.getA(), bStage = op.getB();
  if (rank == 2) {
    if (Value s = dotOperandConvertSource(op, op.getA())) aStage = s;
    if (Value s = dotOperandConvertSource(op, op.getB())) bStage = s;
  }
  auto aStageTy = cast<RankedTensorType>(aStage.getType());
  auto bStageTy = cast<RankedTensorType>(bStage.getType());
  auto &aNames = names(aStage);
  auto &bNames = names(bStage);
  auto &cInit = names(op.getC());
  int64_t aBytes = cTy.getShape()[rank - 2] * K * (bitsOf(aElem) / 8);

  auto tgPtr = [&](StringRef s) { return ctx.named(("threadgroup " + s.str() + "*")); };
  std::string tgA = fresh(), tgB = fresh();
  body.push_back(ctx.declStmt(tgPtr(aScalar), tgA, astPoolRegion(0, aScalar)));
  body.push_back(ctx.declStmt(tgPtr(bScalar), tgB, astPoolRegion(aBytes, bScalar)));

  tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
  auto outNames = llvm::to_vector(cLL.getOutDimNames());
  StringAttr dRow = outNames[rank - 2], dCol = outNames[rank - 1];
  auto batchCond = [&](RankedTensorType rt, int reg, int64_t bi) {
    return ctx.binary(B::Eq, astBatchCoordExpr(rt, reg),
                      ctx.lit(std::to_string(bi)));
  };

  int nRes = regCount(op.getResult());
  SmallVector<std::string> ids(nRes);
  for (int r = 0; r < nRes; ++r) {
    ids[r] = fresh();
    body.push_back(ctx.declStmt(accTy, ids[r],
                                ctx.var(cInit[cInit.size() == 1 ? 0 : r])));
  }

  for (int64_t bi = 0; bi < Bd; ++bi) {
    body.push_back(ctx.hardBarrier(false));
    for (int r = 0, n = regCount(aStage); r < n; ++r) {
      msl::Stmt *asn = ctx.assignStmt(
          ctx.subscript(ctx.var(tgA), astSliceFlatOffset(aStageTy, r)),
          ctx.var(aNames[r]));
      body.push_back(rank == 3 ? ctx.compactIf(batchCond(aStageTy, r, bi), asn)
                               : asn);
    }
    for (int r = 0, n = regCount(bStage); r < n; ++r) {
      msl::Stmt *asn = ctx.assignStmt(
          ctx.subscript(ctx.var(tgB), astSliceFlatOffset(bStageTy, r)),
          ctx.var(bNames[r]));
      body.push_back(rank == 3 ? ctx.compactIf(batchCond(bStageTy, r, bi), asn)
                               : asn);
    }
    body.push_back(ctx.hardBarrier(false));

    for (int r = 0; r < nRes; ++r) {
      std::string mrow = fresh(), ncol = fresh(), acc = fresh(), kv = fresh();
      body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), mrow,
                                  astLayoutCoordExpr(cTy, r, dRow)));
      body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), ncol,
                                  astLayoutCoordExpr(cTy, r, dCol)));
      body.push_back(ctx.declStmt(accTy, acc,
                                  ctx.cast(CS::CStyle, accTy, ctx.lit("0"))));
      // for (int kv = 0; kv < K; ++kv) { acc += (acc)tgA[mrow*K+kv] * (acc)tgB[kv*N+ncol]; }
      msl::Stmt *init = ctx.declStmt(ctx.scalar(msl::Scalar::I32), kv, ctx.lit("0"));
      msl::Expr *cond = ctx.binary(B::Lt, ctx.var(kv), ctx.lit(std::to_string(K)));
      msl::Stmt *step = ctx.exprStmt(ctx.raw("++" + kv));
      msl::Expr *aElemE = ctx.cast(
          CS::CStyle, accTy,
          ctx.subscript(ctx.var(tgA),
                        ctx.add(ctx.mul(ctx.var(mrow), ctx.i32lit(K)),
                                ctx.var(kv))));
      msl::Expr *bElemE = ctx.cast(
          CS::CStyle, accTy,
          ctx.subscript(ctx.var(tgB),
                        ctx.add(ctx.mul(ctx.var(kv), ctx.i32lit(N)),
                                ctx.var(ncol))));
      msl::Block loop;
      loop.push_back(ctx.addAssignStmt(ctx.var(acc),
                                       ctx.binary(B::Mul, aElemE, bElemE)));
      body.push_back(ctx.forScope(init, cond, step, std::move(loop)));
      msl::Stmt *accum = ctx.addAssignStmt(ctx.var(ids[r]), ctx.var(acc));
      body.push_back(rank == 3 ? ctx.compactIf(batchCond(cTy, r, bi), accum)
                               : accum);
    }
  }
  body.push_back(ctx.hardBarrier(false));
  valMap[op.getResult()] = ids;
  return true;
}

//===----------------------------------------------------------------------===//
// simdgroup-matrix fragment MMA sub-builders
//===----------------------------------------------------------------------===//

msl::MatrixType *MSLEmitter::astSgFragType(Type t) {
  if (t.isF16())
    return ctx.matrix(msl::MatrixType::Elem::Half);
  if (t.isBF16())
    return ctx.matrix(msl::MatrixType::Elem::Bfloat);
  return ctx.matrix(msl::MatrixType::Elem::Float);
}

// `base + off` (no outer paren - emitted bare inside the call).
msl::Expr *MSLEmitter::astFragAddr(StringRef base, int64_t off) {
  return ctx.binary(msl::BinOp::Add, ctx.var(base), ctx.lit(std::to_string(off)));
}

// `frag name;` - uninitialized operand fragment decl.
msl::Stmt *MSLEmitter::astFragDecl(msl::MatrixType *frag, StringRef name) {
  return ctx.declStmt(frag, name);
}

// `frag name = frag(0.0f);` - zeroed accumulator fragment. The type name doubles
// as the ctor callee (simdgroup_float8x8(0.0f)); read the printed name back off
// the MatrixType so the call callee matches the decl type exactly.
msl::Stmt *MSLEmitter::astAccFragDecl(msl::MatrixType *frag, StringRef name) {
  StringRef ctorName = frag->elem == msl::MatrixType::Elem::Half
                           ? msl::builtin::sg::Half8x8
                       : frag->elem == msl::MatrixType::Elem::Bfloat
                           ? msl::builtin::sg::Bfloat8x8
                           : msl::builtin::sg::Float8x8;
  return ctx.declStmt(frag, name, ctx.call(ctorName, {ctx.lit("0.0f")}));
}

// `simdgroup_load(frag, base + off, ld);`
msl::Stmt *MSLEmitter::astSgLoad(StringRef frag, StringRef base, int64_t off,
                                 int64_t ld) {
  return ctx.exprStmt(ctx.call(
      msl::builtin::sg::Load,
      {ctx.var(frag), astFragAddr(base, off), ctx.lit(std::to_string(ld))}));
}

// `simdgroup_store(acc, base + off, ld);`
msl::Stmt *MSLEmitter::astSgStore(StringRef acc, StringRef base, int64_t off,
                                  int64_t ld) {
  return ctx.exprStmt(ctx.call(
      msl::builtin::sg::Store,
      {ctx.var(acc), astFragAddr(base, off), ctx.lit(std::to_string(ld))}));
}

// `simdgroup_multiply_accumulate(acc, a, b, acc);`
msl::Stmt *MSLEmitter::astSgMultiplyAccumulate(StringRef acc, StringRef a,
                                               StringRef b) {
  return ctx.exprStmt(
      ctx.call(msl::builtin::sg::MultiplyAccumulate,
               {ctx.var(acc), ctx.var(a), ctx.var(b), ctx.var(acc)}));
}

// C readback value: `buf[off] + base` (off already carries its outer parens).
msl::Expr *MSLEmitter::astReadbackValue(StringRef buf, msl::Expr *off,
                                        StringRef base) {
  return ctx.binary(msl::BinOp::Add, ctx.subscript(ctx.var(buf), off),
                    ctx.var(base));
}

} // namespace mlir::triton::applegpu
