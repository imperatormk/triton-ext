// EmitMSLOps.cpp - op-dispatch spine + its AST lowering helpers.
//
// The astEmitOp waterfall and the per-family helpers it calls
// (astElemwiseDecls/astCombineN/astScanWarpCarry/astEmitMapCFG/astEmitFusedGemm/
// astDeclResultVars/astDeclBind/astDerefPtr/astBandRoundTrip/astStoreBody) plus
// the local cmpBinOp string->BinOp map.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

using B = msl::BinOp;
using CS = msl::Cast::Style;

//===----------------------------------------------------------------------===//
// Op dispatch spine
//===----------------------------------------------------------------------===//

// Append the nodes for a single elementwise/expr op's per-register DeclStmts,
// using the expr-builder `mk(r)` per register. astEmitOp mints the real names.
bool MSLEmitter::astElemwiseDecls(
    Operation *op, msl::Type *declTy, int &id, msl::Block &body,
    llvm::function_ref<msl::Expr *(int)> mk) {
  LocalGen g{id};
  int rc = regCount(op->getResult(0));
  for (int r = 0; r < rc; ++r)
    body.push_back(ctx.declStmt(declTy, g.fresh(), mk(r)));
  return true;
}

// Evaluate a reduce/scan/map combiner region into `body`: bind the 2N block
// args to the operand names, AST-walk the region (its ops - arith/math/select),
// and return the N terminator result names. The combiner ops are all
// single-block, so a plain walk suffices.
bool MSLEmitter::astCombineN(Region &region, ArrayRef<std::string> aVals,
                             ArrayRef<std::string> bVals, msl::Block &body,
                             SmallVectorImpl<std::string> &results) {
  Block &blk = region.front();
  int n = aVals.size();
  for (int i = 0; i < n; ++i) {
    bindScalar(blk.getArgument(i), aVals[i]);
    bindScalar(blk.getArgument(n + i), bVals[i]);
  }
  int savedIndent = indent;
  for (Operation &o : blk.without_terminator()) {
    if (astEmitOp(&o, body))
      continue;
    o.emitError("EmitMSL: unhandled combiner op '" +
                o.getName().getStringRef() + "'");
    emitFailed = true;
  }
  indent = savedIndent;
  if (emitFailed)
    return false;
  Operation *term = blk.getTerminator();
  for (Value r : term->getOperands())
    results.push_back(names(r)[0]);
  return true;
}

// Cross-warp inclusive-carry for one register group. Compound guard/index exprs
// are ctx.raw leaves.
bool MSLEmitter::astScanWarpCarry(
    Region &region, int nOp, ArrayRef<std::string> scTys,
    ArrayRef<int64_t> byteWidths, ArrayRef<std::pair<int, int32_t>> warpBits,
    ArrayRef<int> regs, SmallVector<SmallVector<std::string>> &accs,
    ArrayRef<std::string> laneScan, StringRef axisTopLane, unsigned axisWarpMask,
    int numWarps, bool rev, SmallVectorImpl<std::string> &runTotalOut,
    msl::Block &body) {
  SmallVector<msl::Type *> scT(nOp);
  for (int k = 0; k < nOp; ++k)
    scT[k] = ctx.named(scTys[k]);

  if (warpBits.empty()) {
    for (int k = 0; k < nOp; ++k) {
      std::string s =
          astShuffle("simd_shuffle", scTys[k], laneScan[k], axisTopLane, body);
      body.push_back(ctx.assignStmt(ctx.var(runTotalOut[k]), ctx.var(s)));
    }
    return true;
  }

  SmallVector<std::string> scratch(nOp);
  int64_t byteOff = 0;
  for (int k = 0; k < nOp; ++k) {
    scratch[k] = fresh();
    body.push_back(ctx.declStmt(
        ctx.ptr(scT[k], msl::AddrSpace::Threadgroup), scratch[k],
        astPoolRegion(byteOff, scTys[k])));
    byteOff += (int64_t)numWarps * 32 * byteWidths[k];
  }
  body.push_back(ctx.hardBarrier(false));
  msl::Expr *topGuard =
      axisTopLane == StringRef(laneId)
          ? static_cast<msl::Expr *>(ctx.lit("true"))
          : ctx.binary(B::Eq, ctx.var(laneId), ctx.var(axisTopLane));
  for (int k = 0; k < nOp; ++k) {
    // scratch[k][warp * 32 + lane] = laneScan[k];
    msl::Expr *idx = ctx.binary(
        B::Add, ctx.binary(B::Mul, ctx.var(warpId), ctx.lit("32")),
        ctx.var(laneId));
    msl::Stmt *asn = ctx.assignStmt(ctx.subscript(ctx.var(scratch[k]), idx),
                                    ctx.var(laneScan[k]));
    body.push_back(ctx.compactIf(topGuard, asn));
  }
  body.push_back(ctx.hardBarrier(false));

  // ((warpId & ~axisWarpMask) * 32 + axisTopLane)
  msl::Expr *base = ctx.paren(ctx.add(
      ctx.mul(ctx.paren(ctx.binary(B::And, ctx.var(warpId),
                                   ctx.i32lit(~axisWarpMask))),
              ctx.lit("32")),
      ctx.var(axisTopLane)));

  SmallVector<int> maskBits;
  for (size_t r = 0; r < warpBits.size(); ++r)
    maskBits.push_back(warpBits[r].first);
  int nParts = 1 << warpBits.size();
  auto partWarp = [&](int part) {
    int w = 0;
    for (size_t b = 0; b < maskBits.size(); ++b)
      if (part & (1 << b))
        w |= (1 << maskBits[b]);
    return w;
  };

  std::string myPart = fresh();
  {
    // posTerms[r] = ((((warpId >> maskBits[r]) & 1) << r))
    SmallVector<msl::Expr *> posTerms;
    for (size_t r = 0; r < maskBits.size(); ++r)
      posTerms.push_back(ctx.paren(ctx.paren(ctx.binary(
          B::Shl,
          ctx.paren(ctx.binary(
              B::And,
              ctx.paren(ctx.binary(B::Shr, ctx.var(warpId),
                                   ctx.i32lit(maskBits[r]))),
              ctx.i32lit(1))),
          ctx.i32lit(r)))));
    msl::Expr *warpPos = posTerms[0];
    for (size_t i = 1; i < posTerms.size(); ++i)
      warpPos = ctx.paren(ctx.binary(B::Or, warpPos, posTerms[i]));
    body.push_back(
        ctx.declStmt(ctx.scalar(msl::Scalar::I32), myPart, warpPos));
  }

  SmallVector<int> order;
  for (int p = 0; p < nParts; ++p)
    order.push_back(rev ? nParts - 1 - p : p);

  auto slot = [&](int k, int part) -> msl::Expr * {
    return ctx.subscript(ctx.var(scratch[k]),
                         ctx.binary(B::Add, base,
                                    ctx.lit(std::to_string(partWarp(part) * 32))));
  };

  SmallVector<std::string> grand(nOp);
  for (int k = 0; k < nOp; ++k) {
    grand[k] = fresh();
    body.push_back(ctx.declStmt(scT[k], grand[k], slot(k, order[0])));
  }
  for (int idx = 1; idx < nParts; ++idx) {
    SmallVector<std::string> pv(nOp);
    for (int k = 0; k < nOp; ++k) {
      pv[k] = fresh();
      body.push_back(ctx.declStmt(scT[k], pv[k], slot(k, order[idx])));
    }
    SmallVector<std::string> out;
    if (!astCombineN(region, grand, pv, body, out))
      return false;
    for (int k = 0; k < nOp; ++k)
      body.push_back(ctx.assignStmt(ctx.var(grand[k]), ctx.var(out[k])));
  }
  for (int k = 0; k < nOp; ++k)
    body.push_back(ctx.assignStmt(ctx.var(runTotalOut[k]), ctx.var(grand[k])));

  SmallVector<std::string> carry(nOp);
  for (int k = 0; k < nOp; ++k) {
    carry[k] = fresh();
    body.push_back(ctx.declStmt(scT[k], carry[k], ctx.var(grand[k])));
  }
  std::string init = fresh();
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), init, ctx.lit("false")));
  for (int idx = 0; idx < nParts; ++idx) {
    int p = order[idx];
    msl::Expr *cond = ctx.binary(rev ? B::Lt : B::Gt, ctx.var(myPart),
                                 ctx.lit(std::to_string(p)));
    msl::Block ifBody;
    SmallVector<std::string> pv(nOp);
    for (int k = 0; k < nOp; ++k) {
      pv[k] = fresh();
      ifBody.push_back(ctx.declStmt(scT[k], pv[k], slot(k, p)));
    }
    // if (init) { carry = combine(carry, pv); } else { carry = pv; init = true; }
    msl::Block thenB, elseB;
    {
      SmallVector<std::string> out;
      if (!astCombineN(region, carry, pv, thenB, out))
        return false;
      for (int k = 0; k < nOp; ++k)
        thenB.push_back(ctx.assignStmt(ctx.var(carry[k]), ctx.var(out[k])));
    }
    for (int k = 0; k < nOp; ++k)
      elseB.push_back(ctx.assignStmt(ctx.var(carry[k]), ctx.var(pv[k])));
    elseB.push_back(ctx.assignStmt(ctx.var(init), ctx.lit("true")));
    ifBody.push_back(ctx.ifElseScope(ctx.var(init), std::move(thenB),
                                     std::move(elseB)));
    body.push_back(ctx.ifScope(cond, std::move(ifBody)));
  }
  for (int r : regs) {
    SmallVector<std::string> ar(nOp);
    for (int k = 0; k < nOp; ++k)
      ar[k] = accs[k][r];
    SmallVector<std::string> out;
    if (!astCombineN(region, carry, ar, body, out))
      return false;
    for (int k = 0; k < nOp; ++k)
      // accs[k][r] = (init ? out[k] : accs[k][r]);
      body.push_back(ctx.assignStmt(
          ctx.var(accs[k][r]),
          ctx.paren(ctx.ternary(ctx.var(init), ctx.var(out[k]),
                                ctx.var(accs[k][r])))));
  }
  body.push_back(ctx.hardBarrier(false));
  return true;
}

// Multi-block map_elementwise region -> state machine. Like astEmitBlockCFG but
// the map_elementwise.return terminator spills operands into caller `capture`
// slots then breaks. Appends the predeclarations + state machine to `body`.
void MSLEmitter::astEmitMapCFG(Region &region, ArrayRef<std::string> capture,
                              msl::Block &body) {
  blockLabel.clear();
  int idx = 0;
  for (Block &blk : region)
    blockLabel[&blk] = std::to_string(idx++);
  for (Block &blk : llvm::drop_begin(region))
    for (BlockArgument arg : blk.getArguments()) {
      if (isDatalessType(arg.getType())) {
        valMap[arg] = SmallVector<std::string>{};
        continue;
      }
      valMap[arg] = astDeclResultVars(arg, body);
    }
  llvm::DenseMap<Value, SmallVector<std::string>> hoist;
  for (Block &blk : region)
    for (Operation &op : blk)
      for (Value res : op.getResults()) {
        if (isDatalessType(res.getType()))
          continue;
        bool crosses = llvm::any_of(res.getUsers(), [&](Operation *u) {
          return u->getBlock() != &blk;
        });
        if (crosses)
          hoist[res] = astDeclResultVars(res, body);
      }
  std::string state = fresh();
  cfgState = state;
  llvm::SmallVector<std::pair<std::string, msl::Block>> cases;
  for (Block &blk : region) {
    msl::Block caseBody = astWalkBlock2(blk, hoist);
    Operation *term = blk.getTerminator();
    if (term->getName().getStringRef() == "tt.map_elementwise.return") {
      for (auto [i, operand] : llvm::enumerate(term->getOperands()))
        caseBody.push_back(astMapReturnSpill(capture[i], names(operand)[0]));
      caseBody.push_back(ctx.breakStmt());
    } else {
      for (msl::Stmt *s : astTerminatorEdge(term, state))
        caseBody.push_back(s);
    }
    cases.push_back({blockLabel[&blk], std::move(caseBody)});
  }
  cfgState.clear();
  body.push_back(astMapCFGStateMachine(state, cases));
}

// Fused GEMM K-loop: carry iter-args, run the dot Decl phase, the K-loop with
// the MMA-phase dot in its body, the non-acc carry, direct-store setup, then the
// Readback-phase dot.
bool MSLEmitter::astEmitFusedGemm(scf::ForOp op, tt::DotOp dot, unsigned iterIdx,
                                  msl::Block &body) {
  SmallVector<SmallVector<std::string>> carried;
  SmallVector<std::string> initBase;
  for (auto [i, init, res] :
       llvm::enumerate(op.getInitArgs(), op.getResults())) {
    if (isDatalessType(res.getType())) {
      valMap[op.getRegionIterArg(i)] = SmallVector<std::string>{};
      valMap[res] = SmallVector<std::string>{};
      carried.push_back({});
      continue;
    }
    auto &initNames = names(init);
    if (i == iterIdx) {
      SmallVector<std::string> ids = astDeclResultVars(res, body);
      initBase.assign(initNames.begin(), initNames.end());
      fusedDot.ids = ids;
      valMap[op.getRegionIterArg(i)] = ids;
      valMap[res] = ids;
      carried.push_back({});
      continue;
    }
    SmallVector<std::string> vars = astDeclResultVars(res, body);
    for (size_t r = 0; r < vars.size(); ++r)
      body.push_back(ctx.assignStmt(
          ctx.var(vars[r]), ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
    valMap[op.getRegionIterArg(i)] = vars;
    valMap[res] = vars;
    carried.push_back(vars);
  }

  fusedDot.baseNames = initBase;
  fusedDot.phase = FusedDotPhase::Decl;
  if (!astEmitDot(dot, body))
    return false;

  std::string iv = fresh();
  bindScalar(op.getInductionVar(), iv);
  std::string ivTy = mslScalarType(op.getInductionVar().getType());
  if (ivTy.empty())
    ivTy = "int";

  // Loop body: MMA-phase dot (walked) + non-acc carry.
  fusedDot.phase = FusedDotPhase::MMA;
  msl::Block loopBody = astWalkBlock(op.getRegion().front(), (unsigned)indent + 1);
  if (emitFailed)
    return false;
  fusedDot.phase = FusedDotPhase::None;
  auto *term = op.getBody()->getTerminator();
  for (auto [i, operand] : llvm::enumerate(term->getOperands())) {
    if (i == iterIdx || carried[i].empty())
      continue;
    auto &src = names(operand);
    for (size_t r = 0; r < carried[i].size(); ++r)
      loopBody.push_back(ctx.assignStmt(
          ctx.var(carried[i][r]), ctx.var(src[src.size() == 1 ? 0 : r])));
  }
  body.push_back(astForScope(op, std::move(loopBody), iv, ivTy));

  DirectStore ds;
  if (matchDirectStore(op.getResult(iterIdx), ds)) {
    int64_t M = cast<RankedTensorType>(dot.getResult().getType()).getShape()[0];
    int64_t N = cast<RankedTensorType>(dot.getResult().getType()).getShape()[1];
    std::string ft = fresh();
    ds.fullTileVar = ft;
    msl::Expr *cond;
    if (ds.boundM) {
      // (rowBase + M <= boundM && colBase + N <= boundN)
      cond = ctx.paren(ctx.binary(
          B::LAnd,
          ctx.binary(B::Le,
                     ctx.binary(B::Add, ctx.var(names(ds.rowBase)[0]),
                                ctx.lit(std::to_string(M))),
                     ctx.var(names(ds.boundM)[0])),
          ctx.binary(B::Le,
                     ctx.binary(B::Add, ctx.var(names(ds.colBase)[0]),
                                ctx.lit(std::to_string(N))),
                     ctx.var(names(ds.boundN)[0]))));
    } else {
      cond = ctx.lit("true");
    }
    body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), ft, cond));
    fusedDot.direct = ds;
    directStoreHandled[ds.store.getOperation()] = ft;
  }

  fusedDot.phase = FusedDotPhase::Readback;
  if (!astEmitDot(dot, body))
    return false;
  fusedDot = FusedDotCtx{};
  return true;
}

// Predeclare a value's per-register result variables (`sc id;`) with no init,
// mirroring declResultVars; returns the minted names (caller binds valMap).
SmallVector<std::string>
MSLEmitter::astDeclResultVars(Value v, msl::Block &body) {
  Type elem = v.getType();
  if (auto rt = dyn_cast<RankedTensorType>(elem))
    elem = rt.getElementType();
  msl::Type *sc = isa<tt::PointerType>(elem)
                      ? astStorageType(v.getType())
                      : astScalarType(elementScalarType(v.getType()));
  int rc = regCount(v);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    body.push_back(ctx.declStmt(sc, id, nullptr));
    ids.push_back(id);
  }
  return ids;
}

// Per-register decls: mints fresh() names (advancing nextId so downstream ops
// stay in lockstep) and binds valMap[result].
bool MSLEmitter::astDeclBind(Operation *op, msl::Type *declTy, msl::Block &body,
                             llvm::function_ref<msl::Expr *(int)> mk) {
  int rc = regCount(op->getResult(0));
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    body.push_back(ctx.declStmt(declTy, id, mk(r)));
    ids.push_back(id);
  }
  valMap[op->getResult(0)] = ids;
  return true;
}

// `*p`, or the coherent-cast deref `*(device coherent(device) sc*)p` when a
// scalar spinlock forces a coherent access.
msl::Expr *MSLEmitter::astDerefPtr(Value, StringRef name, StringRef scName) {
  if (scalarSpinlock) {
    // Exact `device coherent(device) sc*` cast form used by the load/store path
    // (the printer's plain coherent-ptr form omits the leading `device`).
    msl::Type *cp = ctx.named("device coherent(device) " + scName.str() + "*");
    return ctx.deref(ctx.cast(CS::CStyle, cp, ctx.var(name)));
  }
  return ctx.deref(ctx.var(name));
}

// Shared banded threadgroup round-trip (trans/reshape): for each band, barrier +
// scatter each src register to buf[srcOff] + barrier + gather each res register
// from buf[resOff]. `band == total` uses the direct `buf[off]=v;` form; a smaller
// band wraps each in `{ int __f=off; if (__f>=lo && __f<hi) buf[__f-lo]=v; }`.
void MSLEmitter::astBandRoundTrip(
    msl::Block &body, StringRef buf, int64_t total, int64_t band, int srcRc,
    int resRc, ArrayRef<std::string> outs,
    llvm::function_ref<msl::Expr *(int)> srcOff,
    llvm::function_ref<msl::Expr *(int)> srcVal,
    llvm::function_ref<msl::Expr *(int)> resOff) {
  auto banded = [&](msl::Expr *off, int64_t lo, int64_t hi, bool toBuf,
                    msl::Expr *reg) -> msl::Stmt * {
    msl::Block b;
    b.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), "__f", off));
    msl::Expr *cond = ctx.binary(
        B::LAnd, ctx.binary(B::Ge, ctx.var("__f"), ctx.lit(std::to_string(lo))),
        ctx.binary(B::Lt, ctx.var("__f"), ctx.lit(std::to_string(hi))));
    msl::Expr *idx = ctx.binary(B::Sub, ctx.var("__f"), ctx.lit(std::to_string(lo)));
    msl::Expr *slot = ctx.subscript(ctx.var(buf), idx);
    msl::Stmt *asn = toBuf ? ctx.assignStmt(slot, reg) : ctx.assignStmt(reg, slot);
    b.push_back(ctx.compactIf(cond, asn));
    return ctx.plainScope(std::move(b));
  };
  for (int64_t lo = 0; lo < total; lo += band) {
    int64_t hi = std::min(lo + band, total);
    body.push_back(ctx.hardBarrier(false));
    for (int r = 0; r < srcRc; ++r) {
      msl::Expr *off = srcOff(r);
      msl::Expr *sv = srcVal(r);
      if (band == total)
        body.push_back(ctx.assignStmt(ctx.subscript(ctx.var(buf), off), sv));
      else
        body.push_back(banded(off, lo, hi, /*toBuf=*/true, sv));
    }
    body.push_back(ctx.hardBarrier(false));
    for (int r = 0; r < resRc; ++r) {
      msl::Expr *off = resOff(r);
      if (band == total)
        body.push_back(
            ctx.assignStmt(ctx.var(outs[r]), ctx.subscript(ctx.var(buf), off)));
      else
        body.push_back(banded(off, lo, hi, /*toBuf=*/false, ctx.var(outs[r])));
    }
  }
}

// Per-register store: `[if (guard)] *p = v;` with the thread predicate + mask
// guard.
void MSLEmitter::astStoreBody(tt::StoreOp op, msl::Block &body) {
  auto &ptrs = names(op.getPtr());
  auto &vals = names(op.getValue());
  bool hasMask = op.getMask() != nullptr;
  SmallVector<std::string> *mask = hasMask ? &names(op.getMask()) : nullptr;
  bool uniform = !isa<RankedTensorType>(op.getPtr().getType());
  int rc = ptrs.size();

  unsigned laneFree = 0, warpFree = 0;
  if (!uniform) {
    auto ptrTy = cast<RankedTensorType>(op.getPtr().getType());
    tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
    MLIRContext *c = op.getContext();
    auto masks = ll.getFreeVariableMasks();
    laneFree = masks.lookup(StringAttr::get(c, "lane"));
    warpFree = masks.lookup(StringAttr::get(c, "warp"));
  }
  // Thread predicate as an Expr.
  msl::Expr *threadPred = nullptr;
  if (uniform) {
    threadPred = ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0"));
  } else {
    if (laneFree)
      threadPred = ctx.paren(ctx.binary(
          B::Eq,
          ctx.paren(ctx.binary(B::And, ctx.var(laneId),
                               ctx.lit(std::to_string(laneFree)))),
          ctx.lit("0")));
    if (warpFree) {
      msl::Expr *wp = ctx.paren(ctx.binary(
          B::Eq,
          ctx.paren(ctx.binary(B::And, ctx.var(warpId),
                               ctx.lit(std::to_string(warpFree)))),
          ctx.lit("0")));
      threadPred = threadPred ? ctx.binary(B::LAnd, threadPred, wp) : wp;
    }
  }

  std::string scName = mslScalarType(elementScalarType(op.getValue().getType()));
  for (int r = 0; r < rc; ++r) {
    msl::Expr *lhs = astDerefPtr(op.getPtr(), ptrs[r], scName);
    msl::Expr *v = ctx.var(vals[vals.size() == 1 ? 0 : r]);
    msl::Stmt *assign = ctx.assignStmt(lhs, v);
    msl::Expr *guard = threadPred;
    if (hasMask) {
      msl::Expr *m = ctx.var((*mask)[mask->size() == 1 ? 0 : r]);
      guard = guard ? ctx.binary(B::LAnd, guard, m) : m;
    }
    if (guard)
      body.push_back(ctx.compactIf(guard, assign));
    else
      body.push_back(assign);
  }
}

static const char *axisComp(tt::ProgramIDDim axis) {
  return axis == tt::ProgramIDDim::X   ? "x"
         : axis == tt::ProgramIDDim::Y ? "y"
                                       : "z";
}

// `int id = (int)(builtinVar.comp);` and bind the result.
void MSLEmitter::astProgramDim(Operation *op, StringRef builtinVar,
                               tt::ProgramIDDim axis, msl::Block &body) {
  msl::Expr *e = ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::I32),
                          ctx.paren(ctx.member(ctx.var(builtinVar),
                                               axisComp(axis))));
  std::string id = fresh();
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), id, e));
  bindScalar(op->getResult(0), id);
}

// Route `op` to its builder(s), appending nodes to `body`. Returns true when
// handled (including alias/dataless ops that append nothing); false for an
// unsupported op, which the caller turns into a hard error.
bool MSLEmitter::astEmitOp(Operation *op, msl::Block &body) {
  // Barriers: BarrierStmt nodes (printer peephole collapses adjacent ones).
  if (auto b = dyn_cast<ttg::BarrierOp>(op)) {
    uint32_t bits = static_cast<uint32_t>(b.getAddrSpace());
    bool device = bits & (static_cast<uint32_t>(ttg::AddrSpace::GlobalRead) |
                          static_cast<uint32_t>(ttg::AddrSpace::GlobalWrite));
    body.push_back(ctx.barrier(device));
    return true;
  }
  if (isa<mlir::gpu::BarrierOp>(op)) {
    body.push_back(ctx.barrier(/*device=*/true));
    return true;
  }
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op)) {
    body.push_back(ctx.barrier(/*device=*/false));
    for (Value r : op->getResults())
      valMap[r] = SmallVector<std::string>{};
    return true;
  }

  // Structural no-ops that neither emit text nor bind a named value.
  if (isa<ttg::LocalDeallocOp, scf::YieldOp, scf::ConditionOp>(op))
    return true;
  if (op->getName().getStringRef() == "llvm.intr.assume")
    return true;
  if (isa<tt::AssertOp, tt::PrintOp>(op)) {
    for (Value r : op->getResults())
      valMap[r] = SmallVector<std::string>{};
    return true;
  }

  // Op-family sub-dispatchers, tried in the original arm order. Each returns
  // nullopt for an op it doesn't own (fall through to the next family), or the
  // handled/failed bool. Unmatched by all -> false (a hard error upstream).
  using Family = std::optional<bool> (MSLEmitter::*)(Operation *, msl::Block &);
  static constexpr Family families[] = {
      &MSLEmitter::astEmitArithBinop, &MSLEmitter::astEmitConstGrid,
      &MSLEmitter::astEmitArithMisc,  &MSLEmitter::astEmitMath,
      &MSLEmitter::astEmitReshape,    &MSLEmitter::astEmitMemDesc,
      &MSLEmitter::astEmitDotMap,     &MSLEmitter::astEmitAtomic,
      &MSLEmitter::astEmitScanReduce, &MSLEmitter::astEmitTensorMove,
      &MSLEmitter::astEmitCallReturn, &MSLEmitter::astEmitControlFlow};
  for (Family f : families)
    if (std::optional<bool> r = (this->*f)(op, body))
      return *r;

  // Unsupported op: astWalkBlock turns this false into a hard error.
  return false;
}

// float / int / bitwise / shift binops.
std::optional<bool> MSLEmitter::astEmitArithBinop(Operation *op,
                                                  msl::Block &body) {
  auto opnd = [&](Value v, int r) -> StringRef {
    auto &nm = names(v);
    return nm[nm.size() == 1 ? 0 : r];
  };
  Type resElem = op->getNumResults()
                     ? elementScalarType(op->getResult(0).getType())
                     : Type();

  // Float binaries: `sc id = (a o b);`
  if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp,
          tt::PreciseDivFOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astElementwiseExpr(arithBinOp(op), nullptr,
                                opnd(op->getOperand(0), r),
                                opnd(op->getOperand(1), r));
    });

  // Integer add/sub/mul/div/rem (astIntBinaryExpr handles the i1 and unsigned
  // paths; the decl type must match the unsigned promotion).
  if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::DivSIOp,
          arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
    msl::Type *declTy = astScalarType(resElem);
    if (auto it = dyn_cast<IntegerType>(resElem); it && it.getWidth() == 1)
      declTy = ctx.scalar(msl::Scalar::I1);
    else if (isa<arith::DivUIOp, arith::RemUIOp>(op))
      declTy = astUnsignedType(resElem);
    return astDeclBind(op, declTy, body, [&](int r) {
      return astIntBinaryExpr(op, opnd(op->getOperand(0), r),
                              opnd(op->getOperand(1), r));
    });
  }

  // Bitwise/logical and/or/xor.
  if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astElementwiseExpr(arithBinOp(op), nullptr,
                                opnd(op->getOperand(0), r),
                                opnd(op->getOperand(1), r));
    });
  }

  // Shifts.
  if (isa<arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astShiftExpr(op, opnd(op->getOperand(0), r),
                          opnd(op->getOperand(1), r));
    });

  return std::nullopt;
}

// program_id / num_programs / arith.constant / make_range.
std::optional<bool> MSLEmitter::astEmitConstGrid(Operation *op,
                                                 msl::Block &body) {
  // Program-id / num-programs: `int id = (int)(builtin.comp);`
  if (auto p = dyn_cast<tt::GetProgramIdOp>(op)) {
    astProgramDim(op, tgposId, p.getAxis(), body);
    return true;
  }
  if (auto n = dyn_cast<tt::GetNumProgramsOp>(op)) {
    astProgramDim(op, numTgId, n.getAxis(), body);
    return true;
  }

  // arith.constant: scalar / splat-tensor / dense-table.
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    Value res = cst.getResult();
    if (auto rt = dyn_cast<RankedTensorType>(res.getType())) {
      auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense)
        return false; // unsupported: caller emits the error
      Type elemTy = rt.getElementType();
      msl::Type *sc = astScalarType(elemTy);
      bool isFloat = isa<FloatType>(elemTy);
      if (dense.isSplat()) {
        std::string lit =
            isFloat ? floatLit(dense.getSplatValue<APFloat>(), elemTy)
                    : std::to_string(dense.getSplatValue<APInt>().getSExtValue());
        return astDeclBind(op, sc, body,
                           [&](int) { return ctx.lit(lit); });
      }
      // Dense table: `sc tbl[N] = {..}; sc id = tbl[flatTileOffset];`
      SmallVector<msl::Expr *> init;
      if (isFloat)
        for (const APFloat &v : dense.getValues<APFloat>())
          init.push_back(astFloatLit(v, elemTy));
      else
        for (const APInt &v : dense.getValues<APInt>())
          init.push_back(ctx.lit(std::to_string(v.getSExtValue())));
      std::string tbl = fresh();
      body.push_back(ctx.arrayDeclStmt(sc, tbl, dense.getNumElements(), init));
      return astDeclBind(op, sc, body, [&](int r) {
        return ctx.subscript(ctx.var(tbl), astFlatTileOffset(rt, r));
      });
    }
    msl::Type *sc = astScalarType(res.getType());
    msl::Expr *lit;
    if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
      lit = astFloatLit(fa.getValue(), res.getType());
    else if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
      lit = ctx.lit(std::to_string(ia.getInt()));
    else
      return false;
    std::string id = fresh();
    body.push_back(ctx.declStmt(sc, id, lit));
    bindScalar(res, id);
    return true;
  }

  // make_range: `int id = start + off;`
  if (auto mr = dyn_cast<tt::MakeRangeOp>(op)) {
    auto rt = cast<RankedTensorType>(mr.getResult().getType());
    int start = mr.getStart();
    return astDeclBind(op, ctx.scalar(msl::Scalar::I32), body, [&](int r) {
      return astMakeRangeElem(start, astLayoutOffsetExpr(rt, r));
    });
  }

  return std::nullopt;
}

// cmp / select / clamp / casts / negf / min-max family / precise_sqrt.
std::optional<bool> MSLEmitter::astEmitArithMisc(Operation *op,
                                                 msl::Block &body) {
  auto opnd = [&](Value v, int r) -> StringRef {
    auto &nm = names(v);
    return nm[nm.size() == 1 ? 0 : r];
  };
  Type resElem = op->getNumResults()
                     ? elementScalarType(op->getResult(0).getType())
                     : Type();
  // Integer compare: `bool id = (casta o castb);`
  if (auto ci = dyn_cast<arith::CmpIOp>(op)) {
    msl::BinOp bo;
    bool uns = false;
    switch (ci.getPredicate()) {
    case arith::CmpIPredicate::ult: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::slt: bo = B::Lt; break;
    case arith::CmpIPredicate::ule: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::sle: bo = B::Le; break;
    case arith::CmpIPredicate::ugt: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::sgt: bo = B::Gt; break;
    case arith::CmpIPredicate::uge: uns = true; [[fallthrough]];
    case arith::CmpIPredicate::sge: bo = B::Ge; break;
    case arith::CmpIPredicate::eq: bo = B::Eq; break;
    case arith::CmpIPredicate::ne: bo = B::Ne; break;
    }
    msl::Type *opCast =
        uns ? astUnsignedType(elementScalarType(ci.getLhs().getType())) : nullptr;
    return astDeclBind(op, ctx.scalar(msl::Scalar::I1), body, [&](int r) {
      return astElementwiseExpr(bo, opCast, opnd(op->getOperand(0), r),
                                opnd(op->getOperand(1), r));
    });
  }
  if (auto cf = dyn_cast<arith::CmpFOp>(op)) {
    msl::BinOp bo;
    switch (cf.getPredicate()) {
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT: bo = B::Lt; break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE: bo = B::Le; break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT: bo = B::Gt; break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE: bo = B::Ge; break;
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ: bo = B::Eq; break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE: bo = B::Ne; break;
    default: return false; // unsupported predicate: caller emits the error
    }
    return astDeclBind(op, ctx.scalar(msl::Scalar::I1), body, [&](int r) {
      return astElementwiseExpr(bo, nullptr, opnd(op->getOperand(0), r),
                                opnd(op->getOperand(1), r));
    });
  }

  // Select: `sc id = c ? t : f;`
  if (auto s = dyn_cast<arith::SelectOp>(op)) {
    Type re = op->getResult(0).getType();
    if (auto rt = dyn_cast<RankedTensorType>(re))
      re = rt.getElementType();
    msl::Type *declTy = isa<tt::PointerType>(re)
                            ? astStorageType(op->getResult(0).getType())
                            : astScalarType(elementScalarType(re));
    return astDeclBind(op, declTy, body, [&](int r) {
      return astSelectExpr(opnd(s.getCondition(), r), opnd(s.getTrueValue(), r),
                           opnd(s.getFalseValue(), r));
    });
  }

  // Clamp.
  if (auto c = dyn_cast<tt::ClampFOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astClampExpr(c, opnd(c.getX(), r), opnd(c.getMin(), r),
                          opnd(c.getMax(), r));
    });

  // Casts (non fp-narrowing) / bitcast / ptr<->int.
  if (isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
          arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astCastExpr(op, opnd(op->getOperand(0), r));
    });
  // TruncF / FpToFp: f32->half/bfloat narrowing emits a self-contained multi-line
  // block (RTZ or RTNE); other float casts are a plain static_cast. The narrowing
  // block is a captureRaw leaf (imperative multi-stmt) that still advances the
  // real nextId + binds valMap.
  if (isa<arith::TruncFOp, tt::FpToFpOp>(op)) {
    std::string dst = mslScalarType(elementScalarType(op->getResult(0).getType()));
    Type srcElem = elementScalarType(op->getOperand(0).getType());
    bool toHalf = dst == "half" || dst == "bfloat";
    bool rtz = false, narrow = false;
    if (auto f = dyn_cast<tt::FpToFpOp>(op)) {
      if (auto rnd = f.getRounding()) {
        narrow = srcElem.isF32() && toHalf;
        rtz = *rnd == tt::RoundingMode::RTZ;
      }
    } else if (srcElem.isF32() && toHalf) {
      narrow = true;
    }
    if (narrow) {
      auto &a = names(op->getOperand(0));
      int rc = regCount(op->getResult(0));
      SmallVector<std::string> ids;
      for (int r = 0; r < rc; ++r) {
        const std::string &v = a[a.size() == 1 ? 0 : r];
        std::string out;
        body.push_back(captureRaw([&] {
          out = rtz ? emitTruncatedFloatValue(dst, v)
                    : emitRoundedHalfValueFull(dst, v);
        }));
        ids.push_back(out);
      }
      valMap[op->getResult(0)] = ids;
      return true;
    }
    // Non-narrowing float cast: static_cast<dst>(v).
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astCastExpr(op, opnd(op->getOperand(0), r));
    });
  }
  if (isa<arith::BitcastOp, tt::BitcastOp>(op))
    return astDeclBind(op, astStorageType(op->getResult(0).getType()), body,
                       [&](int r) {
                         return astBitcastExpr(op, opnd(op->getOperand(0), r));
                       });
  if (isa<tt::IntToPtrOp, tt::PtrToIntOp>(op))
    return astDeclBind(op, astStorageType(op->getResult(0).getType()), body,
                       [&](int r) {
                         return astPtrIntCastExpr(op, opnd(op->getOperand(0), r));
                       });

  // negf: `sc id = -a;`
  if (isa<arith::NegFOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return ctx.unary(msl::UnOp::Neg, ctx.var(opnd(op->getOperand(0), r)));
    });

  // min/max family (decl type is always the signed scalar; opCast/propagateNan
  // set per op). Covers arith min/max, mulhi, remf(fmod).
  {
    StringRef fn;
    msl::Type *opCast = nullptr;
    bool propagateNan = false;
    bool isMinMax = true;
    if (isa<arith::MaximumFOp>(op)) { fn = "max"; propagateNan = true; }
    else if (isa<arith::MinimumFOp>(op)) { fn = "min"; propagateNan = true; }
    else if (isa<arith::MaxUIOp>(op)) {
      fn = "max"; opCast = astUnsignedType(resElem);
    } else if (isa<arith::MinUIOp>(op)) {
      fn = "min"; opCast = astUnsignedType(resElem);
    } else if (isa<arith::MaxNumFOp, arith::MaxSIOp>(op)) fn = "max";
    else if (isa<arith::MinNumFOp, arith::MinSIOp>(op)) fn = "min";
    else if (isa<arith::RemFOp>(op)) fn = "metal::fmod";
    else if (isa<tt::MulhiUIOp>(op)) {
      fn = "mulhi"; opCast = astUnsignedType(resElem);
    } else isMinMax = false;
    if (isMinMax)
      return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
        return astMinMaxExpr(fn, opCast, propagateNan,
                             opnd(op->getOperand(0), r),
                             opnd(op->getOperand(1), r));
      });
  }

  // precise_sqrt: `sc id = (sc)metal::precise::sqrt(a);`
  if (isa<tt::PreciseSqrtOp>(op))
    return astDeclBind(op, astScalarType(resElem), body, [&](int r) {
      return astUnaryExpr(msl::builtin::precise::Sqrt, astScalarType(resElem),
                          opnd(op->getOperand(0), r));
    });

  return std::nullopt;
}

// math.* dialect: unary/binary transcendentals, fma, exp10.
std::optional<bool> MSLEmitter::astEmitMath(Operation *op, msl::Block &body) {
  auto opnd = [&](Value v, int r) -> StringRef {
    auto &nm = names(v);
    return nm[nm.size() == 1 ? 0 : r];
  };
  Type resElem = op->getNumResults()
                     ? elementScalarType(op->getResult(0).getType())
                     : Type();
  if (op->getDialect() ==
      op->getContext()->getLoadedDialect<math::MathDialect>()) {
    msl::Type *sc = astScalarType(resElem);
    StringRef n = op->getName().getStringRef();
    namespace bi = msl::builtin;
    static const llvm::StringMap<StringRef> unary = {
        {"math.exp", bi::precise::Exp},   {"math.exp2", bi::precise::Exp2},
        {"math.log", bi::precise::Log},   {"math.log2", bi::precise::Log2},
        {"math.log10", bi::precise::Log10}, {"math.sin", bi::precise::Sin},
        {"math.cos", bi::precise::Cos},   {"math.tan", bi::precise::Tan},
        {"math.tanh", bi::precise::Tanh}, {"math.sinh", bi::precise::Sinh},
        {"math.cosh", bi::precise::Cosh}, {"math.asin", bi::precise::Asin},
        {"math.acos", bi::precise::Acos}, {"math.atan", bi::precise::Atan},
        {"math.sqrt", bi::precise::Sqrt}, {"math.rsqrt", bi::precise::Rsqrt},
        {"math.cbrt", bi::precise::Cbrt}, {"math.floor", bi::math::Floor},
        {"math.ceil", bi::math::Ceil},    {"math.absf", bi::math::Fabs},
        {"math.absi", bi::math::Abs},     {"math.erf", "tt_erf"},
        {"math.round", bi::math::Round},  {"math.trunc", bi::math::Trunc},
        {"math.roundeven", bi::math::Rint}};
    if (auto it = unary.find(n); it != unary.end()) {
      StringRef fn = it->second;
      return astDeclBind(op, sc, body, [&](int r) {
        return astUnaryExpr(fn, sc, opnd(op->getOperand(0), r));
      });
    }
    static const llvm::StringMap<StringRef> binary = {
        {"math.atan2", bi::precise::Atan2}, {"math.powf", bi::precise::Pow},
        {"math.fpowi", bi::precise::Pow}, {"math.copysign", bi::math::Copysign}};
    if (auto it = binary.find(n); it != binary.end()) {
      StringRef fn = it->second;
      return astDeclBind(op, sc, body, [&](int r) {
        return astMinMaxExpr(fn, nullptr, false, opnd(op->getOperand(0), r),
                             opnd(op->getOperand(1), r));
      });
    }
    if (n == "math.fma")
      return astDeclBind(op, sc, body, [&](int r) {
        return astTernaryCallExpr(bi::math::Fma, opnd(op->getOperand(0), r),
                                  opnd(op->getOperand(1), r),
                                  opnd(op->getOperand(2), r));
      });
    if (n == "math.exp10")
      return astDeclBind(op, sc, body, [&](int r) {
        // pow((sc)10, a)
        msl::Expr *ten = ctx.cast(CS::CStyle, sc, ctx.lit("10"));
        return ctx.call(bi::precise::Pow, {ten, ctx.var(opnd(op->getOperand(0), r))});
      });
    return false; // unhandled math op: caller emits the error
  }

  return std::nullopt;
}

// Shape ops: splat/unsplat/expand/broadcast/join/split (rebinds) + trans/reshape.
std::optional<bool> MSLEmitter::astEmitReshape(Operation *op, msl::Block &body) {
  // Pure register-rebind ops (no text emitted): splat / expand_dims / broadcast /
  // join / split / unsplat. Their handlers only rewrite valMap, so calling them
  // here writes nothing and keeps the symbol table correct.
  if (auto sp = dyn_cast<tt::SplatOp>(op))
    return succeeded(emitSplat(sp));
  if (auto u = dyn_cast<tt::UnsplatOp>(op)) {
    bindScalar(u.getResult(), names(u.getSrc())[0]);
    return true;
  }
  if (auto e = dyn_cast<tt::ExpandDimsOp>(op))
    return succeeded(emitReshapeLike(e.getResult(), e.getSrc(), e.getAxis(), true));
  if (auto b = dyn_cast<tt::BroadcastOp>(op))
    return succeeded(emitReshapeLike(b.getResult(), b.getSrc(), -1, false));
  if (auto j = dyn_cast<tt::JoinOp>(op))
    return succeeded(emitJoin(j));
  if (auto sp = dyn_cast<tt::SplitOp>(op))
    return succeeded(emitSplit(sp));

  // tt.trans: round-trip through a threadgroup buffer keyed by row-major offset.
  if (auto tr = dyn_cast<tt::TransOp>(op)) {
    if (!isa<RankedTensorType>(tr.getResult().getType()))
      return false;
    Value src = tr.getSrc();
    Value res = tr.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    auto perm = tr.getOrder();
    msl::Type *scTy = astScalarType(resTy.getElementType());
    std::string sc = mslScalarType(resTy.getElementType());
    auto &srcNames = names(src);
    int srcRc = regCount(src);
    int resRc = regCount(res);
    int64_t elemBytes = bitsOf(resTy.getElementType()) / 8;
    int64_t total = tileSize(resTy);

    std::string buf = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                                astPoolRegion(0, sc)));
    int64_t band = total * elemBytes > 32768
                       ? reshapeBandElems(total, elemBytes)
                       : total;
    SmallVector<std::string> ids = astDeclResultVars(res, body);
    astBandRoundTrip(
        body, buf, total, band, srcRc, resRc, ids,
        [&](int r) { return astTransFlatOffset(srcTy, perm, resTy.getShape(), r); },
        [&](int r) { return static_cast<msl::Expr *>(
                         ctx.var(srcNames[srcNames.size() == 1 ? 0 : r])); },
        [&](int r) { return astFlatTileOffset(resTy, r); });
    valMap[res] = ids;
    return true;
  }

  // tt.reshape: row-major flat-offset identity round-trip (same skeleton as
  // trans, src offset uses the source's own flat offset).
  if (auto rs = dyn_cast<tt::ReshapeOp>(op)) {
    if (!isa<RankedTensorType>(rs.getResult().getType()))
      return false;
    Value src = rs.getSrc();
    Value res = rs.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    msl::Type *scTy = astScalarType(resTy.getElementType());
    std::string sc = mslScalarType(resTy.getElementType());
    auto &srcNames = names(src);
    int srcRc = regCount(src);
    int resRc = regCount(res);
    int64_t elemBytes = bitsOf(resTy.getElementType()) / 8;
    int64_t total = tileSize(resTy);

    std::string buf = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                                astPoolRegion(0, sc)));
    int64_t band = total * elemBytes > 32768
                       ? reshapeBandElems(total, elemBytes)
                       : total;
    SmallVector<std::string> outs = astDeclResultVars(res, body);
    astBandRoundTrip(
        body, buf, total, band, srcRc, resRc, outs,
        [&](int r) { return astFlatTileOffset(srcTy, r); },
        [&](int r) { return static_cast<msl::Expr *>(
                         ctx.var(srcNames[srcNames.size() == 1 ? 0 : r])); },
        [&](int r) { return astFlatTileOffset(resTy, r); });
    valMap[res] = outs;
    return true;
  }

  return std::nullopt;
}

// memdesc index/subslice, local_alloc/store/load, async_copy, convert_layout.
std::optional<bool> MSLEmitter::astEmitMemDesc(Operation *op, msl::Block &body) {
  // memdesc_index / memdesc_subslice: pure memdescMap rebinds (no text).
  if (auto mi = dyn_cast<ttg::MemDescIndexOp>(op))
    return succeeded(emitMemDescIndex(mi));
  if (auto ms = dyn_cast<ttg::MemDescSubsliceOp>(op))
    return succeeded(emitMemDescSubslice(ms));

  // ttg.local_alloc: declare a threadgroup buffer, optional init scatter.
  if (auto la = dyn_cast<ttg::LocalAllocOp>(op)) {
    auto mt = cast<ttg::MemDescType>(la.getResult().getType());
    // `threadgroup sc buf[N];` - the address space is part of the element type
    // spelling for a threadgroup array decl.
    msl::Type *scTy = ctx.named("threadgroup " + mslScalarType(mt.getElementType()));
    std::string buf = "__tg_buf_" + std::to_string(tgScratchId++);
    body.push_back(ctx.arrayDeclStmt(scTy, buf, memdescFlatSize(mt)));
    memdescMap[la.getResult()] = {buf, "0"};
    if (Value init = la.getSrc()) {
      auto srcTy = cast<RankedTensorType>(init.getType());
      auto &vals = names(init);
      body.push_back(ctx.hardBarrier(false));
      for (int r = 0, n = regCount(init); r < n; ++r)
        body.push_back(ctx.assignStmt(
            ctx.subscript(ctx.var(buf), astFlatTileOffset(srcTy, r)),
            ctx.var(vals[r])));
      body.push_back(ctx.hardBarrier(false));
    }
    return true;
  }

  // ttg.local_store: scatter src registers into the memdesc buffer + barrier.
  if (auto ls = dyn_cast<ttg::LocalStoreOp>(op)) {
    auto srcTy = cast<RankedTensorType>(ls.getSrc().getType());
    MemDescInfo dst = memdescMap[ls.getDst()];
    auto &vals = names(ls.getSrc());
    for (int r = 0, n = regCount(ls.getSrc()); r < n; ++r)
      body.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(dst.buf), astMemdescElemAddr(dst, srcTy, r)),
          ctx.var(vals[r])));
    body.push_back(ctx.barrier(false));
    return true;
  }

  // ttg.local_load: gather result registers from the memdesc buffer.
  if (auto ll = dyn_cast<ttg::LocalLoadOp>(op)) {
    if (localLoadIsDeadDotStage(ll)) {
      valMap[ll.getResult()] = SmallVector<std::string>{};
      return true;
    }
    auto resTy = cast<RankedTensorType>(ll.getResult().getType());
    MemDescInfo src = memdescMap[ll.getSrc()];
    msl::Type *scTy = astScalarType(resTy.getElementType());
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(ll.getResult()); r < n; ++r) {
      std::string id = fresh();
      body.push_back(ctx.declStmt(
          scTy, id,
          ctx.subscript(ctx.var(src.buf), astMemdescElemAddr(src, resTy, r))));
      ids.push_back(id);
    }
    valMap[ll.getResult()] = ids;
    return true;
  }

  // ttg.async_copy_global_to_local: synchronous masked per-thread stage + barrier.
  if (auto ac = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    auto srcTy = cast<RankedTensorType>(ac.getSrc().getType());
    MemDescInfo dst = memdescMap[ac.getResult()];
    auto &ptrs = names(ac.getSrc());
    bool hasMask = ac.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(ac.getMask()) : nullptr;
    for (int r = 0, n = regCount(ac.getSrc()); r < n; ++r) {
      msl::Expr *addr =
          ctx.subscript(ctx.var(dst.buf), astMemdescElemAddr(dst, srcTy, r));
      msl::Expr *load = ctx.deref(ctx.var(ptrs[r]));
      msl::Stmt *asn = ctx.assignStmt(addr, load);
      if (hasMask)
        body.push_back(ctx.compactIf(
            ctx.var((*mask)[mask->size() == 1 ? 0 : r]), asn));
      else
        body.push_back(asn);
    }
    body.push_back(ctx.barrier(false));
    valMap[ac.getResult()] = SmallVector<std::string>{};
    return true;
  }

  // ttg.convert_layout: full-tile threadgroup round-trip (3 banding modes).
  if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (convertLayoutIsDeadDotStage(cl) ||
        convertLayoutIsDeadDotStageSource(cl)) {
      valMap[cl.getResult()] = SmallVector<std::string>{};
      return true;
    }
    Value src = cl.getSrc(), res = cl.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    Type elemTy = resTy.getElementType();
    bool isPtr = isa<tt::PointerType>(elemTy);
    std::string ptrTyStr = mslStorageType(resTy);
    std::string scStr = isPtr ? "ulong" : ptrTyStr;
    msl::Type *scTy = isPtr ? ctx.scalar(msl::Scalar::U64) : astStorageType(resTy);
    msl::Type *ptrDeclTy = astStorageType(resTy);
    auto &srcNames = names(src);
    int64_t elemBytes = bitsOf(elemTy) / 8;
    int64_t tileBytes = tileSize(resTy) * elemBytes;
    int rank = resTy.getRank();
    ArrayRef<int64_t> shape = resTy.getShape();

    std::string buf = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                                astPoolRegion(0, scStr)));
    // scatter value expr: `(ulong)sv` when isPtr, else `sv`.
    auto scatterVal = [&](StringRef nm) -> msl::Expr * {
      msl::Expr *v = ctx.var(nm);
      return isPtr ? ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U64), v) : v;
    };
    // gather value expr: `(ptrTy)buf[off]` when isPtr, else `buf[off]`.
    auto gatherVal = [&](msl::Expr *slot) -> msl::Expr * {
      return isPtr ? ctx.cast(CS::CStyle, ptrDeclTy, slot) : slot;
    };

    int64_t convCap = poolBudget();
    if (tileBytes > convCap && rank >= 2) {
      int64_t N = shape[rank - 1];
      int64_t bandRows = convCap / (N * elemBytes);
      if (bandRows < 1) bandRows = 1;
      int64_t rowsTotal = shape[rank - 2];
      auto srcOut = llvm::to_vector(ttg::toLinearLayout(srcTy).getOutDimNames());
      StringAttr srcRowDim = srcOut[rank - 2];
      auto resOut = llvm::to_vector(ttg::toLinearLayout(resTy).getOutDimNames());
      StringAttr resRowDim = resOut[rank - 2];
      SmallVector<std::string> ids(regCount(res));

      auto bandOffset = [&](RankedTensorType rt, int reg,
                            int64_t r0) -> msl::Expr * {
        auto outN = llvm::to_vector(ttg::toLinearLayout(rt).getOutDimNames());
        msl::Expr *expr = nullptr;
        int64_t stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
          msl::Expr *c = astLayoutCoordExpr(rt, reg, outN[d]);
          if (d == rank - 2)
            c = ctx.paren(ctx.binary(B::Sub, c, ctx.lit(std::to_string(r0))));
          msl::Expr *term = stride == 1
                                ? c
                                : ctx.paren(ctx.binary(
                                      B::Mul, c, ctx.lit(std::to_string(stride))));
          expr = expr ? ctx.paren(ctx.binary(B::Add, expr, term)) : term;
          stride *= (d == rank - 2) ? bandRows : shape[d];
        }
        return expr ? expr : ctx.lit("0");
      };
      for (int64_t r0 = 0; r0 < rowsTotal; r0 += bandRows) {
        int64_t r1 = std::min<int64_t>(r0 + bandRows, rowsTotal);
        body.push_back(ctx.hardBarrier(false));
        for (int r = 0, n = regCount(src); r < n; ++r) {
          msl::Expr *rowc = astLayoutCoordExpr(srcTy, r, srcRowDim);
          msl::Expr *cond = ctx.binary(
              B::LAnd, ctx.binary(B::Ge, rowc, ctx.lit(std::to_string(r0))),
              ctx.binary(B::Lt, astLayoutCoordExpr(srcTy, r, srcRowDim),
                         ctx.lit(std::to_string(r1))));
          body.push_back(ctx.compactIf(
              cond, ctx.assignStmt(
                        ctx.subscript(ctx.var(buf), bandOffset(srcTy, r, r0)),
                        scatterVal(srcNames[r]))));
        }
        body.push_back(ctx.hardBarrier(false));
        for (int r = 0, n = regCount(res); r < n; ++r) {
          if (ids[r].empty()) {
            ids[r] = fresh();
            body.push_back(ctx.declStmt(ptrDeclTy, ids[r], nullptr));
          }
          msl::Expr *rowc = astLayoutCoordExpr(resTy, r, resRowDim);
          msl::Expr *cond = ctx.binary(
              B::LAnd, ctx.binary(B::Ge, rowc, ctx.lit(std::to_string(r0))),
              ctx.binary(B::Lt, astLayoutCoordExpr(resTy, r, resRowDim),
                         ctx.lit(std::to_string(r1))));
          msl::Expr *rd = gatherVal(
              ctx.subscript(ctx.var(buf), bandOffset(resTy, r, r0)));
          body.push_back(
              ctx.compactIf(cond, ctx.assignStmt(ctx.var(ids[r]), rd)));
        }
      }
      valMap[res] = ids;
      return true;
    }

    if (tileBytes > convCap) {
      int64_t total = tileSize(resTy);
      int64_t band = reshapeBandElems(total, elemBytes, convCap);
      SmallVector<std::string> ids(regCount(res));
      for (int r = 0, n = regCount(res); r < n; ++r) {
        ids[r] = fresh();
        body.push_back(ctx.declStmt(ptrDeclTy, ids[r], nullptr));
      }
      for (int64_t lo = 0; lo < total; lo += band) {
        int64_t hi = std::min(lo + band, total);
        body.push_back(ctx.hardBarrier(false));
        for (int r = 0, n = regCount(src); r < n; ++r) {
          msl::Block b;
          b.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), "__f",
                                   astFlatTileOffset(srcTy, r)));
          msl::Expr *cond = ctx.binary(
              B::LAnd,
              ctx.binary(B::Ge, ctx.var("__f"), ctx.lit(std::to_string(lo))),
              ctx.binary(B::Lt, ctx.var("__f"), ctx.lit(std::to_string(hi))));
          msl::Expr *idx = ctx.binary(B::Sub, ctx.var("__f"),
                                      ctx.lit(std::to_string(lo)));
          b.push_back(ctx.compactIf(
              cond, ctx.assignStmt(ctx.subscript(ctx.var(buf), idx),
                                   scatterVal(srcNames[r]))));
          body.push_back(ctx.plainScope(std::move(b)));
        }
        body.push_back(ctx.hardBarrier(false));
        for (int r = 0, n = regCount(res); r < n; ++r) {
          msl::Block b;
          b.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), "__f",
                                   astFlatTileOffset(resTy, r)));
          msl::Expr *cond = ctx.binary(
              B::LAnd,
              ctx.binary(B::Ge, ctx.var("__f"), ctx.lit(std::to_string(lo))),
              ctx.binary(B::Lt, ctx.var("__f"), ctx.lit(std::to_string(hi))));
          msl::Expr *idx = ctx.binary(B::Sub, ctx.var("__f"),
                                      ctx.lit(std::to_string(lo)));
          msl::Expr *rd = gatherVal(ctx.subscript(ctx.var(buf), idx));
          b.push_back(
              ctx.compactIf(cond, ctx.assignStmt(ctx.var(ids[r]), rd)));
          body.push_back(ctx.plainScope(std::move(b)));
        }
      }
      valMap[res] = ids;
      return true;
    }

    body.push_back(ctx.hardBarrier(false));
    for (int r = 0, n = regCount(src); r < n; ++r)
      body.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(buf), astFlatTileOffset(srcTy, r)),
          scatterVal(srcNames[r])));
    body.push_back(ctx.hardBarrier(false));
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(res); r < n; ++r) {
      std::string id = fresh();
      msl::Expr *rd = gatherVal(
          ctx.subscript(ctx.var(buf), astFlatTileOffset(resTy, r)));
      body.push_back(ctx.declStmt(ptrDeclTy, id, rd));
      ids.push_back(id);
    }
    valMap[res] = ids;
    return true;
  }

  return std::nullopt;
}

// tt.dot / tt.map_elementwise / tt.histogram.
std::optional<bool> MSLEmitter::astEmitDotMap(Operation *op, msl::Block &body) {
  // tt.dot: simdgroup-matrix / scalar GEMM (all staging + fusion phases).
  if (auto dt = dyn_cast<tt::DotOp>(op))
    return astEmitDot(dt, body);

  // tt.map_elementwise: per-group inline of the scalar region. Single-block
  // groups AST-walk the region; multi-block groups use the state-machine
  // lowering (astEmitMapCFG).
  if (auto mp = dyn_cast<tt::MapElementwiseOp>(op)) {
    Region &region = mp.getScalarOp();
    Block &blk = region.front();
    int nSrc = mp.getNumOperands();
    int nRes = mp.getNumResults();
    int pack = mp.getPack();
    SmallVector<SmallVector<std::string>> srcNames(nSrc);
    for (int s = 0; s < nSrc; ++s)
      srcNames[s] = names(mp.getOperand(s));
    int nReg = srcNames[0].size();
    int nGroup = nReg / pack;
    bool multiBlock = !region.hasOneBlock();

    SmallVector<SmallVector<std::string>> resIds(nRes);
    for (int g = 0; g < nGroup; ++g) {
      for (int s = 0; s < nSrc; ++s)
        for (int p = 0; p < pack; ++p)
          bindScalar(blk.getArgument(s * pack + p), srcNames[s][g * pack + p]);
      if (multiBlock) {
        SmallVector<std::string> capture(nRes * pack);
        for (int i = 0; i < nRes * pack; ++i) {
          capture[i] = fresh();
          Value r = mp->getResult(i / pack);
          body.push_back(astMapCaptureDecl(
              mslScalarType(elementScalarType(r.getType())), capture[i]));
        }
        astEmitMapCFG(region, capture, body);
        if (emitFailed)
          return false;
        for (int k = 0; k < nRes; ++k)
          for (int p = 0; p < pack; ++p)
            resIds[k].push_back(capture[k * pack + p]);
        continue;
      }
      for (Operation &o : blk.without_terminator()) {
        if (!astEmitOp(&o, body)) {
          o.emitError("EmitMSL: unhandled map op '" +
                      o.getName().getStringRef() + "'");
          emitFailed = true;
        }
      }
      if (emitFailed)
        return false;
      Operation *term = blk.getTerminator();
      for (int k = 0; k < nRes; ++k)
        for (int p = 0; p < pack; ++p)
          resIds[k].push_back(names(term->getOperand(k * pack + p))[0]);
    }
    for (int k = 0; k < nRes; ++k)
      valMap[mp->getResult(k)] = resIds[k];
    return true;
  }

  // tt.histogram: zero-init threadgroup bins, per-register guarded fetch_add,
  // barrier, then per-result-register atomic_load.
  if (auto hg = dyn_cast<tt::HistogramOp>(op)) {
    auto srcTy = cast<RankedTensorType>(hg.getSrc().getType());
    auto resTy = cast<RankedTensorType>(hg.getResult().getType());
    int64_t nBins = tileSize(resTy);
    int64_t threads = 32;
    if (auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      threads = nw.getInt() * 32;

    std::string bins = fresh();
    body.push_back(astHistBinsDecl(bins));
    std::string zi = fresh();
    body.push_back(astHistZeroInit(bins, zi, nBins, threads));
    body.push_back(ctx.hardBarrier(false));

    auto &srcVals = names(hg.getSrc());
    SmallVector<std::string> *maskVals = hg.getMask() ? &names(hg.getMask()) : nullptr;
    std::string srcU = mslUnsignedType(elementScalarType(hg.getSrc().getType()));

    MLIRContext *mctx = hg.getContext();
    tt::LinearLayout srcLL = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(mctx, "lane");
    auto kWarp = StringAttr::get(mctx, "warp");
    auto srcOut = llvm::to_vector(srcLL.getOutDimNames());
    uint32_t freeMask = 0;
    auto scanFree = [&](StringAttr in, int shift) {
      if (!srcLL.hasInDim(in)) return;
      for (int b = 0, n = srcLL.getInDimSizeLog2(in); b < n; ++b) {
        bool moves = false;
        for (auto od : srcOut)
          if (srcLL.getBasis(in, b, od) != 0) moves = true;
        if (!moves) freeMask |= 1u << (shift + b);
      }
    };
    scanFree(kLane, 0);
    scanFree(kWarp, 5);
    // (tidId.x & freeMask u) == 0u
    msl::Expr *ownerGuard =
        freeMask == 0
            ? nullptr
            : ctx.binary(B::Eq,
                         ctx.paren(ctx.binary(
                             B::And, ctx.member(ctx.var(tidId), "x"),
                             ctx.u32lit(freeMask))),
                         ctx.lit("0u"));
    for (int r = 0; r < (int)srcVals.size(); ++r) {
      const std::string &v = srcVals[r];
      // (srcU)v < nBins u
      msl::Expr *guard =
          ctx.binary(B::Lt, ctx.cast(CS::CStyle, ctx.named(srcU), ctx.var(v)),
                     ctx.u32lit(nBins));
      if (ownerGuard)
        guard = ctx.binary(B::LAnd, ownerGuard, ctx.paren(guard));
      if (maskVals)
        guard = ctx.binary(
            B::LAnd,
            ctx.paren(ctx.var((*maskVals)[maskVals->size() == 1 ? 0 : r])),
            ctx.paren(guard));
      body.push_back(astHistFetchAdd(guard, bins, v));
    }
    body.push_back(ctx.hardBarrier(false));

    auto outDims = llvm::to_vector(ttg::toLinearLayout(resTy).getOutDimNames());
    msl::Type *resScTy = astScalarType(resTy.getElementType());
    std::string resSc = mslScalarType(resTy.getElementType());
    int nResReg = regCount(hg.getResult());
    SmallVector<std::string> resIds;
    for (int r = 0; r < nResReg; ++r) {
      msl::Expr *idx = astLayoutCoordExpr(resTy, r, outDims[0]);
      std::string id = fresh();
      // (resSc)atomic_load_explicit(&bins[idx], memory_order_relaxed)
      msl::Expr *load = ctx.cast(
          CS::CStyle, resScTy,
          ctx.call("atomic_load_explicit",
                   {ctx.addrOf(ctx.subscript(ctx.var(bins), idx)),
                    ctx.lit("memory_order_relaxed")}));
      body.push_back(ctx.declStmt(resScTy, id, load));
      resIds.push_back(id);
    }
    valMap[hg.getResult()] = resIds;
    return true;
  }

  return std::nullopt;
}

// tt.atomic_rmw / tt.atomic_poll / tt.atomic_cas.
std::optional<bool> MSLEmitter::astEmitAtomic(Operation *op, msl::Block &body) {
  // tt.atomic_rmw: native fetch_* (AST) or fp-emulated CAS loop (captured), with
  // redundant-thread guard + lane/warp replica broadcast.
  if (auto ar = dyn_cast<tt::AtomicRMWOp>(op)) {
    Value res = ar.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    msl::Type *scTy = astScalarType(scalarTy);
    bool isFloat = isa<FloatType>(scalarTy);
    unsigned bw = scalarTy.getIntOrFloatBitWidth();
    tt::RMWOp kind = ar.getAtomicRmwOp();
    if (!isFloat && bw == 64)
      return false;
    bool floatNative = isFloat && bw == 32 &&
                       (kind == tt::RMWOp::ADD || kind == tt::RMWOp::FADD);
    bool floatEmulated = isFloat && !floatNative;
    if (floatEmulated && bw != 16 && bw != 32)
      return false;
    std::string atomicTy = isFloat ? "atomic_float"
                           : (kind == tt::RMWOp::UMAX || kind == tt::RMWOp::UMIN)
                               ? "atomic_uint"
                               : (bw == 64 ? "atomic_long" : "atomic_int");
    const char *fn = nullptr;
    if (!floatEmulated) {
      switch (kind) {
      case tt::RMWOp::ADD:
      case tt::RMWOp::FADD: fn = "atomic_fetch_add_explicit"; break;
      case tt::RMWOp::MAX:
      case tt::RMWOp::UMAX: fn = "atomic_fetch_max_explicit"; break;
      case tt::RMWOp::MIN:
      case tt::RMWOp::UMIN: fn = "atomic_fetch_min_explicit"; break;
      case tt::RMWOp::AND: fn = "atomic_fetch_and_explicit"; break;
      case tt::RMWOp::OR: fn = "atomic_fetch_or_explicit"; break;
      case tt::RMWOp::XOR: fn = "atomic_fetch_xor_explicit"; break;
      case tt::RMWOp::XCHG: fn = "atomic_exchange_explicit"; break;
      default: return false;
      }
    }
    auto &ptrs = names(ar.getPtr());
    auto &vals = names(ar.getVal());
    bool hasMask = ar.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(ar.getMask()) : nullptr;
    bool uniform = !isa<RankedTensorType>(ar.getPtr().getType());
    int rc = ptrs.size();

    unsigned laneFree = 0, warpFree = 0, regFree = 0;
    if (!uniform) {
      auto ptrTy = cast<RankedTensorType>(ar.getPtr().getType());
      tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
      MLIRContext *c = ar.getContext();
      auto masks = ll.getFreeVariableMasks();
      laneFree = masks.lookup(StringAttr::get(c, "lane"));
      warpFree = masks.lookup(StringAttr::get(c, "warp"));
      regFree = masks.lookup(StringAttr::get(c, "register"));
    }
    std::string threadPred;
    if (laneFree)
      threadPred = "((" + laneId + " & " + std::to_string(laneFree) + ") == 0)";
    if (warpFree) {
      std::string wp = "((" + warpId + " & " + std::to_string(warpFree) + ") == 0)";
      threadPred = threadPred.empty() ? wp : threadPred + " && " + wp;
    }
    tt::MemSemantic sem = ar.getSem();

    SmallVector<std::string> ids(rc);
    for (int r = 0; r < rc; ++r) {
      if (regFree && (r & regFree) != 0) {
        ids[r] = ids[r & ~regFree];
        continue;
      }
      const std::string &p = ptrs[r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      std::string id = fresh();
      body.push_back(ctx.declStmt(scTy, id, astInit0(sc)));
      msl::Expr *guard = nullptr;
      if (uniform)
        guard = ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0"));
      else if (!threadPred.empty())
        guard = ctx.var(threadPred);
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        guard = guard ? static_cast<msl::Expr *>(ctx.binary(B::LAnd, guard,
                                                            ctx.var(m)))
                      : ctx.var(m);
      }
      msl::Block inner;
      if (floatEmulated) {
        // Self-contained emulated CAS loop: captured verbatim (advances nextId).
        // Bake indentation for the guard-body depth (inner nests inside ifScope).
        int savedInd = indent;
        if (guard)
          ++indent;
        inner.push_back(captureRaw([&] {
          std::string cur = fresh();
          if (bw == 16) {
            std::string vh = emitRoundedHalfValue(sc, v);
            std::string newExpr = floatRmwExpr(kind, cur, vh);
            auto base = emitPacked16Base(p, sc);
            emitPacked16CASLoop(base.first, base.second, sc, cur, newExpr, id);
          } else {
            std::string newExpr = floatRmwExpr(kind, cur, "(float)(" + v + ")");
            emitFloat32CASLoop(p, cur, newExpr, id);
          }
        }));
        indent = savedInd;
      } else {
        // Metal device atomics are relaxed-only; acquire/release/acq_rel are
        // not valid MSL memory orders (the acq_rel form fails to compile).
        (void)sem;
        inner.push_back(ctx.assignStmt(
            ctx.var(id),
            astAtomicRmwCall(fn, atomicTy, p, v, "memory_order_relaxed",
                             /*memFlags=*/false)));
      }
      if (guard)
        body.push_back(ctx.ifScope(guard, std::move(inner)));
      else
        for (msl::Stmt *s : inner)
          body.push_back(s);
      if (!uniform && laneFree) {
        std::string src = "(uint)(" + laneId + " & " +
                          std::to_string(~laneFree & 31) + ")";
        id = astShuffle("simd_shuffle", sc, id, src, body);
      }
      ids[r] = id;
    }

    if (!uniform && warpFree) {
      auto ptrTy = cast<RankedTensorType>(ar.getPtr().getType());
      tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
      MLIRContext *c = ar.getContext();
      int64_t numWarps = ll.hasInDim(StringAttr::get(c, "warp"))
                             ? ll.getInDimSize(StringAttr::get(c, "warp")) : 1;
      std::string bcbuf = fresh();
      body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup),
                                  bcbuf, astPoolRegion(0, sc)));
      body.push_back(ctx.hardBarrier(false));
      // ((warpId & warpFree) == 0)
      msl::Expr *wcanon = ctx.paren(ctx.binary(
          B::Eq,
          ctx.paren(ctx.binary(B::And, ctx.var(warpId), ctx.i32lit(warpFree))),
          ctx.i32lit(0)));
      // (warpId & (~warpFree & (numWarps-1)))
      msl::Expr *warpKey = ctx.paren(ctx.binary(
          B::And, ctx.var(warpId),
          ctx.i32lit(~warpFree & (numWarps - 1))));
      // ((warpKey * rc*32) + reg*32 + (laneId & (~laneFree & 31)))
      auto slotFor = [&](int reg) -> msl::Expr * {
        return ctx.paren(ctx.addChain(
            {ctx.paren(ctx.mul(warpKey, ctx.i32lit(rc * 32))),
             ctx.mul(ctx.i32lit(reg), ctx.i32lit(32)),
             ctx.paren(ctx.binary(B::And, ctx.var(laneId),
                                  ctx.i32lit(~laneFree & 31)))}));
      };
      for (int r = 0; r < rc; ++r) {
        if (regFree && (r & regFree) != 0) continue;
        body.push_back(ctx.compactIf(
            wcanon,
            ctx.assignStmt(ctx.subscript(ctx.var(bcbuf), slotFor(r)),
                           ctx.var(ids[r]))));
      }
      body.push_back(ctx.hardBarrier(false));
      for (int r = 0; r < rc; ++r) {
        int srcr = regFree ? (r & ~regFree) : r;
        std::string bc = fresh();
        body.push_back(ctx.declStmt(
            scTy, bc, ctx.subscript(ctx.var(bcbuf), slotFor(srcr))));
        ids[r] = bc;
      }
    }
    valMap[res] = ids;
    return true;
  }

  // tt.atomic_poll: elected-thread spin/probe on the aligned word + broadcast.
  if (auto pl = dyn_cast<tt::AtomicPollOp>(op)) {
    Type expTy = pl.getExpected().getType();
    unsigned bw = expTy.getIntOrFloatBitWidth();
    if (bw != 16 && bw != 32 && bw != 64)
      return false;
    bool acquire = pl.getSem() == tt::MemSemantic::ACQUIRE;
    const std::string &p = names(pl.getPtr())[0];
    const std::string &exp = names(pl.getExpected())[0];
    // Poll barriers spell the flags device-first for acquire (differs from the
    // peephole's canonical order), so emit as a raw call ExprStmt.
    std::string barrierFlags =
        acquire ? "mem_flags::mem_device | mem_flags::mem_threadgroup"
                : "mem_flags::mem_threadgroup";
    msl::Stmt *pollBarrier =
        ctx.exprStmt(ctx.raw("threadgroup_barrier(" + barrierFlags + ")"));
    std::string wordTy = bw == 16 ? "ushort" : (bw == 64 ? "ulong" : "uint");

    // Bind wordPtr decls into `into`, returning the load-expr string.
    auto probe = [&](msl::Block &into) -> std::string {
      std::string wordPtr = fresh();
      std::string loadExpr;
      if (bw == 16) {
        std::string isHigh = fresh();
        into.push_back(ctx.declStmt(
            ctx.scalar(msl::Scalar::I1), isHigh,
            ctx.raw("((size_t)(" + p + ") & 2u) != 0u")));
        into.push_back(ctx.declStmt(
            ctx.named("device atomic_uint *"), wordPtr,
            ctx.raw("(device atomic_uint *)((size_t)(" + p + ") & ~(size_t)3)")));
        loadExpr = "(ushort)((" + isHigh +
                   ") ? (atomic_load_explicit(" + wordPtr +
                   ", memory_order_relaxed) >> 16) : (atomic_load_explicit(" +
                   wordPtr + ", memory_order_relaxed) & 0xffffu))";
      } else if (bw == 64) {
        into.push_back(ctx.declStmt(
            ctx.named("volatile device ulong *"), wordPtr,
            ctx.raw("(volatile device ulong *)(" + p + ")")));
        loadExpr = "(*" + wordPtr + ")";
      } else {
        into.push_back(ctx.declStmt(
            ctx.named("device atomic_uint *"), wordPtr,
            ctx.raw("(device atomic_uint *)(" + p + ")")));
        loadExpr = "atomic_load_explicit(" + wordPtr + ", memory_order_relaxed)";
      }
      return loadExpr;
    };

    std::string result = fresh();
    if (!pl.getTimeout()) {
      msl::Block ifBody;
      std::string loadExpr = probe(ifBody);
      std::string want = fresh();
      ifBody.push_back(ctx.declStmt(ctx.named(wordTy), want,
                                    ctx.raw("(" + wordTy + ")" + exp)));
      ifBody.push_back(ctx.whileScope(
          ctx.binary(B::Ne, ctx.raw(loadExpr), ctx.var(want)), msl::Block{}));
      body.push_back(ctx.ifScope(
          ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
          std::move(ifBody)));
      body.push_back(pollBarrier);
      body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), result,
                                  ctx.lit("true")));
      valMap[pl.getResult()] = {result};
      return true;
    }
    std::string flag = fresh();
    body.push_back(ctx.declStmt(ctx.named("threadgroup bool"), flag, nullptr));
    msl::Block ifBody;
    std::string loadExpr = probe(ifBody);
    std::string want = fresh(), loaded = fresh();
    ifBody.push_back(ctx.declStmt(ctx.named(wordTy), want,
                                  ctx.raw("(" + wordTy + ")" + exp)));
    ifBody.push_back(ctx.declStmt(ctx.named(wordTy), loaded, ctx.raw(loadExpr)));
    ifBody.push_back(ctx.assignStmt(
        ctx.var(flag),
        ctx.paren(ctx.binary(B::Eq, ctx.var(loaded), ctx.var(want)))));
    body.push_back(ctx.ifScope(
        ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
        std::move(ifBody)));
    body.push_back(pollBarrier);
    body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), result,
                                ctx.var(flag)));
    valMap[pl.getResult()] = {result};
    return true;
  }

  // tt.atomic_cas: int32/float32/packed16 compare-exchange + uniform spinlock.
  if (auto ca = dyn_cast<tt::AtomicCASOp>(op)) {
    Value res = ca.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    msl::Type *scTy = astScalarType(scalarTy);
    bool isFloat = isa<FloatType>(scalarTy);
    unsigned bw = scalarTy.getIntOrFloatBitWidth();
    bool packed16 = isFloat && bw == 16;
    bool word32 = bw == 32;
    if (!packed16 && !word32)
      return false;
    auto &ptrs = names(ca.getPtr());
    auto &cmps = names(ca.getCmp());
    auto &vals = names(ca.getVal());
    bool uniform = !isa<RankedTensorType>(ca.getPtr().getType());
    int rc = ptrs.size();
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      const std::string &p = ptrs[r];
      const std::string &c = cmps[cmps.size() == 1 ? 0 : r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      std::string id = fresh();
      msl::Block casBody;
      // The CAS leaf declares `id` when !uniform; when uniform we pre-declare it.
      if (packed16) {
        std::string wp, ih;
        for (msl::Stmt *s : astPacked16Base(p, wp, ih))
          casBody.push_back(s);
        for (msl::Stmt *s : astPacked16CAS(wp, ih, c, v, sc, id, !uniform))
          casBody.push_back(s);
      } else if (isFloat)
        for (msl::Stmt *s : astFloat32CAS(p, c, v, id, !uniform))
          casBody.push_back(s);
      else
        for (msl::Stmt *s : astInt32CAS(p, c, v, sc, id, !uniform))
          casBody.push_back(s);

      if (uniform) {
        body.push_back(ctx.declStmt(scTy, id, ctx.var(c)));
        body.push_back(ctx.ifScope(
            ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
            std::move(casBody)));
        // Broadcast lane-0's result to every lane through a scratch slot.
        std::string bcast = fresh();
        body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup),
                                    bcast, astPoolRegion(0, sc)));
        body.push_back(ctx.compactIf(
            ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
            ctx.assignStmt(ctx.subscript(ctx.var(bcast), ctx.lit("0")),
                           ctx.var(id))));
        body.push_back(astAcquireFence());
        body.push_back(ctx.hardBarrier(false));
        body.push_back(ctx.assignStmt(
            ctx.var(id), ctx.subscript(ctx.var(bcast), ctx.lit("0"))));
        body.push_back(ctx.hardBarrier(false));
      } else {
        for (msl::Stmt *s : casBody)
          body.push_back(s);
      }
      ids.push_back(id);
    }
    valMap[res] = ids;
    return true;
  }

  return std::nullopt;
}

// tt.scan / tt.reduce: cross-lane/warp fold + prefix.
std::optional<bool> MSLEmitter::astEmitScanReduce(Operation *op,
                                                  msl::Block &body) {
  // tt.scan: per-run register fold + lane shuffle prefix + cross-warp carry +
  // cross-run carry.
  if (auto sn = dyn_cast<tt::ScanOp>(op)) {
    bool rev = sn.getReverse();
    int nOp = sn.getNumOperands();
    auto srcTy = cast<RankedTensorType>(sn.getOperand(0).getType());
    int axis = sn.getAxis();
    Region &region = sn.getRegion();
    SmallVector<std::string> scTys(nOp);
    SmallVector<msl::Type *> scTypes(nOp);
    SmallVector<int64_t> byteWidths(nOp);
    for (int k = 0; k < nOp; ++k) {
      scTys[k] = mslScalarType(elementScalarType(sn.getResult()[k].getType()));
      scTypes[k] = astScalarType(elementScalarType(sn.getResult()[k].getType()));
      byteWidths[k] = bitsOf(elementScalarType(sn.getResult()[k].getType())) / 8;
    }
    SmallVector<SmallVector<std::string>> srcNames(nOp);
    for (int k = 0; k < nOp; ++k)
      srcNames[k] = names(sn.getOperand(k));
    int nReg = srcNames[0].size();

    MLIRContext *mctx = sn.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(mctx, "lane");
    auto kWarp = StringAttr::get(mctx, "warp");
    auto outDims = llvm::to_vector(ll.getOutDimNames());
    auto outDim = outDims[axis];
    auto laneBits = axisBits(ll, kLane, outDim);
    auto warpBits = axisBits(ll, kWarp, outDim);

    unsigned axisLaneMask = 0;
    for (auto &pr : laneBits) axisLaneMask |= (1u << pr.first);
    unsigned axisLaneLow = axisLaneMask & (~axisLaneMask + 1);
    unsigned normMask = axisLaneMask / (axisLaneLow ? axisLaneLow : 1);
    if (axisLaneMask && (normMask & (normMask + 1)))
      return false; // unsupported lane layout: caller emits the error
    unsigned axisWarpMask = 0;
    for (auto &pr : warpBits) axisWarpMask |= (1u << pr.first);
    int numWarps = ll.hasInDim(kWarp) ? ll.getInDimSize(kWarp) : 1;

    int32_t laneWarpReach = 0;
    for (auto &pr : laneBits) laneWarpReach = std::max(laneWarpReach, pr.second);
    for (auto &pr : warpBits) laneWarpReach = std::max(laneWarpReach, pr.second);

    auto keyOf = [&](int reg) {
      SmallVector<int32_t> coords = registerCoords(srcTy, reg);
      std::string key;
      for (int d = 0; d < (int)coords.size(); ++d)
        if (d != axis) key += std::to_string(coords[d]) + ",";
      return key;
    };
    SmallVector<int> runId(nReg, 0);
    for (int r = 0; r < nReg; ++r) {
      int32_t c = registerCoords(srcTy, r)[axis];
      runId[r] = laneWarpReach ? (c / (2 * laneWarpReach)) : 0;
    }

    SmallVector<SmallVector<std::string>> accs(nOp, SmallVector<std::string>(nReg));
    for (int k = 0; k < nOp; ++k)
      for (int r = 0; r < nReg; ++r) {
        accs[k][r] = fresh();
        body.push_back(ctx.declStmt(scTypes[k], accs[k][r],
                                    ctx.var(srcNames[k][r])));
      }

    const char *shuf = rev ? "simd_shuffle_down" : "simd_shuffle_up";
    std::string axisTopLane =
        axisLaneMask == 0
            ? laneId
            : ("((" + laneId + " & " + std::to_string(~axisLaneMask) + ") | " +
               (rev ? "0" : std::to_string(axisLaneMask)) + ")");

    std::map<std::string, SmallVector<int>> keys;
    SmallVector<std::string> keyOrder;
    for (int r = 0; r < nReg; ++r) {
      std::string k = keyOf(r);
      if (keys.find(k) == keys.end()) keyOrder.push_back(k);
      keys[k].push_back(r);
    }

    for (std::string &key : keyOrder) {
      SmallVector<int> &keyRegs = keys[key];
      SmallVector<int> runOrder;
      for (int r : keyRegs)
        if (llvm::find(runOrder, runId[r]) == runOrder.end())
          runOrder.push_back(runId[r]);
      llvm::sort(runOrder, [&](int a, int b) { return rev ? a > b : a < b; });
      SmallVector<SmallVector<std::string>> runTotals(
          runOrder.size(), SmallVector<std::string>(nOp));

      for (size_t ri = 0; ri < runOrder.size(); ++ri) {
        int run = runOrder[ri];
        SmallVector<int> regs;
        for (int r : keyRegs)
          if (runId[r] == run) regs.push_back(r);
        llvm::sort(regs, [&](int a, int b) {
          int32_t ca = registerCoords(srcTy, a)[axis];
          int32_t cb = registerCoords(srcTy, b)[axis];
          return rev ? ca > cb : ca < cb;
        });

        for (size_t i = 1; i < regs.size(); ++i) {
          SmallVector<std::string> a(nOp), b(nOp);
          for (int k = 0; k < nOp; ++k) {
            a[k] = accs[k][regs[i - 1]];
            b[k] = accs[k][regs[i]];
          }
          SmallVector<std::string> out;
          if (!astCombineN(region, a, b, body, out)) return false;
          for (int k = 0; k < nOp; ++k)
            body.push_back(ctx.assignStmt(ctx.var(accs[k][regs[i]]),
                                          ctx.var(out[k])));
        }

        SmallVector<std::string> laneScan(nOp);
        for (int k = 0; k < nOp; ++k) {
          laneScan[k] = fresh();
          body.push_back(ctx.declStmt(scTypes[k], laneScan[k],
                                      ctx.var(accs[k][regs.back()])));
        }
        for (auto &pr : laneBits) {
          unsigned delta = 1u << pr.first;
          SmallVector<std::string> nb(nOp);
          for (int k = 0; k < nOp; ++k)
            nb[k] = astShuffle(shuf, scTys[k], laneScan[k],
                               std::to_string(delta) + "u", body);
          SmallVector<std::string> out;
          if (!astCombineN(region, nb, laneScan, body, out)) return false;
          // (laneId & axisLaneMask) [<= mask-delta | >= delta]
          msl::Expr *local = ctx.paren(
              ctx.binary(B::And, ctx.var(laneId), ctx.i32lit(axisLaneMask)));
          msl::Expr *guard =
              rev ? ctx.binary(B::Le, local, ctx.i32lit(axisLaneMask - delta))
                  : ctx.binary(B::Ge, local, ctx.i32lit(delta));
          for (int k = 0; k < nOp; ++k)
            body.push_back(ctx.assignStmt(
                ctx.var(laneScan[k]),
                ctx.paren(ctx.ternary(guard, ctx.var(out[k]),
                                      ctx.var(laneScan[k])))));
        }
        if (!laneBits.empty()) {
          SmallVector<std::string> lanePrefix(nOp);
          for (int k = 0; k < nOp; ++k)
            lanePrefix[k] = astShuffle(shuf, scTys[k], laneScan[k],
                                       std::to_string(axisLaneLow) + "u", body);
          // (laneId & axisLaneMask) [<= mask-low | >= low]
          msl::Expr *local = ctx.paren(
              ctx.binary(B::And, ctx.var(laneId), ctx.i32lit(axisLaneMask)));
          msl::Expr *guard =
              rev ? ctx.binary(B::Le, local,
                               ctx.i32lit(axisLaneMask - axisLaneLow))
                  : ctx.binary(B::Ge, local, ctx.i32lit(axisLaneLow));
          for (int r : regs) {
            SmallVector<std::string> out, ar(nOp);
            for (int k = 0; k < nOp; ++k) ar[k] = accs[k][r];
            if (!astCombineN(region, lanePrefix, ar, body, out)) return false;
            for (int k = 0; k < nOp; ++k)
              body.push_back(ctx.assignStmt(
                  ctx.var(accs[k][r]),
                  ctx.paren(ctx.ternary(guard, ctx.var(out[k]),
                                        ctx.var(accs[k][r])))));
          }
        }

        for (int k = 0; k < nOp; ++k) {
          runTotals[ri][k] = fresh();
          body.push_back(ctx.declStmt(scTypes[k], runTotals[ri][k], nullptr));
        }
        if (!astScanWarpCarry(region, nOp, scTys, byteWidths, warpBits, regs,
                              accs, laneScan, axisTopLane, axisWarpMask, numWarps,
                              rev, runTotals[ri], body))
          return false;
      }

      for (size_t ri = 1; ri < runOrder.size(); ++ri) {
        SmallVector<std::string> carry = runTotals[0];
        for (size_t j = 1; j < ri; ++j) {
          SmallVector<std::string> out;
          if (!astCombineN(region, carry, runTotals[j], body, out)) return false;
          carry = out;
        }
        int run = runOrder[ri];
        for (int r : keyRegs) {
          if (runId[r] != run) continue;
          SmallVector<std::string> ar(nOp);
          for (int k = 0; k < nOp; ++k) ar[k] = accs[k][r];
          SmallVector<std::string> out;
          if (!astCombineN(region, carry, ar, body, out)) return false;
          for (int k = 0; k < nOp; ++k)
            body.push_back(ctx.assignStmt(ctx.var(accs[k][r]), ctx.var(out[k])));
        }
      }
    }
    for (int k = 0; k < nOp; ++k)
      valMap[sn.getResult()[k]] = accs[k];
    return true;
  }

  // tt.reduce: per-group register fold + lane-shuffle xor + optional cross-warp
  // threadgroup combine.
  if (auto rd = dyn_cast<tt::ReduceOp>(op)) {
    int nOp = rd.getNumOperands();
    auto srcTy = cast<RankedTensorType>(rd.getOperand(0).getType());
    int axis = rd.getAxis();
    bool tensorResult = isa<RankedTensorType>(rd.getResult()[0].getType());
    SmallVector<std::string> scTys(nOp);
    SmallVector<msl::Type *> scTypes(nOp);
    for (int k = 0; k < nOp; ++k) {
      scTys[k] = mslScalarType(elementScalarType(rd.getResult()[k].getType()));
      scTypes[k] = astScalarType(elementScalarType(rd.getResult()[k].getType()));
    }
    Region &region = rd.getCombineOp();
    SmallVector<SmallVector<std::string>> srcNames(nOp);
    for (int k = 0; k < nOp; ++k)
      srcNames[k] = names(rd.getOperand(k));
    int nReg = srcNames[0].size();

    MLIRContext *mctx = rd.getContext();
    tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(mctx, "lane");
    auto kWarp = StringAttr::get(mctx, "warp");
    auto outDims = llvm::to_vector(ll.getOutDimNames());
    auto redDim = outDims[axis];

    auto survKey = [&](int reg) {
      SmallVector<int32_t> coords = registerCoords(srcTy, reg);
      std::string key;
      for (int d = 0; d < (int)coords.size(); ++d)
        if (d != axis) key += std::to_string(coords[d]) + ",";
      return key;
    };
    auto fullKey = [&](int reg) {
      SmallVector<int32_t> coords = registerCoords(srcTy, reg);
      std::string key;
      for (int32_t c : coords) key += std::to_string(c) + ",";
      return key;
    };
    std::map<std::string, SmallVector<int>> groups;
    std::set<std::string> seenFull;
    for (int r = 0; r < nReg; ++r)
      if (seenFull.insert(fullKey(r)).second)
        groups[survKey(r)].push_back(r);

    unsigned laneMask = reduceMask(ll, kLane, redDim);
    unsigned warpMask = reduceMask(ll, kWarp, redDim);
    int numWarps = ll.hasInDim(kWarp) ? ll.getInDimSize(kWarp) : 1;
    std::map<std::string, SmallVector<std::string>> groupResult;

    for (auto &g : groups) {
      SmallVector<int> &regs = g.second;
      SmallVector<std::string> accs(nOp);
      for (int k = 0; k < nOp; ++k) {
        accs[k] = fresh();
        body.push_back(ctx.declStmt(scTypes[k], accs[k],
                                    ctx.var(srcNames[k][regs[0]])));
      }
      for (size_t i = 1; i < regs.size(); ++i) {
        SmallVector<std::string> bVals(nOp);
        for (int k = 0; k < nOp; ++k) bVals[k] = srcNames[k][regs[i]];
        SmallVector<std::string> out;
        if (!astCombineN(region, accs, bVals, body, out))
          return false;
        for (int k = 0; k < nOp; ++k)
          body.push_back(ctx.assignStmt(ctx.var(accs[k]), ctx.var(out[k])));
      }
      for (int bit = 31; bit >= 0; --bit) {
        unsigned m = 1u << bit;
        if ((laneMask & m) == 0) continue;
        SmallVector<std::string> others(nOp);
        for (int k = 0; k < nOp; ++k)
          others[k] = astShuffle("simd_shuffle_xor", scTys[k], accs[k],
                                 std::to_string(m) + "u", body);
        SmallVector<std::string> out;
        if (!astCombineN(region, accs, others, body, out))
          return false;
        for (int k = 0; k < nOp; ++k)
          body.push_back(ctx.assignStmt(ctx.var(accs[k]), ctx.var(out[k])));
      }
      if (warpMask != 0) {
        SmallVector<std::string> scratch(nOp);
        int64_t byteOff = 0;
        for (int k = 0; k < nOp; ++k) {
          scratch[k] = fresh();
          body.push_back(ctx.declStmt(
              ctx.ptr(scTypes[k], msl::AddrSpace::Threadgroup), scratch[k],
              astPoolRegion(byteOff, scTys[k])));
          byteOff += numWarps * 32 *
                     std::max<int64_t>(1, bitsOf(elementScalarType(
                                              rd.getResult()[k].getType())) / 8);
        }
        body.push_back(ctx.hardBarrier(false));
        // scratch[k][warp * 32 + lane] = accs[k];
        for (int k = 0; k < nOp; ++k) {
          msl::Expr *idx = ctx.binary(
              B::Add, ctx.binary(B::Mul, ctx.var(warpId), ctx.lit("32")),
              ctx.var(laneId));
          body.push_back(ctx.assignStmt(
              ctx.subscript(ctx.var(scratch[k]), idx), ctx.var(accs[k])));
        }
        body.push_back(ctx.hardBarrier(false));
        SmallVector<int> redVals = subsetsOf(warpMask, numWarps);
        // base = ((warp & ~warpMask) * 32 + lane)
        msl::Expr *base = ctx.paren(ctx.binary(
            B::Add,
            ctx.binary(B::Mul,
                       ctx.paren(ctx.binary(B::And, ctx.var(warpId),
                                            ctx.lit(std::to_string(~warpMask)))),
                       ctx.lit("32")),
            ctx.var(laneId)));
        SmallVector<std::string> wacc(nOp);
        for (int k = 0; k < nOp; ++k) {
          wacc[k] = fresh();
          body.push_back(ctx.declStmt(scTypes[k], wacc[k],
                                      ctx.subscript(ctx.var(scratch[k]), base)));
        }
        for (size_t i = 1; i < redVals.size(); ++i) {
          SmallVector<std::string> wv(nOp);
          for (int k = 0; k < nOp; ++k) {
            wv[k] = fresh();
            msl::Expr *idx = ctx.binary(B::Add, base,
                                        ctx.lit(std::to_string(redVals[i] * 32)));
            body.push_back(ctx.declStmt(scTypes[k], wv[k],
                                        ctx.subscript(ctx.var(scratch[k]), idx)));
          }
          SmallVector<std::string> out;
          if (!astCombineN(region, wacc, wv, body, out))
            return false;
          for (int k = 0; k < nOp; ++k)
            body.push_back(ctx.assignStmt(ctx.var(wacc[k]), ctx.var(out[k])));
        }
        accs = wacc;
      }
      groupResult[g.first] = accs;
    }

    if (!tensorResult) {
      for (int k = 0; k < nOp; ++k)
        bindScalar(rd.getResult()[k], groupResult.begin()->second[k]);
      return true;
    }
    auto resTy = cast<RankedTensorType>(rd.getResult()[0].getType());
    int nResReg = regCount(rd.getResult()[0]);
    SmallVector<SmallVector<std::string>> resIds(nOp);
    for (int r = 0; r < nResReg; ++r) {
      SmallVector<int32_t> rc = registerCoords(resTy, r);
      std::string key;
      for (int32_t c : rc) key += std::to_string(c) + ",";
      auto it = groupResult.find(key);
      if (it == groupResult.end())
        return false;
      for (int k = 0; k < nOp; ++k)
        resIds[k].push_back(it->second[k]);
    }
    for (int k = 0; k < nOp; ++k)
      valMap[rd.getResult()[k]] = resIds[k];
    return true;
  }

  return std::nullopt;
}

// tt.cat / tt.gather: threadgroup scatter/gather tile moves.
std::optional<bool> MSLEmitter::astEmitTensorMove(Operation *op,
                                                  msl::Block &body) {
  // tt.cat: scatter both halves (rhs shifted past lhs flat size), gather result.
  if (auto ct = dyn_cast<tt::CatOp>(op)) {
    Value lhs = ct.getLhs(), rhs = ct.getRhs(), res = ct.getResult();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    auto rhsTy = cast<RankedTensorType>(rhs.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    msl::Type *scTy = astScalarType(resTy.getElementType());
    std::string sc = mslScalarType(resTy.getElementType());
    auto &lhsNames = names(lhs);
    auto &rhsNames = names(rhs);
    int64_t lhsFlat = tileSize(lhsTy);

    std::string buf = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                                astPoolRegion(0, sc)));
    body.push_back(ctx.hardBarrier(false));
    for (int r = 0, n = regCount(lhs); r < n; ++r)
      body.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(buf), astFlatTileOffset(lhsTy, r)),
          ctx.var(lhsNames[r])));
    for (int r = 0, n = regCount(rhs); r < n; ++r) {
      msl::Expr *off = ctx.binary(B::Add, astFlatTileOffset(rhsTy, r),
                                  ctx.lit(std::to_string(lhsFlat)));
      body.push_back(ctx.assignStmt(ctx.subscript(ctx.var(buf), off),
                                    ctx.var(rhsNames[r])));
    }
    body.push_back(ctx.hardBarrier(false));
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(res); r < n; ++r) {
      std::string id = fresh();
      body.push_back(ctx.declStmt(
          scTy, id, ctx.subscript(ctx.var(buf), astFlatTileOffset(resTy, r))));
      ids.push_back(id);
    }
    valMap[res] = ids;
    return true;
  }

  // tt.gather: stage src tile, then read each result register at the
  // index-selected source offset (row-major fold, dim `axis` uses idx).
  if (auto ga = dyn_cast<tt::GatherOp>(op)) {
    Value src = ga.getSrc(), idx = ga.getIndices(), res = ga.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    int axis = ga.getAxis();
    msl::Type *scTy = astScalarType(elementScalarType(resTy));
    std::string sc = mslScalarType(elementScalarType(resTy));
    auto &srcNames = names(src);
    auto &idxNames = names(idx);
    int srcRc = regCount(src);
    int resRc = regCount(res);
    auto srcShape = srcTy.getShape();
    tt::LinearLayout resLL = ttg::toLinearLayout(resTy);
    auto resOut = llvm::to_vector(resLL.getOutDimNames());

    std::string buf = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                                astPoolRegion(0, sc)));
    body.push_back(ctx.hardBarrier(false));
    for (int r = 0; r < srcRc; ++r)
      body.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(buf), astFlatTileOffset(srcTy, r)),
          ctx.var(srcNames[srcNames.size() == 1 ? 0 : r])));
    body.push_back(ctx.hardBarrier(false));

    SmallVector<std::string> outs;
    for (int r = 0; r < resRc; ++r) {
      msl::Expr *off = nullptr;
      int64_t stride = 1;
      for (int d = (int)srcShape.size() - 1; d >= 0; --d) {
        msl::Expr *c =
            (d == axis)
                ? static_cast<msl::Expr *>(ctx.cast(
                      CS::CStyle, ctx.scalar(msl::Scalar::I32),
                      ctx.paren(ctx.var(idxNames[idxNames.size() == 1 ? 0 : r]))))
                : astLayoutCoordExpr(resTy, r, resOut[d]);
        msl::Expr *term =
            stride == 1
                ? c
                : ctx.paren(ctx.binary(B::Mul, c, ctx.lit(std::to_string(stride))));
        off = off ? ctx.paren(ctx.binary(B::Add, off, term)) : term;
        stride *= srcShape[d];
      }
      if (!off)
        off = ctx.lit("0");
      std::string id = fresh();
      body.push_back(ctx.declStmt(scTy, id, ctx.subscript(ctx.var(buf), off)));
      outs.push_back(id);
    }
    valMap[res] = outs;
    return true;
  }

  return std::nullopt;
}

// return / call / ub.poison / addptr / load / store.
std::optional<bool> MSLEmitter::astEmitCallReturn(Operation *op,
                                                  msl::Block &body) {
  // tt.return
  if (auto ret = dyn_cast<tt::ReturnOp>(op)) {
    body.push_back(astReturn(ret));
    return true;
  }

  // tt.call: build the device-fn call expr, bind results (scalar/multi/tensor).
  if (auto cl = dyn_cast<tt::CallOp>(op)) {
    auto callee = mod.lookupSymbol<tt::FuncOp>(cl.getCalleeAttr().getValue());
    if (!callee)
      return false;
    llvm::SmallVector<msl::Expr *> args;
    for (Value operand : cl.getOperands()) {
      auto &nm = names(operand);
      if (nm.size() != 1)
        return false;
      args.push_back(ctx.var(nm[0]));
    }
    args.push_back(ctx.var(tgposId));
    args.push_back(ctx.var(tidId));
    args.push_back(ctx.var(numTgId));
    if (globalPoolBytes > 0)
      args.push_back(ctx.var(poolBuf.empty() ? "__pool" : poolBuf));
    auto call = ctx.call(mslDeviceFuncName(callee.getName()), args);

    unsigned nRes = cl.getNumResults();
    if (nRes == 1 && isa<RankedTensorType>(cl.getResult(0).getType())) {
      Value res = cl.getResult(0);
      msl::Type *scTy = astScalarType(
          cast<RankedTensorType>(res.getType()).getElementType());
      std::string tmp = fresh();
      body.push_back(ctx.declStmt(astDeviceRetType(callee), tmp, call));
      int rc = regCount(res);
      SmallVector<std::string> idsV;
      for (int i = 0; i < rc; ++i) {
        std::string id = fresh();
        body.push_back(ctx.declStmt(
            scTy, id, ctx.member(ctx.var(tmp), "f" + std::to_string(i))));
        idsV.push_back(id);
      }
      valMap[res] = idsV;
      return true;
    }
    if (nRes == 0) {
      body.push_back(ctx.exprStmt(call));
      return true;
    }
    if (nRes == 1) {
      if (!isa<IntegerType, FloatType>(cl.getResult(0).getType()))
        return false;
      std::string id = fresh();
      body.push_back(ctx.declStmt(astScalarType(cl.getResult(0).getType()), id,
                                  call));
      bindScalar(cl.getResult(0), id);
      return true;
    }
    std::string tmp = fresh();
    body.push_back(ctx.declStmt(astDeviceRetType(callee), tmp, call));
    for (auto [i, res] : llvm::enumerate(cl.getResults())) {
      std::string id = fresh();
      body.push_back(ctx.declStmt(astScalarType(res.getType()), id,
                                  ctx.member(ctx.var(tmp), "f" + std::to_string(i))));
      bindScalar(res, id);
    }
    return true;
  }

  // ub.poison: `sc id = nullptr;` (ptr) or `sc id = (sc)0;`
  if (op->getName().getStringRef() == "ub.poison") {
    Value res = op->getResult(0);
    Type elem = res.getType();
    if (auto rt = dyn_cast<RankedTensorType>(elem))
      elem = rt.getElementType();
    bool isPtr = isa<tt::PointerType>(elem);
    msl::Type *sc = isPtr ? astStorageType(res.getType())
                          : astScalarType(elementScalarType(res.getType()));
    std::string scName = isPtr ? mslStorageType(res.getType())
                               : mslScalarType(elementScalarType(res.getType()));
    if (scName.empty())
      return false;
    return astDeclBind(op, sc, body, [&](int) -> msl::Expr * {
      if (isPtr)
        return ctx.lit("nullptr");
      return ctx.cast(CS::CStyle, sc, ctx.lit("0"));
    });
  }

  // addptr: `device sc* id = b + o;`
  if (auto ap = dyn_cast<tt::AddPtrOp>(op)) {
    msl::Type *sc = ctx.ptr(astScalarType(elementScalarType(op->getResult(0).getType())),
                            msl::AddrSpace::Device);
    auto &base = names(ap.getPtr());
    auto &offs = names(ap.getOffset());
    return astDeclBind(op, sc, body, [&](int r) {
      return ctx.binary(B::Add, ctx.var(base[base.size() == 1 ? 0 : r]),
                        ctx.var(offs[offs.size() == 1 ? 0 : r]));
    });
  }

  // load: `sc id = init; [if (m)] id = *p;`
  if (auto ld = dyn_cast<tt::LoadOp>(op)) {
    Value res = ld.getResult();
    msl::Type *sc = astScalarType(elementScalarType(res.getType()));
    std::string scName = mslScalarType(elementScalarType(res.getType()));
    auto &ptrs = names(ld.getPtr());
    bool hasMask = ld.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(ld.getMask()) : nullptr;
    SmallVector<std::string> *other = ld.getOther() ? &names(ld.getOther()) : nullptr;
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      msl::Expr *init =
          other ? static_cast<msl::Expr *>(
                      ctx.var((*other)[other->size() == 1 ? 0 : r]))
                : static_cast<msl::Expr *>(ctx.lit("0"));
      body.push_back(ctx.declStmt(sc, id, init));
      msl::Expr *deref = astDerefPtr(ld.getPtr(), ptrs[r], scName);
      msl::Stmt *assign = ctx.assignStmt(ctx.var(id), deref);
      if (hasMask)
        body.push_back(ctx.compactIf(
            ctx.var((*mask)[mask->size() == 1 ? 0 : r]), assign));
      else
        body.push_back(assign);
      ids.push_back(id);
    }
    valMap[res] = ids;
    return true;
  }

  // store: `[if (guard)] *p = v;` (optionally wrapped by a directStore guard).
  if (auto st = dyn_cast<tt::StoreOp>(op)) {
    auto handled = directStoreHandled.find(st.getOperation());
    if (handled != directStoreHandled.end()) {
      msl::Block inner;
      astStoreBody(st, inner);
      // if (!<fullTile>) { <store body> }
      body.push_back(ctx.ifScope(
          ctx.unary(msl::UnOp::LNot, ctx.var(handled->second)),
          std::move(inner)));
      return true;
    }
    astStoreBody(st, body);
    return true;
  }

  return std::nullopt;
}

// scf.if / scf.for (+ fused-GEMM route) / scf.while.
std::optional<bool> MSLEmitter::astEmitControlFlow(Operation *op,
                                                   msl::Block &body) {
  // scf.if: predeclare result vars, then IfScope with then/else sub-blocks.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    SmallVector<SmallVector<std::string>> results;
    for (Value res : ifOp.getResults())
      results.push_back(astDeclResultVars(res, body));

    unsigned d = (unsigned)indent + 1;
    msl::Block thenB = astWalkBlock(ifOp.getThenRegion().front(), d);
    if (!results.empty())
      for (msl::Stmt *s :
           astYieldAssign(ifOp.thenBlock()->getTerminator(), results))
        thenB.push_back(s);
    if (ifOp.getElseRegion().empty()) {
      body.push_back(ctx.ifScope(ctx.var(names(ifOp.getCondition())[0]),
                                 std::move(thenB)));
    } else {
      msl::Block elseB = astWalkBlock(ifOp.getElseRegion().front(), d);
      if (!results.empty())
        for (msl::Stmt *s :
             astYieldAssign(ifOp.elseBlock()->getTerminator(), results))
          elseB.push_back(s);
      body.push_back(ctx.ifElseScope(ctx.var(names(ifOp.getCondition())[0]),
                                     std::move(thenB), std::move(elseB)));
    }
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      valMap[res] = results[i];
    return true;
  }

  // scf.for. Fused GEMM K-loops route to astEmitFusedGemm; i64-IV loops take the
  // wide-IV shape below.
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (auto m = matchGemmDotLoop(forOp))
      return astEmitFusedGemm(forOp, m->first, m->second, body);
    Type ivType = forOp.getInductionVar().getType();
    bool wideIv = ivType.isInteger(64);

    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init, res] :
         llvm::enumerate(forOp.getInitArgs(), forOp.getResults())) {
      if (isDatalessType(res.getType())) {
        valMap[forOp.getRegionIterArg(i)] = SmallVector<std::string>{};
        valMap[res] = SmallVector<std::string>{};
        carried.push_back({});
        continue;
      }
      auto &initNames = names(init);
      SmallVector<std::string> vars = astDeclResultVars(res, body);
      for (size_t r = 0; r < vars.size(); ++r)
        body.push_back(ctx.assignStmt(
            ctx.var(vars[r]),
            ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
      valMap[forOp.getRegionIterArg(i)] = vars;
      valMap[res] = vars;
      carried.push_back(vars);
    }

    // Wide-IV (i64) loops carry `tc` as the header counter and the real IV decl
    // as the first body stmt (the AGX i65 Gauss-sum dodge). Mint iv before tc.
    std::string iv = fresh();
    std::string tc = wideIv ? fresh() : "";
    bindScalar(forOp.getInductionVar(), iv);
    std::string ivTy = mslScalarType(ivType);
    if (ivTy.empty())
      ivTy = "int";

    unsigned d = (unsigned)indent + 1;
    msl::Block loopBody = astWalkBlock(forOp.getRegion().front(), d);
    for (msl::Stmt *s :
         astYieldAssign(forOp.getBody()->getTerminator(), carried))
      loopBody.push_back(s);
    body.push_back(astForNode(forOp, std::move(loopBody), iv, tc, ivTy, wideIv));
    return true;
  }

  // scf.while: `while (true) { <before> if (!(c)) { <fwd> break; } <after> <yield> }`
  if (auto wh = dyn_cast<scf::WhileOp>(op)) {
    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init] : llvm::enumerate(wh.getInits())) {
      auto &initNames = names(init);
      SmallVector<std::string> vars = astDeclResultVars(init, body);
      for (size_t r = 0; r < vars.size(); ++r)
        body.push_back(ctx.assignStmt(
            ctx.var(vars[r]),
            ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
      valMap[wh.getBeforeArguments()[i]] = vars;
      carried.push_back(vars);
    }
    SmallVector<SmallVector<std::string>> results;
    for (Value res : wh.getResults())
      results.push_back(astDeclResultVars(res, body));

    unsigned d = (unsigned)indent + 1;
    msl::Block loopBody = astWalkBlock(wh.getBefore().front(), d);
    auto cond = cast<scf::ConditionOp>(wh.getBefore().front().getTerminator());
    // if (!(c)) { <forward results> break; }
    msl::Block brk;
    for (auto [i, fwd] : llvm::enumerate(cond.getArgs())) {
      auto &src = names(fwd);
      for (size_t r = 0; r < results[i].size(); ++r)
        brk.push_back(ctx.assignStmt(ctx.var(results[i][r]),
                                     ctx.var(src[src.size() == 1 ? 0 : r])));
    }
    brk.push_back(ctx.breakStmt());
    msl::Expr *guard = ctx.unary(
        msl::UnOp::LNot, ctx.paren(ctx.var(names(cond.getCondition())[0])));
    loopBody.push_back(ctx.ifScope(guard, std::move(brk)));

    for (auto [i, fwd] : llvm::enumerate(cond.getArgs()))
      valMap[wh.getAfterArguments()[i]] = names(fwd);

    for (msl::Stmt *s : astWalkBlock(wh.getAfter().front(), d))
      loopBody.push_back(s);
    for (msl::Stmt *s :
         astYieldAssign(wh.getAfter().front().getTerminator(), carried))
      loopBody.push_back(s);

    body.push_back(ctx.whileScope(nullptr, std::move(loopBody)));
    for (auto [i, res] : llvm::enumerate(wh.getResults()))
      valMap[res] = results[i];
    return true;
  }

  return std::nullopt;
}

} // namespace mlir::triton::applegpu
