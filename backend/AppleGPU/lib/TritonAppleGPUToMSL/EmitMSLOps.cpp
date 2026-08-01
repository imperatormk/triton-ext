// EmitMSLOps.cpp - op-dispatch spine + its AST lowering helpers.
//
// The emitOp waterfall and the per-family helpers it calls
// (elemwiseDecls/combineN/scanWarpCarry/emitMapCFG/emitFusedGemm/
// declResultVars/declBind/derefPtr/bandRoundTrip/storeBody) plus
// the local cmpBinOp string->BinOp map.

#include "MSLConstants.h"
#include "MSLEmitter.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace mlir::triton::applegpu {

// Below this many registers the peeled all-true arm costs more in code size
// and register pressure than the serialization it removes.
static constexpr int kMaskFastPathMinRegs = 4;

using B = msl::BinOp;
using CS = msl::Cast::Style;

//===----------------------------------------------------------------------===//
// Op dispatch spine
//===----------------------------------------------------------------------===//

// Append the nodes for a single elementwise/expr op's per-register DeclStmts,
// using the expr-builder `mk(r)` per register. emitOp mints the real names.
bool MSLEmitter::elemwiseDecls(Operation *op, msl::Type *declTy, int &id,
                               msl::Block &body,
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
bool MSLEmitter::combineN(Region &region, ArrayRef<std::string> aVals,
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
    if (emitOp(&o, body))
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
    results.push_back(scalarName(r).str());
  return true;
}

// Cross-warp inclusive-carry for one register group. Compound guard/index exprs
// are ctx.raw leaves.
bool MSLEmitter::scanWarpCarry(
    Region &region, int nOp, ArrayRef<std::string> scTys,
    ArrayRef<int64_t> byteWidths, ArrayRef<std::pair<int, int32_t>> warpBits,
    ArrayRef<int> regs, SmallVector<SmallVector<std::string>> &accs,
    ArrayRef<std::string> laneScan, StringRef axisTopLane,
    unsigned axisWarpMask, int numWarps, bool rev,
    SmallVectorImpl<std::string> &runTotalOut, msl::Block &body) {
  SmallVector<msl::Type *> scT(nOp);
  for (int k = 0; k < nOp; ++k)
    scT[k] = ctx.named(scTys[k]);

  if (warpBits.empty()) {
    for (int k = 0; k < nOp; ++k) {
      std::string s = shuffle(msl::builtin::simd::Shuffle, scTys[k],
                              laneScan[k], axisTopLane, body);
      body.push_back(ctx.assignStmt(ctx.var(runTotalOut[k]), ctx.var(s)));
    }
    return true;
  }

  SmallVector<std::string> scratch(nOp);
  int64_t byteOff = 0;
  for (int k = 0; k < nOp; ++k) {
    scratch[k] = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scT[k], msl::AddrSpace::Threadgroup),
                                scratch[k], poolRegion(byteOff, scTys[k])));
    byteOff += (int64_t)numWarps * 32 * byteWidths[k];
  }
  body.push_back(ctx.hardBarrier(false));
  msl::Expr *topGuard =
      axisTopLane == StringRef(laneId)
          ? static_cast<msl::Expr *>(ctx.lit("true"))
          : ctx.binary(B::Eq, ctx.var(laneId), ctx.var(axisTopLane));
  for (int k = 0; k < nOp; ++k) {
    // scratch[k][warp * 32 + lane] = laneScan[k];
    msl::Expr *idx =
        ctx.binary(B::Add, ctx.binary(B::Mul, ctx.var(warpId), ctx.lit("32")),
                   ctx.var(laneId));
    msl::Stmt *asn = ctx.assignStmt(ctx.subscript(ctx.var(scratch[k]), idx),
                                    ctx.var(laneScan[k]));
    body.push_back(ctx.compactIf(topGuard, asn));
  }
  body.push_back(ctx.hardBarrier(false));

  // ((warpId & ~axisWarpMask) * 32 + axisTopLane)
  msl::Expr *base = ctx.paren(
      ctx.add(ctx.mul(ctx.paren(ctx.binary(B::And, ctx.var(warpId),
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
          ctx.paren(ctx.binary(B::And,
                               ctx.paren(ctx.binary(B::Shr, ctx.var(warpId),
                                                    ctx.i32lit(maskBits[r]))),
                               ctx.i32lit(1))),
          ctx.i32lit(r)))));
    msl::Expr *warpPos = posTerms[0];
    for (size_t i = 1; i < posTerms.size(); ++i)
      warpPos = ctx.paren(ctx.binary(B::Or, warpPos, posTerms[i]));
    body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), myPart, warpPos));
  }

  SmallVector<int> order;
  for (int p = 0; p < nParts; ++p)
    order.push_back(rev ? nParts - 1 - p : p);

  auto slot = [&](int k, int part) -> msl::Expr * {
    return ctx.subscript(
        ctx.var(scratch[k]),
        ctx.binary(B::Add, base, ctx.lit(std::to_string(partWarp(part) * 32))));
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
    if (!combineN(region, grand, pv, body, out))
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
  body.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::I1), init, ctx.lit("false")));
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
    // if (init) { carry = combine(carry, pv); } else { carry = pv; init = true;
    // }
    msl::Block thenB, elseB;
    {
      SmallVector<std::string> out;
      if (!combineN(region, carry, pv, thenB, out))
        return false;
      for (int k = 0; k < nOp; ++k)
        thenB.push_back(ctx.assignStmt(ctx.var(carry[k]), ctx.var(out[k])));
    }
    for (int k = 0; k < nOp; ++k)
      elseB.push_back(ctx.assignStmt(ctx.var(carry[k]), ctx.var(pv[k])));
    elseB.push_back(ctx.assignStmt(ctx.var(init), ctx.lit("true")));
    ifBody.push_back(
        ctx.ifElseScope(ctx.var(init), std::move(thenB), std::move(elseB)));
    body.push_back(ctx.ifScope(cond, std::move(ifBody)));
  }
  for (int r : regs) {
    SmallVector<std::string> ar(nOp);
    for (int k = 0; k < nOp; ++k)
      ar[k] = accs[k][r];
    SmallVector<std::string> out;
    if (!combineN(region, carry, ar, body, out))
      return false;
    for (int k = 0; k < nOp; ++k)
      // accs[k][r] = (init ? out[k] : accs[k][r]);
      body.push_back(
          ctx.assignStmt(ctx.var(accs[k][r]),
                         ctx.paren(ctx.ternary(ctx.var(init), ctx.var(out[k]),
                                               ctx.var(accs[k][r])))));
  }
  body.push_back(ctx.hardBarrier(false));
  return true;
}

// Multi-block map_elementwise region -> state machine. Like emitBlockCFG but
// the map_elementwise.return terminator spills operands into caller `capture`
// slots then breaks. Appends the predeclarations + state machine to `body`.
void MSLEmitter::emitMapCFG(Region &region, ArrayRef<std::string> capture,
                            msl::Block &body) {
  blockLabel.clear();
  int idx = 0;
  for (Block &blk : region)
    blockLabel[&blk] = std::to_string(idx++);
  for (Block &blk : llvm::drop_begin(region))
    for (BlockArgument arg : blk.getArguments()) {
      if (isDatalessType(arg.getType())) {
        bindDataless(arg);
        continue;
      }
      bindRegs(arg, declResultVars(arg, body));
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
          hoist[res] = declResultVars(res, body);
      }
  std::string state = fresh();
  cfgState = state;
  llvm::SmallVector<std::pair<std::string, msl::Block>> cases;
  for (Block &blk : region) {
    msl::Block caseBody = walkBlock2(blk, hoist);
    Operation *term = blk.getTerminator();
    if (term->getName().getStringRef() == "tt.map_elementwise.return") {
      for (auto [i, operand] : llvm::enumerate(term->getOperands()))
        caseBody.push_back(mapReturnSpill(capture[i], scalarName(operand)));
      caseBody.push_back(ctx.breakStmt());
    } else {
      for (msl::Stmt *s : terminatorEdge(term, state))
        caseBody.push_back(s);
    }
    cases.push_back({blockLabel[&blk], std::move(caseBody)});
  }
  cfgState.clear();
  body.push_back(mapCFGStateMachine(state, cases));
}

// Fused GEMM K-loop: carry iter-args, run the dot Decl phase, the K-loop with
// the MMA-phase dot in its body, the non-acc carry, direct-store setup, then
// the Readback-phase dot.
bool MSLEmitter::emitFusedGemm(scf::ForOp op, tt::DotOp dot, unsigned iterIdx,
                               msl::Block &body) {
  SmallVector<SmallVector<std::string>> carried;
  SmallVector<std::string> initBase;
  for (auto [i, init, res] :
       llvm::enumerate(op.getInitArgs(), op.getResults())) {
    if (isDatalessType(res.getType())) {
      bindDataless(op.getRegionIterArg(i));
      bindDataless(res);
      carried.push_back({});
      continue;
    }
    auto &initNames = names(init);
    if (i == iterIdx) {
      SmallVector<std::string> ids = declResultVars(res, body);
      initBase.assign(initNames.begin(), initNames.end());
      fusedDot.ids = ids;
      bindRegs(op.getRegionIterArg(i), ids);
      bindRegs(res, ids);
      carried.push_back({});
      continue;
    }
    SmallVector<std::string> vars = declResultVars(res, body);
    for (size_t r = 0; r < vars.size(); ++r)
      body.push_back(ctx.assignStmt(
          ctx.var(vars[r]), ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
    bindRegs(op.getRegionIterArg(i), vars);
    bindRegs(res, vars);
    carried.push_back(vars);
  }

  fusedDot.baseNames = initBase;
  fusedDot.phase = FusedDotPhase::Decl;
  if (!emitDot(dot, body))
    return false;

  std::string iv = fresh();
  bindScalar(op.getInductionVar(), iv);
  std::string ivTy = mslScalarType(op.getInductionVar().getType());
  if (ivTy.empty())
    ivTy = "int";

  // Loop body: MMA-phase dot (walked) + non-acc carry.
  fusedDot.phase = FusedDotPhase::MMA;
  msl::Block loopBody = walkBlock(op.getRegion().front(), (unsigned)indent + 1);
  if (emitFailed)
    return false;
  fusedDot.phase = FusedDotPhase::None;
  auto *term = op.getBody()->getTerminator();
  for (auto [i, operand] : llvm::enumerate(term->getOperands())) {
    if (i == iterIdx || carried[i].empty())
      continue;
    auto &src = names(operand);
    for (size_t r = 0; r < carried[i].size(); ++r)
      loopBody.push_back(ctx.assignStmt(ctx.var(carried[i][r]),
                                        ctx.var(src[src.size() == 1 ? 0 : r])));
  }
  body.push_back(forScope(op, std::move(loopBody), iv, ivTy));

  if (auto ds = matchDirectStore(op.getResult(iterIdx))) {
    int64_t M = cast<RankedTensorType>(dot.getResult().getType()).getShape()[0];
    int64_t N = cast<RankedTensorType>(dot.getResult().getType()).getShape()[1];
    std::string ft = fresh();
    ds->fullTileVar = ft;
    msl::Expr *cond;
    if (ds->boundM) {
      auto uniform = [&](const UniformInt &u) -> msl::Expr * {
        return u.lit ? static_cast<msl::Expr *>(ctx.i32lit(*u.lit))
                     : ctx.var(scalarName(u.val));
      };
      // (rowBase + M <= boundM && colBase + N <= boundN)
      cond = ctx.paren(ctx.binary(
          B::LAnd,
          ctx.binary(B::Le,
                     ctx.binary(B::Add, ctx.var(scalarName(ds->rowBase)),
                                ctx.lit(std::to_string(M))),
                     uniform(ds->boundM)),
          ctx.binary(B::Le,
                     ctx.binary(B::Add, ctx.var(scalarName(ds->colBase)),
                                ctx.lit(std::to_string(N))),
                     uniform(ds->boundN))));
    } else {
      cond = ctx.lit("true");
      ds->alwaysFullTile = true;
    }
    body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), ft, cond));
    fusedDot.direct = ds;
    directStoreHandled[ds->store.getOperation()] = ft;
    // On the direct path the fragments go straight to device memory, so the
    // registers feeding the fallback store are never produced. Everything
    // between the accumulator and the store has to sit under the same guard or
    // it reads (and stores) undefined values.
    if (ds->narrowOp)
      directStoreHandled[ds->narrowOp] = ft;
    if (ds->cvt)
      directStoreHandled[ds->cvt] = ft;
  }

  fusedDot.phase = FusedDotPhase::Readback;
  if (!emitDot(dot, body))
    return false;
  fusedDot = FusedDotCtx{};
  return true;
}

// Predeclare a value's per-register result variables (`sc id;`) with no init,
// mirroring declResultVars; returns the minted names (caller binds valMap).
SmallVector<std::string> MSLEmitter::declResultVars(Value v, msl::Block &body) {
  Type elem = v.getType();
  if (auto rt = dyn_cast<RankedTensorType>(elem))
    elem = rt.getElementType();
  msl::Type *sc = isa<tt::PointerType>(elem)
                      ? storageType(v.getType())
                      : scalarType(elementScalarType(v.getType()));
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
bool MSLEmitter::declBind(Operation *op, msl::Type *declTy, msl::Block &body,
                          llvm::function_ref<msl::Expr *(int)> mk) {
  int rc = regCount(op->getResult(0));
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    body.push_back(ctx.declStmt(declTy, id, mk(r)));
    ids.push_back(id);
  }
  bindRegs(op->getResult(0), ids);
  return true;
}

bool MSLEmitter::needsCoherentDeref(Value ptr) const {
  if (coherentScalarPtrs.empty() || !ptr)
    return false;
  if (isa<RankedTensorType>(ptr.getType()))
    return false;
  Value base = traceToKernelArg(ptr);
  return base && coherentScalarPtrs.contains(base);
}

// `*p`, or the coherent-cast deref `*(device coherent(device) sc*)p` when a
// scalar spinlock or a loop-carried scalar RMW forces a coherent access.
msl::Expr *MSLEmitter::derefPtr(Value ptr, StringRef name, StringRef scName) {
  if (scalarSpinlock || needsCoherentDeref(ptr)) {
    msl::Type *cp = ctx.ptr(ctx.named(scName), msl::AddrSpace::Device,
                            /*coherent=*/true);
    return ctx.deref(ctx.cast(CS::CStyle, cp, ctx.var(name)));
  }
  return ctx.deref(ctx.var(name));
}

tt::ModuleAxisInfoAnalysis &MSLEmitter::getAxisInfo() {
  if (!axisInfo)
    axisInfo = std::make_unique<tt::ModuleAxisInfoAnalysis>(mod);
  return *axisInfo;
}

// A run of `w` consecutive registers is one vecW access iff (a) those registers
// are adjacent along a single tensor dim, (b) the pointer is contiguous and
// w-aligned along that dim, and (c) the run does not straddle a lane/warp
// boundary -- all three are exactly what the register bases below encode.
int MSLEmitter::accessVectorWidth(Type valueTy, Value ptr) {
  auto rt = dyn_cast<RankedTensorType>(valueTy);
  if (!rt)
    return 1;
  Type elem = elementScalarType(valueTy);
  unsigned bits = elem.getIntOrFloatBitWidth();
  if (bits != 16 && bits != 32)
    return 1;

  MLIRContext *c = rt.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto kReg = StringAttr::get(c, "register");
  auto outs = llvm::to_vector(ll.getOutDimNames());
  int nRegLog2 = ll.getInDimSizeLog2(kReg);
  if (nRegLog2 == 0)
    return 1;

  // Register bit b must move by exactly 2^b along one fixed dim and nowhere
  // else; that makes registers 0..2^n-1 the elements [0, 2^n) of that dim.
  int dim = -1;
  int runLog2 = 0;
  for (int b = 0; b < nRegLog2; ++b) {
    int hit = -1;
    bool bad = false;
    for (auto [d, name] : llvm::enumerate(outs)) {
      int32_t basis = ll.getBasis(kReg, b, name);
      if (basis == 0)
        continue;
      if (hit >= 0 || basis != (1 << b)) {
        bad = true;
        break;
      }
      hit = d;
    }
    if (bad || hit < 0 || (dim >= 0 && hit != dim))
      break;
    dim = hit;
    ++runLog2;
  }
  if (dim < 0 || runLog2 == 0)
    return 1;

  // Metal's portable device vector widths are 2/3/4; the 8-wide extended
  // vectors (e.g. `bfloat8`) collide with the metal:: names and are not worth
  // a second load width here.
  int width = std::min(1 << runLog2, 4);
  if (width < 2)
    return 1;

  tt::AxisInfo *ai = getAxisInfo().getAxisInfo(ptr);
  if (!ai)
    return 1;
  int64_t contig = ai->getContiguity(dim);
  int64_t align = ai->getDivisibility(dim);
  while (width > 1 && (contig < width || align < width))
    width >>= 1;
  return width;
}

int MSLEmitter::loadVectorWidth(tt::LoadOp ld) {
  static const bool disabled = getenv("MSL_DISABLE_LOAD_VECTORIZE") != nullptr;
  if (disabled)
    return 1;
  return accessVectorWidth(ld.getResult().getType(), ld.getPtr());
}

int MSLEmitter::storeVectorWidth(tt::StoreOp st) {
  static const bool disabled = getenv("MSL_DISABLE_STORE_VECTORIZE") != nullptr;
  if (disabled)
    return 1;
  return accessVectorWidth(st.getValue().getType(), st.getPtr());
}

// Shared banded threadgroup round-trip (trans/reshape): for each band, barrier
// + scatter each src register to buf[srcOff] + barrier + gather each res
// register from buf[resOff]. `band == total` uses the direct `buf[off]=v;`
// form; a smaller band wraps each in `{ int __f=off; if (__f>=lo && __f<hi)
// buf[__f-lo]=v; }`.
void MSLEmitter::bandRoundTrip(msl::Block &body, StringRef buf, int64_t total,
                               int64_t band, int srcRc, int resRc,
                               ArrayRef<std::string> outs,
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
    msl::Expr *idx =
        ctx.binary(B::Sub, ctx.var("__f"), ctx.lit(std::to_string(lo)));
    msl::Expr *slot = ctx.subscript(ctx.var(buf), idx);
    msl::Stmt *asn =
        toBuf ? ctx.assignStmt(slot, reg) : ctx.assignStmt(reg, slot);
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

void MSLEmitter::emitTileRoundTrip(
    Value res, Value src, RankedTensorType srcTy, RankedTensorType resTy,
    msl::Block &body, llvm::function_ref<msl::Expr *(int)> srcOff) {
  msl::Type *scTy = scalarType(resTy.getElementType());
  std::string sc = mslScalarType(resTy.getElementType());
  auto &srcNames = names(src);
  int srcRc = regCount(src);
  int resRc = regCount(res);
  int64_t elemBytes = byteWidth(resTy.getElementType());
  int64_t total = tileSize(resTy);

  std::string buf = fresh();
  body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                              poolRegion(0, sc)));
  int64_t band = reshapeBand(total, elemBytes);
  SmallVector<std::string> ids = declResultVars(res, body);
  bandRoundTrip(
      body, buf, total, band, srcRc, resRc, ids, srcOff,
      [&](int r) {
        return static_cast<msl::Expr *>(
            ctx.var(srcNames[srcNames.size() == 1 ? 0 : r]));
      },
      [&](int r) { return layout.flatTileOffset(resTy, r); });
  bindRegs(res, ids);
}

// Per-register store: `[if (guard)] *p = v;` with the thread predicate + mask
// guard.
void MSLEmitter::storeBody(tt::StoreOp op, msl::Block &body) {
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
    threadPred =
        ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0"));
  } else {
    if (laneFree)
      threadPred = ctx.paren(
          ctx.binary(B::Eq,
                     ctx.paren(ctx.binary(B::And, ctx.var(laneId),
                                          ctx.lit(std::to_string(laneFree)))),
                     ctx.lit("0")));
    if (warpFree) {
      msl::Expr *wp = ctx.paren(
          ctx.binary(B::Eq,
                     ctx.paren(ctx.binary(B::And, ctx.var(warpId),
                                          ctx.lit(std::to_string(warpFree)))),
                     ctx.lit("0")));
      threadPred = threadPred ? ctx.binary(B::LAnd, threadPred, wp) : wp;
    }
  }

  Type elemTy = elementScalarType(op.getValue().getType());
  std::string scName = mslScalarType(elemTy);
  int vw = storeVectorWidth(op);
  if (vw < 2 || rc % vw != 0 || vals.size() == 1)
    vw = 1;

  auto maskOf = [&](int r) {
    return ctx.var((*mask)[mask->size() == 1 ? 0 : r]);
  };
  auto scalarStore = [&](int r) -> msl::Stmt * {
    msl::Stmt *a = ctx.assignStmt(derefPtr(op.getPtr(), ptrs[r], scName),
                                  ctx.var(vals[vals.size() == 1 ? 0 : r]));
    msl::Expr *guard = threadPred;
    if (hasMask)
      guard =
          guard
              ? static_cast<msl::Expr *>(ctx.binary(B::LAnd, guard, maskOf(r)))
              : static_cast<msl::Expr *>(maskOf(r));
    return guard ? static_cast<msl::Stmt *>(ctx.compactIf(guard, a)) : a;
  };

  for (int base = 0; base < rc; base += vw) {
    if (vw == 1) {
      body.push_back(scalarStore(base));
      continue;
    }
    auto *scTy = cast<msl::ScalarType>(scalarType(elemTy));
    msl::Type *vecTy = ctx.vector(scTy->s, vw);
    msl::Type *vecPtr = ctx.ptr(vecTy, msl::AddrSpace::Device);
    std::string vid = fresh();
    msl::Block pack;
    pack.push_back(ctx.declStmt(vecTy, vid));
    for (int i = 0; i < vw; ++i)
      pack.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(vid), ctx.lit(std::to_string(i))),
          ctx.var(vals[base + i])));
    pack.push_back(ctx.assignStmt(
        ctx.deref(ctx.cast(CS::CStyle, vecPtr, ctx.var(ptrs[base]))),
        ctx.var(vid)));

    // The run's lanes are contiguous but their predicates are not provably
    // equal, so the wide store is only legal when all of them hold; otherwise
    // fall back to per-lane scalar stores so masked-off lanes stay unwritten.
    if (!hasMask) {
      if (threadPred) {
        body.push_back(ctx.ifScope(threadPred, std::move(pack)));
      } else {
        for (msl::Stmt *s : pack)
          body.push_back(s);
      }
      continue;
    }
    SmallVector<msl::Expr *> ms;
    for (int i = 0; i < vw; ++i)
      ms.push_back(maskOf(base + i));
    msl::Expr *all = ctx.chain(B::And, ms);
    if (threadPred)
      all = ctx.binary(B::LAnd, threadPred, all);
    msl::Block cold;
    for (int i = 0; i < vw; ++i)
      cold.push_back(scalarStore(base + i));
    body.push_back(ctx.ifElseScope(all, std::move(pack), std::move(cold)));
  }
}

static const char *axisComp(tt::ProgramIDDim axis) {
  return axis == tt::ProgramIDDim::X   ? "x"
         : axis == tt::ProgramIDDim::Y ? "y"
                                       : "z";
}

// `int id = (int)(builtinVar.comp);` and bind the result.
void MSLEmitter::programDim(Operation *op, StringRef builtinVar,
                            tt::ProgramIDDim axis, msl::Block &body) {
  msl::Expr *e =
      ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::I32),
               ctx.paren(ctx.member(ctx.var(builtinVar), axisComp(axis))));
  std::string id = fresh();
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), id, e));
  bindScalar(op->getResult(0), id);
}

// Route `op` to its builder(s), appending nodes to `body`. Returns true when
// handled (including alias/dataless ops that append nothing); false for an
// unsupported op, which the caller turns into a hard error.
bool MSLEmitter::emitOp(Operation *op, msl::Block &body) {
  // An op on the direct-store fallback chain only has defined inputs when the
  // direct path was not taken, so it runs under the same guard. Its results are
  // declared outside the guard and assigned inside, keeping later ops (and the
  // guarded store) able to name them.
  if (!isa<tt::StoreOp>(op)) {
    auto guarded = directStoreHandled.find(op);
    if (guarded != directStoreHandled.end()) {
      std::string ftVar = guarded->second;
      directStoreHandled.erase(guarded);
      msl::Block inner;
      if (!emitOp(op, inner))
        return false;
      SmallVector<SmallVector<std::string>> outer;
      for (Value r : op->getResults()) {
        auto it = valMap.find(r);
        if (it == valMap.end()) {
          outer.push_back({});
          continue;
        }
        SmallVector<std::string> names;
        for (const std::string &n : it->second) {
          std::string decl = fresh();
          body.push_back(ctx.declStmt(
              scalarType(elementScalarType(r.getType())), decl, nullptr));
          names.push_back(decl);
        }
        outer.push_back(names);
      }
      for (auto [ri, r] : llvm::enumerate(op->getResults())) {
        auto it = valMap.find(r);
        if (it == valMap.end())
          continue;
        for (auto [i, n] : llvm::enumerate(it->second))
          inner.push_back(ctx.assignStmt(ctx.var(outer[ri][i]), ctx.var(n)));
      }
      body.push_back(ctx.ifScope(ctx.unary(msl::UnOp::LNot, ctx.var(ftVar)),
                                 std::move(inner)));
      for (auto [ri, r] : llvm::enumerate(op->getResults()))
        if (!outer[ri].empty())
          valMap[r] = outer[ri];
      return true;
    }
  }

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
    // async_commit_group only closes a batch; it is async_wait that makes the
    // staged tiles visible. A barrier for the commit as well puts a full
    // threadgroup sync between the copies of a single trip, and a wait whose
    // tail is already fenced (nothing but address arithmetic since the last
    // barrier) adds a second sync at the same point.
    if (isa<ttg::AsyncWaitOp>(op)) {
      // Wait on the copies this wait actually closes. The loop body is walked
      // more than once (Decl then MMA phase), so a mutable "pending" list is
      // drained by the first walk and empty for the second -- the tokens have
      // to come from the IR, through the per-site map.
      if (auto w = dyn_cast<ttg::AsyncWaitOp>(op)) {
        // A token reaches the wait as a loop iter-arg, so resolve block
        // arguments back to the commit the loop yields for that slot.
        llvm::SmallVector<std::string> waits;
        std::function<void(Value)> collect = [&](Value tok) {
          if (auto arg = dyn_cast<BlockArgument>(tok)) {
            auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
            if (!forOp || arg.getArgNumber() == 0)
              return;
            auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            collect(yield.getOperand(arg.getArgNumber() - 1));
            return;
          }
          auto commit = dyn_cast_or_null<ttg::AsyncCommitGroupOp>(
              tok.getDefiningOp());
          if (!commit)
            return;
          for (Value inner : commit.getInputTokens())
            if (Operation *cp = inner.getDefiningOp()) {
              auto it = dmaHandleFor.find(cp);
              if (it != dmaHandleFor.end() &&
                  !llvm::is_contained(waits, it->second))
                waits.push_back(it->second);
            }
        };
        for (Value tok : w.getAsyncToken())
          collect(tok);
        for (const std::string &h : waits)
          body.push_back(dmaWait(h));
      }
      pendingDmaHandles.clear();
      if (!barrierCoversTail(body))
        body.push_back(ctx.barrier(/*device=*/false));
      // The batch ends here; the next copy opens a new one and fences again.
      asyncCopyFenced = false;
    }
    for (Value r : op->getResults())
      bindDataless(r);
    return true;
  }

  // Structural no-ops that neither emit text nor bind a named value.
  if (isa<ttg::LocalDeallocOp, scf::YieldOp, scf::ConditionOp>(op))
    return true;
  if (op->getName().getStringRef() == "llvm.intr.assume")
    return true;
  if (isa<tt::AssertOp, tt::PrintOp>(op)) {
    for (Value r : op->getResults())
      bindDataless(r);
    return true;
  }

  // Op-family sub-dispatchers, tried in the original arm order. Each returns
  // nullopt for an op it doesn't own (fall through to the next family), or the
  // handled/failed bool. Unmatched by all -> false (a hard error upstream).
  using Family = std::optional<bool> (MSLEmitter::*)(Operation *, msl::Block &);
  static constexpr Family families[] = {
      &MSLEmitter::emitArithBinop, &MSLEmitter::emitConstGrid,
      &MSLEmitter::emitArithMisc,  &MSLEmitter::emitMath,
      &MSLEmitter::emitReshape,    &MSLEmitter::emitMemDesc,
      &MSLEmitter::emitDotMap,     &MSLEmitter::emitAtomicRMW,
      &MSLEmitter::emitAtomicPoll, &MSLEmitter::emitAtomicCAS,
      &MSLEmitter::emitScan,       &MSLEmitter::emitReduce,
      &MSLEmitter::emitTensorMove, &MSLEmitter::emitCallReturn,
      &MSLEmitter::emitControlFlow};
  for (Family f : families)
    if (std::optional<bool> r = (this->*f)(op, body))
      return *r;

  // Unsupported op: walkBlock turns this false into a hard error.
  return false;
}

// float / int / bitwise / shift binops.
std::optional<bool> MSLEmitter::emitArithBinop(Operation *op,
                                               msl::Block &body) {
  Type resElem = op->getNumResults()
                     ? elementScalarType(op->getResult(0).getType())
                     : Type();

  // Float binaries: `sc id = (a o b);`
  if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp,
          tt::PreciseDivFOp>(op))
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return elementwiseExpr(arithBinOp(op), nullptr, reg(op->getOperand(0), r),
                             reg(op->getOperand(1), r));
    });

  // Integer add/sub/mul/div/rem (intBinaryExpr handles the i1 and unsigned
  // paths; the decl type must match the unsigned promotion).
  if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::DivSIOp,
          arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
    msl::Type *declTy = scalarType(resElem);
    if (auto it = dyn_cast<IntegerType>(resElem); it && it.getWidth() == 1)
      declTy = ctx.scalar(msl::Scalar::I1);
    else if (isa<arith::DivUIOp, arith::RemUIOp>(op))
      declTy = unsignedType(resElem);
    return declBind(op, declTy, body, [&](int r) {
      return intBinaryExpr(op, reg(op->getOperand(0), r),
                           reg(op->getOperand(1), r));
    });
  }

  // Bitwise/logical and/or/xor.
  if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return elementwiseExpr(arithBinOp(op), nullptr, reg(op->getOperand(0), r),
                             reg(op->getOperand(1), r));
    });
  }

  // Shifts.
  if (isa<arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(op))
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return shiftExpr(op, reg(op->getOperand(0), r),
                       reg(op->getOperand(1), r));
    });

  return std::nullopt;
}

// program_id / num_programs / arith.constant / make_range.
std::optional<bool> MSLEmitter::emitConstGrid(Operation *op, msl::Block &body) {
  // Program-id / num-programs: `int id = (int)(builtin.comp);`
  if (auto p = dyn_cast<tt::GetProgramIdOp>(op)) {
    programDim(op, tgposId, p.getAxis(), body);
    return true;
  }
  if (auto n = dyn_cast<tt::GetNumProgramsOp>(op)) {
    programDim(op, numTgId, n.getAxis(), body);
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
      msl::Type *sc = scalarType(elemTy);
      bool isFloat = isa<FloatType>(elemTy);
      if (dense.isSplat()) {
        std::string lit =
            isFloat
                ? floatLit(dense.getSplatValue<APFloat>(), elemTy)
                : std::to_string(dense.getSplatValue<APInt>().getSExtValue());
        return declBind(op, sc, body, [&](int) { return ctx.lit(lit); });
      }
      // Dense table: `sc tbl[N] = {..}; sc id = tbl[flatTileOffset];`
      SmallVector<msl::Expr *> init;
      if (isFloat)
        for (const APFloat &v : dense.getValues<APFloat>())
          init.push_back(floatLitExpr(v, elemTy));
      else
        for (const APInt &v : dense.getValues<APInt>())
          init.push_back(ctx.lit(std::to_string(v.getSExtValue())));
      std::string tbl = fresh();
      body.push_back(ctx.arrayDeclStmt(sc, tbl, dense.getNumElements(), init));
      return declBind(op, sc, body, [&](int r) {
        return ctx.subscript(ctx.var(tbl), layout.flatTileOffset(rt, r));
      });
    }
    msl::Type *sc = scalarType(res.getType());
    msl::Expr *lit;
    if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
      lit = floatLitExpr(fa.getValue(), res.getType());
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
    return declBind(op, ctx.scalar(msl::Scalar::I32), body, [&](int r) {
      return makeRangeElem(start, layout.layoutOffsetExpr(rt, r));
    });
  }

  return std::nullopt;
}

// cmp / select / clamp / casts / negf / min-max family / precise_sqrt.
std::optional<bool> MSLEmitter::emitArithMisc(Operation *op, msl::Block &body) {
  Type resElem = op->getNumResults()
                     ? elementScalarType(op->getResult(0).getType())
                     : Type();
  // Integer compare: `bool id = (casta o castb);`
  if (auto ci = dyn_cast<arith::CmpIOp>(op)) {
    msl::BinOp bo;
    bool uns = false;
    switch (ci.getPredicate()) {
    case arith::CmpIPredicate::ult:
      uns = true;
      [[fallthrough]];
    case arith::CmpIPredicate::slt:
      bo = B::Lt;
      break;
    case arith::CmpIPredicate::ule:
      uns = true;
      [[fallthrough]];
    case arith::CmpIPredicate::sle:
      bo = B::Le;
      break;
    case arith::CmpIPredicate::ugt:
      uns = true;
      [[fallthrough]];
    case arith::CmpIPredicate::sgt:
      bo = B::Gt;
      break;
    case arith::CmpIPredicate::uge:
      uns = true;
      [[fallthrough]];
    case arith::CmpIPredicate::sge:
      bo = B::Ge;
      break;
    case arith::CmpIPredicate::eq:
      bo = B::Eq;
      break;
    case arith::CmpIPredicate::ne:
      bo = B::Ne;
      break;
    }
    msl::Type *opCast =
        uns ? unsignedType(elementScalarType(ci.getLhs().getType())) : nullptr;
    return declBind(op, ctx.scalar(msl::Scalar::I1), body, [&](int r) {
      return elementwiseExpr(bo, opCast, reg(op->getOperand(0), r),
                             reg(op->getOperand(1), r));
    });
  }
  if (auto cf = dyn_cast<arith::CmpFOp>(op)) {
    msl::BinOp bo;
    // MSL's relational operators are the ORDERED comparisons (false if either
    // operand is NaN) except `!=`, which is unordered. `unordered` marks the
    // predicates whose sense therefore differs from the bare operator and needs
    // an explicit isnan() guard: an unordered form ORs the NaN cases in, the
    // lone ordered form (ONE) ANDs them out.
    bool unordered = false;
    switch (cf.getPredicate()) {
    case arith::CmpFPredicate::OLT:
      bo = B::Lt;
      break;
    case arith::CmpFPredicate::OLE:
      bo = B::Le;
      break;
    case arith::CmpFPredicate::OGT:
      bo = B::Gt;
      break;
    case arith::CmpFPredicate::OGE:
      bo = B::Ge;
      break;
    case arith::CmpFPredicate::OEQ:
      bo = B::Eq;
      break;
    case arith::CmpFPredicate::UNE:
      bo = B::Ne;
      break;
    case arith::CmpFPredicate::ULT:
      bo = B::Lt;
      unordered = true;
      break;
    case arith::CmpFPredicate::ULE:
      bo = B::Le;
      unordered = true;
      break;
    case arith::CmpFPredicate::UGT:
      bo = B::Gt;
      unordered = true;
      break;
    case arith::CmpFPredicate::UGE:
      bo = B::Ge;
      unordered = true;
      break;
    case arith::CmpFPredicate::UEQ:
      bo = B::Eq;
      unordered = true;
      break;
    case arith::CmpFPredicate::ONE:
      bo = B::Ne;
      unordered = true;
      break;
    default:
      return false; // unsupported predicate: caller emits the error
    }
    bool ordered = cf.getPredicate() == arith::CmpFPredicate::ONE;
    return declBind(op, ctx.scalar(msl::Scalar::I1), body, [&](int r) {
      StringRef a = reg(op->getOperand(0), r), b = reg(op->getOperand(1), r);
      msl::Expr *cmp = elementwiseExpr(bo, nullptr, a, b);
      if (!unordered)
        return cmp;
      return cmpFNaNGuard(cmp, a, b, ordered);
    });
  }

  // Select: `sc id = c ? t : f;`
  if (auto s = dyn_cast<arith::SelectOp>(op)) {
    Type re = op->getResult(0).getType();
    if (auto rt = dyn_cast<RankedTensorType>(re))
      re = rt.getElementType();
    msl::Type *declTy = isa<tt::PointerType>(re)
                            ? storageType(op->getResult(0).getType())
                            : scalarType(elementScalarType(re));
    return declBind(op, declTy, body, [&](int r) {
      return selectExpr(reg(s.getCondition(), r), reg(s.getTrueValue(), r),
                        reg(s.getFalseValue(), r));
    });
  }

  // Clamp.
  if (auto c = dyn_cast<tt::ClampFOp>(op))
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return clampExpr(c, reg(c.getX(), r), reg(c.getMin(), r),
                       reg(c.getMax(), r));
    });

  // fp8 (e4m3/e5m2) has no native MSL scalar: it is uchar storage with
  // bit-twiddling pack/unpack helpers. Any float<->fp8 cast routes here first.
  if (isa<arith::TruncFOp, arith::ExtFOp, tt::FpToFpOp>(op)) {
    Type dstE = elementScalarType(op->getResult(0).getType());
    Type srcE = elementScalarType(op->getOperand(0).getType());
    bool dstFp8 = isFp8Type(dstE), srcFp8 = isFp8Type(srcE);
    if (dstFp8 || srcFp8) {
      auto fp8Fn = [](Type t, bool toF8) -> std::string {
        const char *k = isa<Float8E4M3FNType>(t) ? "e4m3" : "e5m2";
        return toF8 ? std::string("tt_f32_to_fp8") + k + "_rtne"
                    : std::string("tt_fp8") + k + "_to_f32";
      };
      return declBind(op, scalarType(dstE), body, [&](int r) -> msl::Expr * {
        msl::Expr *src = ctx.var(reg(op->getOperand(0), r));
        if (srcFp8) {
          msl::Expr *b = ctx.cast(msl::Cast::Style::CStyle,
                                  ctx.scalar(msl::Scalar::U8), src);
          msl::Expr *f = ctx.call(fp8Fn(srcE, false), {b});
          if (dstE.isF32())
            return f;
          return ctx.cast(msl::Cast::Style::CStyle, scalarType(dstE), f);
        }
        msl::Expr *f = ctx.cast(msl::Cast::Style::CStyle,
                                ctx.scalar(msl::Scalar::F32), src);
        return ctx.call(fp8Fn(dstE, true), {f});
      });
    }
  }

  // Casts (non fp-narrowing) / bitcast / ptr<->int.
  if (isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
          arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return castExpr(op, reg(op->getOperand(0), r));
    });
  // TruncF / FpToFp: f32->half/bfloat narrowing calls a preamble helper
  // (tt_rtz_* / tt_rtne_*); other float casts are a plain static_cast.
  if (isa<arith::TruncFOp, tt::FpToFpOp>(op)) {
    std::string dst =
        mslScalarType(elementScalarType(op->getResult(0).getType()));
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
      std::string fn = std::string(rtz ? "tt_rtz_" : "tt_rtne_") + dst;
      msl::Type *dstTy = scalarType(resElem);
      return declBind(op, dstTy, body, [&](int r) {
        msl::Expr *f =
            ctx.cast(msl::Cast::Style::CStyle, ctx.scalar(msl::Scalar::F32),
                     ctx.var(reg(op->getOperand(0), r)));
        return ctx.call(fn, {f});
      });
    }
    // Non-narrowing float cast: static_cast<dst>(v).
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return castExpr(op, reg(op->getOperand(0), r));
    });
  }
  if (isa<arith::BitcastOp, tt::BitcastOp>(op))
    return declBind(
        op, storageType(op->getResult(0).getType()), body,
        [&](int r) { return bitcastExpr(op, reg(op->getOperand(0), r)); });
  if (isa<tt::IntToPtrOp, tt::PtrToIntOp>(op))
    return declBind(
        op, storageType(op->getResult(0).getType()), body,
        [&](int r) { return ptrIntCastExpr(op, reg(op->getOperand(0), r)); });

  // negf: `sc id = -a;`
  if (isa<arith::NegFOp>(op))
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return ctx.unary(msl::UnOp::Neg, ctx.var(reg(op->getOperand(0), r)));
    });

  // min/max family (decl type is always the signed scalar; opCast/propagateNan
  // set per op). Covers arith min/max, mulhi, remf(fmod).
  {
    StringRef fn;
    msl::Type *opCast = nullptr;
    bool propagateNan = false;
    bool isMinMax = true;
    if (isa<arith::MaximumFOp>(op)) {
      fn = "max";
      propagateNan = true;
    } else if (isa<arith::MinimumFOp>(op)) {
      fn = "min";
      propagateNan = true;
    } else if (isa<arith::MaxUIOp>(op)) {
      fn = "max";
      opCast = unsignedType(resElem);
    } else if (isa<arith::MinUIOp>(op)) {
      fn = "min";
      opCast = unsignedType(resElem);
    } else if (isa<arith::MaxNumFOp, arith::MaxSIOp>(op))
      fn = "max";
    else if (isa<arith::MinNumFOp, arith::MinSIOp>(op))
      fn = "min";
    else if (isa<arith::RemFOp>(op))
      fn = msl::builtin::math::Fmod;
    else if (isa<tt::MulhiUIOp>(op)) {
      fn = "mulhi";
      opCast = unsignedType(resElem);
    } else
      isMinMax = false;
    if (isMinMax)
      return declBind(op, scalarType(resElem), body, [&](int r) {
        return minMaxExpr(fn, opCast, propagateNan, reg(op->getOperand(0), r),
                          reg(op->getOperand(1), r));
      });
  }

  // precise_sqrt: `sc id = (sc)metal::precise::sqrt(a);`
  if (isa<tt::PreciseSqrtOp>(op))
    return declBind(op, scalarType(resElem), body, [&](int r) {
      return unaryExpr(msl::builtin::precise::Sqrt, scalarType(resElem),
                       reg(op->getOperand(0), r));
    });

  return std::nullopt;
}

// math.* dialect: unary/binary transcendentals, fma, exp10.
std::optional<bool> MSLEmitter::emitMath(Operation *op, msl::Block &body) {
  Type resElem = op->getNumResults()
                     ? elementScalarType(op->getResult(0).getType())
                     : Type();
  if (op->getDialect() ==
      op->getContext()->getLoadedDialect<math::MathDialect>()) {
    msl::Type *sc = scalarType(resElem);
    StringRef n = op->getName().getStringRef();
    namespace bi = msl::builtin;
    static const llvm::StringMap<StringRef> unary = {
        {"math.exp", bi::precise::Exp},     {"math.exp2", bi::precise::Exp2},
        {"math.log", bi::precise::Log},     {"math.log2", bi::precise::Log2},
        {"math.log10", bi::precise::Log10}, {"math.sin", bi::precise::Sin},
        {"math.cos", bi::precise::Cos},     {"math.tan", bi::precise::Tan},
        {"math.tanh", bi::precise::Tanh},   {"math.sinh", bi::precise::Sinh},
        {"math.cosh", bi::precise::Cosh},   {"math.asin", bi::precise::Asin},
        {"math.acos", bi::precise::Acos},   {"math.atan", bi::precise::Atan},
        {"math.sqrt", bi::precise::Sqrt},   {"math.rsqrt", bi::precise::Rsqrt},
        {"math.cbrt", bi::precise::Cbrt},   {"math.floor", bi::math::Floor},
        {"math.ceil", bi::math::Ceil},      {"math.absf", bi::math::Fabs},
        {"math.absi", bi::math::Abs},       {"math.erf", "tt_erf"},
        {"math.round", bi::math::Round},    {"math.trunc", bi::math::Trunc},
        {"math.roundeven", bi::math::Rint}};
    // fp8 is uchar storage, so fabs would be ambiguous (and meaningless on the
    // raw byte). Both OCP formats are sign-magnitude with the sign in bit 7,
    // so clearing it is abs for e4m3 and e5m2 alike.
    if (n == "math.absf" && isFp8Type(resElem))
      return declBind(op, sc, body, [&](int r) {
        return ctx.paren(ctx.binary(B::And, ctx.var(reg(op->getOperand(0), r)),
                                    ctx.lit("0x7f")));
      });
    if (auto it = unary.find(n); it != unary.end()) {
      StringRef fn = it->second;
      return declBind(op, sc, body, [&](int r) {
        return unaryExpr(fn, sc, reg(op->getOperand(0), r));
      });
    }
    static const llvm::StringMap<StringRef> binary = {
        {"math.atan2", bi::precise::Atan2},
        {"math.powf", bi::precise::Pow},
        {"math.fpowi", bi::precise::Pow},
        {"math.copysign", bi::math::Copysign}};
    if (auto it = binary.find(n); it != binary.end()) {
      StringRef fn = it->second;
      return declBind(op, sc, body, [&](int r) {
        return minMaxExpr(fn, nullptr, false, reg(op->getOperand(0), r),
                          reg(op->getOperand(1), r));
      });
    }
    if (n == "math.fma")
      return declBind(op, sc, body, [&](int r) {
        return ternaryCallExpr(bi::math::Fma, reg(op->getOperand(0), r),
                               reg(op->getOperand(1), r),
                               reg(op->getOperand(2), r));
      });
    if (n == "math.exp10")
      return declBind(op, sc, body, [&](int r) {
        // pow((sc)10, a)
        msl::Expr *ten = ctx.cast(CS::CStyle, sc, ctx.lit("10"));
        return ctx.call(bi::precise::Pow,
                        {ten, ctx.var(reg(op->getOperand(0), r))});
      });
    return false; // unhandled math op: caller emits the error
  }

  return std::nullopt;
}

// Shape ops: splat/unsplat/expand/broadcast/join/split (rebinds) +
// trans/reshape.
std::optional<bool> MSLEmitter::emitReshape(Operation *op, msl::Block &body) {
  // Pure register-rebind ops (no text emitted): splat / expand_dims / broadcast
  // / join / split / unsplat. Their handlers only rewrite valMap, so calling
  // them here writes nothing and keeps the symbol table correct.
  if (auto sp = dyn_cast<tt::SplatOp>(op))
    return succeeded(emitSplat(sp));
  if (auto u = dyn_cast<tt::UnsplatOp>(op)) {
    bindScalar(u.getResult(), names(u.getSrc())[0]);
    return true;
  }
  if (auto e = dyn_cast<tt::ExpandDimsOp>(op))
    return succeeded(
        emitReshapeLike(e.getResult(), e.getSrc(), e.getAxis(), true));
  if (auto b = dyn_cast<tt::BroadcastOp>(op))
    return succeeded(emitReshapeLike(b.getResult(), b.getSrc(), -1, false));
  if (auto j = dyn_cast<tt::JoinOp>(op))
    return succeeded(emitJoin(j));
  if (auto sp = dyn_cast<tt::SplitOp>(op))
    return succeeded(emitSplit(sp));

  // tt.trans: round-trip through a threadgroup buffer keyed by row-major
  // offset.
  if (auto tr = dyn_cast<tt::TransOp>(op)) {
    if (!isa<RankedTensorType>(tr.getResult().getType()))
      return false;
    Value src = tr.getSrc();
    Value res = tr.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    auto perm = tr.getOrder();
    emitTileRoundTrip(res, src, srcTy, resTy, body, [&](int r) {
      return layout.transFlatOffset(srcTy, perm, resTy.getShape(), r);
    });
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
    emitTileRoundTrip(res, src, srcTy, resTy, body,
                      [&](int r) { return layout.flatTileOffset(srcTy, r); });
    return true;
  }

  return std::nullopt;
}

// memdesc index/subslice, local_alloc/store/load, async_copy, convert_layout.
std::optional<bool> MSLEmitter::emitMemDesc(Operation *op, msl::Block &body) {
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
    // emitFunc pre-declared every allocation so planDot could see it; only
    // mint one here if that prescan did not run (device functions).
    if (!memdescMap.count(la.getResult())) {
      msl::Type *scTy =
          ctx.named("threadgroup " + mslScalarType(mt.getElementType()));
      std::string buf = "__tg_buf_" + std::to_string(tgScratchId++);
      body.push_back(ctx.arrayDeclStmt(scTy, buf, memdescFlatSize(mt)));
      memdescMap[la.getResult()] = {buf, nullptr, memdescStrides(mt)};
    }
    if (Value init = la.getSrc()) {
      const std::string &buf = memdescMap[la.getResult()].buf;
      auto srcTy = cast<RankedTensorType>(init.getType());
      auto &vals = names(init);
      body.push_back(ctx.hardBarrier(false));
      for (int r = 0, n = regCount(init); r < n; ++r)
        body.push_back(ctx.assignStmt(
            ctx.subscript(ctx.var(buf),
                          memdescElemAddr(memdescMap[la.getResult()], srcTy, r)),
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
          ctx.subscript(ctx.var(dst.buf), memdescElemAddr(dst, srcTy, r)),
          ctx.var(vals[r])));
    body.push_back(ctx.barrier(false));
    return true;
  }

  // ttg.local_load: gather result registers from the memdesc buffer.
  if (auto ll = dyn_cast<ttg::LocalLoadOp>(op)) {
    if (localLoadIsDeadDotStage(ll)) {
      bindDataless(ll.getResult());
      return true;
    }
    auto resTy = cast<RankedTensorType>(ll.getResult().getType());
    MemDescInfo src = memdescMap[ll.getSrc()];
    msl::Type *scTy = scalarType(resTy.getElementType());
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(ll.getResult()); r < n; ++r) {
      std::string id = fresh();
      body.push_back(ctx.declStmt(
          scTy, id,
          ctx.subscript(ctx.var(src.buf), memdescElemAddr(src, resTy, r))));
      ids.push_back(id);
    }
    bindRegs(ll.getResult(), ids);
    return true;
  }

  // ttg.async_copy_global_to_local: synchronous masked per-thread stage +
  // barrier.
  if (auto ac = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    // Warps in this threadgroup, from the module attribute the launch uses.
    auto numWarpsForOp = [&](Operation *o) -> int64_t {
      if (auto mod = o->getParentOfType<ModuleOp>())
        if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
          return a.getInt();
      return 1;
    };
    // On hardware this copy is asynchronous and only lands at the matching
    // async_wait, so the pipeliner is free to issue it into a slot the current
    // trip is still reading -- with numBuffers==1 it always does. We lower it
    // to a synchronous store, which would clobber that slot immediately, so
    // the reads have to be fenced off first. Consecutive copies share one
    // fence: the second would separate two staging writes that nothing reads
    // in between.
    // Apple's own M1 GEMM issues its whole batch of async copies back to back
    // and only then waits, so the requests overlap in the queue. Match that:
    // fence once ahead of the batch, not between its members. The copies of a
    // batch write disjoint buffers and nothing reads them until the wait.
    if (!asyncCopyFenced && !barrierCoversTail(body))
      body.push_back(ctx.hardBarrier(false));
    asyncCopyFenced = true;

    // True device->threadgroup DMA: air.simdgroup_async_copy_2d moves the tile
    // without it ever entering registers, so the whole load+scatter disappears.
    // ttg.async_copy_global_to_local is literally this operation.
    if (auto ds = asyncCopyDma(ac)) {
      MemDescInfo di = memdescMap[ac.getResult()];
      auto mt = cast<ttg::MemDescType>(ac.getResult().getType());
      int64_t eb = byteWidth(mt.getElementType());
      // air.simdgroup_async_copy_2d is issued *per simdgroup*, so every warp
      // requesting the whole tile moves it numWarps times over -- measured as
      // ~2x per warp doubling (nw2 0.775, nw4 1.006, nw8 1.915 ms). Apple's own
      // GEMM partitions by simdgroup_index_in_threadgroup; do the same and give
      // warp w the row band [w*band, (w+1)*band).
      int64_t rows = mt.getShape()[0], cols = mt.getShape()[1];
      int64_t nw = numWarpsForOp(ac);
      int64_t band = (nw > 0 && rows % nw == 0) ? rows / nw : rows;
      bool split = band != rows;

      msl::Expr *dstPtr = ctx.var(di.buf);
      if (di.baseOffset)
        dstPtr = ctx.paren(ctx.binary(B::Add, dstPtr, di.baseOffset));
      // The wait may sit in an outer scope than the issue, so the token is
      // declared in the function prologue and only assigned here. One token
      // per copy *site*, reused every trip: minting a fresh name per emission
      // leaves the loop waiting on the prologue's stale tokens while its own
      // copies are never waited on at all.
      msl::Expr *srcPtr = nullptr;
      auto hit = dmaHandleFor.find(ac.getOperation());
      if (hit == dmaHandleFor.end())
        return false;
      const std::string &h = hit->second;
      srcPtr = dmaTileOrigin(*ds, dotDmaTripVar(ac));
      if (split) {
        msl::Expr *wRows =
            ctx.paren(ctx.binary(B::Mul, ctx.var(warpId), ctx.i32lit(band)));
        dstPtr = ctx.paren(ctx.binary(
            B::Add, dstPtr,
            ctx.paren(ctx.binary(B::Mul, wRows, ctx.i32lit(cols)))));
        srcPtr = ctx.binary(
            B::Add, srcPtr,
            ctx.paren(ctx.binary(B::Mul, wRows,
                                 ctx.var(scalarName(ds->rowStride)))));
      }
      body.push_back(ctx.assignStmt(
          ctx.var(h),
          ctx.call("__triton_tg_async_copy_begin_" + std::to_string(eb),
                   {dstPtr, ctx.i32lit(cols), srcPtr,
                    ctx.var(scalarName(ds->rowStride)), ctx.i32lit(band),
                    ctx.i32lit(cols)})));
      pendingDmaHandles.push_back(h);
      bindDataless(ac.getResult());
      return true;
    }
    auto srcTy = cast<RankedTensorType>(ac.getSrc().getType());
    MemDescInfo dst = memdescMap[ac.getResult()];
    auto &ptrs = names(ac.getSrc());
    bool hasMask = ac.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(ac.getMask()) : nullptr;
    // A masked-off element must take `other`, exactly as tt.load does. Leaving
    // the slot untouched keeps whatever the previous trip wrote, which with the
    // pipeliner's rotating buffers is another tile's data -- observed as NaN on
    // a boundary-masked pipelined GEMM.
    SmallVector<std::string> *other =
        ac.getOther() ? &names(ac.getOther()) : nullptr;
    auto otherOf = [&](int r) -> msl::Expr * {
      return other ? static_cast<msl::Expr *>(
                         ctx.var((*other)[other->size() == 1 ? 0 : r]))
                   : static_cast<msl::Expr *>(ctx.lit("0"));
    };
    // `buf[slot] = m ? *p : other;` -- one unconditional store either way, so
    // the slot never keeps a previous trip's value.
    auto maskedCopy = [&](int r) -> msl::Stmt * {
      return ctx.assignStmt(
          ctx.subscript(ctx.var(dst.buf), memdescElemAddr(dst, srcTy, r)),
          ctx.ternary(ctx.var((*mask)[mask->size() == 1 ? 0 : r]),
                      ctx.deref(ctx.var(ptrs[r])), otherOf(r)));
    };
    int rc = regCount(ac.getSrc());
    // Mirror the tt.load lowering: a run of registers that is contiguous in
    // device memory is one wide load, and a predicate that holds for every
    // register is hoisted so the run issues unconditionally. Without this the
    // pipelined staging is one predicated scalar load per element -- measured
    // 24 scalar copies where the register path issues 6 vector loads.
    int vw = accessVectorWidth(srcTy, ac.getSrc());
    if (vw < 2 || rc % vw != 0)
      vw = 1;
    // Slot delta between consecutive registers of a run: 1 means the run is
    // contiguous in threadgroup memory and can be stored as one vector, any
    // other constant means it is strided and the lanes are scattered
    // individually (still one wide device load).
    int64_t slotStride = 1;
    // accessVectorWidth only says the *device* addresses of a register run are
    // contiguous. The threadgroup slots have to be contiguous as well, or the
    // wide store writes the run down one row when the operand lays it out down
    // a column. A column-major fp32 operand has sizePerThread [4,1], so its
    // four consecutive registers land 64 slots apart, not 1.
    if (vw > 1) {
      auto slotOf = [&](int r) -> int64_t {
        auto c = layout.registerCoords(srcTy, r);
        auto shape = srcTy.getShape();
        int64_t off = 0;
        for (size_t d = 0; d < c.size(); ++d)
          off = off * shape[d] + c[d];
        return off;
      };
      int64_t base0 = slotOf(0);
      slotStride = slotOf(1) - base0;
      for (int k = 2; k < vw; ++k)
        if (slotOf(k) - base0 != slotStride * k) {
          slotStride = 0; // not an arithmetic progression: no wide store
          break;
        }
      if (slotStride == 0)
        vw = 1;
    }
    Type elem = elementScalarType(srcTy);
    msl::Type *scTy = scalarType(elem);
    std::string scName = mslScalarType(elem);

    auto stageRun = [&](msl::Block &into, int base) {
      if (vw == 1) {
        into.push_back(ctx.assignStmt(
            ctx.subscript(ctx.var(dst.buf), memdescElemAddr(dst, srcTy, base)),
            ctx.deref(ctx.var(ptrs[base]))));
        return;
      }
      msl::Type *vecTy = ctx.vector(cast<msl::ScalarType>(scTy)->s, vw);
      msl::Type *vecPtr = ctx.ptr(vecTy, msl::AddrSpace::Device);
      std::string vid = fresh();
      into.push_back(ctx.declStmt(
          vecTy, vid,
          ctx.deref(ctx.paren(
              ctx.cast(CS::CStyle, vecPtr, ctx.var(ptrs[base]))))));
      // The run's threadgroup slots are consecutive (verified: a vw-run maps to
      // slots s..s+vw-1 for every lane/warp), so the scatter is one vector
      // store through a threadgroup vector pointer rather than vw scalar
      // stores, each of which would recompute the full address expression.
      if (slotStride == 1) {
        msl::Type *tgVecPtr = ctx.ptr(vecTy, msl::AddrSpace::Threadgroup);
        into.push_back(ctx.assignStmt(
            ctx.deref(ctx.paren(ctx.cast(
                CS::CStyle, tgVecPtr,
                ctx.paren(ctx.binary(B::Add, ctx.var(dst.buf),
                                     memdescElemAddr(dst, srcTy, base)))))),
            ctx.var(vid)));
        return;
      }
      // Strided run (a column-major operand lays its registers down a column):
      // the wide device load still holds, only the scatter is per lane.
      for (int k = 0; k < vw; ++k)
        into.push_back(ctx.assignStmt(
            ctx.subscript(ctx.var(dst.buf),
                          memdescElemAddr(dst, srcTy, base + k)),
            ctx.subscript(ctx.var(vid), ctx.i32lit(k))));
    };

    if (hasMask && rc >= kMaskFastPathMinRegs) {
      SmallVector<msl::Expr *> ms;
      llvm::StringSet<> seen;
      for (int r = 0; r < rc; ++r) {
        StringRef mn = (*mask)[mask->size() == 1 ? 0 : r];
        if (seen.insert(mn).second)
          ms.push_back(ctx.var(mn));
      }
      msl::Block hot, cold;
      for (int base = 0; base < rc; base += vw)
        stageRun(hot, base);
      // Every register under one predicate: the hot guard is already the whole
      // condition, so the fallback arm is unreachable and only doubles the
      // load count (`if (m) {...} else { if (m) {...} }`).
      if (ms.size() == 1) {
        body.push_back(ctx.ifScope(ms[0], std::move(hot)));
        bindDataless(ac.getResult());
        return true;
      }
      // The cold arm still runs on the ragged trip, so it is worth
      // vectorising too: when every register of a run shares one mask name the
      // whole run is one predicated wide copy instead of vw scalar ones.
      for (int base = 0; base < rc;) {
        StringRef m0 = (*mask)[mask->size() == 1 ? 0 : base];
        bool uniformRun = vw > 1 && base + vw <= rc;
        for (int k = 1; uniformRun && k < vw; ++k)
          uniformRun = (*mask)[mask->size() == 1 ? 0 : base + k] == m0;
        if (uniformRun) {
          // Whole run shares one predicate: wide copy when it holds, `other`
          // into every slot when it does not.
          msl::Block run, none;
          stageRun(run, base);
          for (int k = 0; k < vw; ++k)
            none.push_back(ctx.assignStmt(
                ctx.subscript(ctx.var(dst.buf),
                              memdescElemAddr(dst, srcTy, base + k)),
                otherOf(base + k)));
          cold.push_back(
              ctx.ifElseScope(ctx.var(m0), std::move(run), std::move(none)));
          base += vw;
          continue;
        }
        cold.push_back(maskedCopy(base));
        ++base;
      }
      msl::Expr *all = ms[0];
      for (size_t i = 1; i < ms.size(); ++i)
        all = ctx.binary(B::And, all, ms[i]);
      body.push_back(ctx.ifElseScope(all, std::move(hot), std::move(cold)));
    } else if (hasMask) {
      for (int r = 0; r < rc; ++r)
        body.push_back(maskedCopy(r));
    } else {
      for (int base = 0; base < rc; base += vw)
        stageRun(body, base);
    }
    // No barrier here: the copy's visibility point is the matching
    // async_wait, which emits one. Emitting a barrier per copy separates the
    // A and B staging of the same trip with a full threadgroup sync that
    // synchronises nothing -- they write different buffers and both have to
    // land before the next trip reads either.
    bindDataless(ac.getResult());
    return true;
  }

  // ttg.convert_layout: full-tile threadgroup round-trip (3 banding modes).
  if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (convertLayoutIsDeadDotStage(cl) ||
        convertLayoutIsDeadDotStageSource(cl)) {
      bindDataless(cl.getResult());
      return true;
    }
    Value src = cl.getSrc(), res = cl.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    // Identical linear layouts place register r of both sides at the same tile
    // offset for every (register, lane, warp), so the scatter/gather pair would
    // hand each thread back exactly the values it wrote: rebind instead.
    if (ttg::toLinearLayout(srcTy) == ttg::toLinearLayout(resTy)) {
      bindRegs(res, names(src));
      return true;
    }
    // Every destination element already sits in the destination lane's own
    // simdgroup: a lane permutation, which simd_shuffle does with no
    // threadgroup traffic and no barrier.
    if (emitIntraWarpShuffleConvert(cl, body))
      return true;
    mslReject(cl, "convertLayout", "threadgroup-roundtrip");
    Type elemTy = resTy.getElementType();
    bool isPtr = isa<tt::PointerType>(elemTy);
    std::string ptrTyStr = mslStorageType(resTy);
    std::string scStr = isPtr ? "ulong" : ptrTyStr;
    msl::Type *scTy = isPtr ? ctx.scalar(msl::Scalar::U64) : storageType(resTy);
    msl::Type *ptrDeclTy = storageType(resTy);
    auto &srcNames = names(src);
    int64_t elemBytes = byteWidth(elemTy);
    int64_t tileBytes = tileSize(resTy) * elemBytes;
    int rank = resTy.getRank();
    ArrayRef<int64_t> shape = resTy.getShape();

    std::string buf = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                                poolRegion(0, scStr)));
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
      if (bandRows < 1)
        bandRows = 1;
      int64_t rowsTotal = shape[rank - 2];
      auto srcOut =
          llvm::to_vector(ttg::toLinearLayout(srcTy).getOutDimNames());
      StringAttr srcRowDim = srcOut[rank - 2];
      auto resOut =
          llvm::to_vector(ttg::toLinearLayout(resTy).getOutDimNames());
      StringAttr resRowDim = resOut[rank - 2];
      SmallVector<std::string> ids(regCount(res));

      auto bandOffset = [&](RankedTensorType rt, int reg,
                            int64_t r0) -> msl::Expr * {
        auto outN = llvm::to_vector(ttg::toLinearLayout(rt).getOutDimNames());
        msl::Expr *expr = nullptr;
        int64_t stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
          msl::Expr *c = layout.layoutCoordExpr(rt, reg, outN[d]);
          if (d == rank - 2)
            c = ctx.paren(ctx.binary(B::Sub, c, ctx.lit(std::to_string(r0))));
          msl::Expr *term =
              stride == 1 ? c
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
          msl::Expr *rowc = layout.layoutCoordExpr(srcTy, r, srcRowDim);
          msl::Expr *cond = ctx.binary(
              B::LAnd, ctx.binary(B::Ge, rowc, ctx.lit(std::to_string(r0))),
              ctx.binary(B::Lt, layout.layoutCoordExpr(srcTy, r, srcRowDim),
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
          msl::Expr *rowc = layout.layoutCoordExpr(resTy, r, resRowDim);
          msl::Expr *cond = ctx.binary(
              B::LAnd, ctx.binary(B::Ge, rowc, ctx.lit(std::to_string(r0))),
              ctx.binary(B::Lt, layout.layoutCoordExpr(resTy, r, resRowDim),
                         ctx.lit(std::to_string(r1))));
          msl::Expr *rd =
              gatherVal(ctx.subscript(ctx.var(buf), bandOffset(resTy, r, r0)));
          body.push_back(
              ctx.compactIf(cond, ctx.assignStmt(ctx.var(ids[r]), rd)));
        }
      }
      bindRegs(res, ids);
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
                                   layout.flatTileOffset(srcTy, r)));
          msl::Expr *cond = ctx.binary(
              B::LAnd,
              ctx.binary(B::Ge, ctx.var("__f"), ctx.lit(std::to_string(lo))),
              ctx.binary(B::Lt, ctx.var("__f"), ctx.lit(std::to_string(hi))));
          msl::Expr *idx =
              ctx.binary(B::Sub, ctx.var("__f"), ctx.lit(std::to_string(lo)));
          b.push_back(ctx.compactIf(
              cond, ctx.assignStmt(ctx.subscript(ctx.var(buf), idx),
                                   scatterVal(srcNames[r]))));
          body.push_back(ctx.plainScope(std::move(b)));
        }
        body.push_back(ctx.hardBarrier(false));
        for (int r = 0, n = regCount(res); r < n; ++r) {
          msl::Block b;
          b.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), "__f",
                                   layout.flatTileOffset(resTy, r)));
          msl::Expr *cond = ctx.binary(
              B::LAnd,
              ctx.binary(B::Ge, ctx.var("__f"), ctx.lit(std::to_string(lo))),
              ctx.binary(B::Lt, ctx.var("__f"), ctx.lit(std::to_string(hi))));
          msl::Expr *idx =
              ctx.binary(B::Sub, ctx.var("__f"), ctx.lit(std::to_string(lo)));
          msl::Expr *rd = gatherVal(ctx.subscript(ctx.var(buf), idx));
          b.push_back(ctx.compactIf(cond, ctx.assignStmt(ctx.var(ids[r]), rd)));
          body.push_back(ctx.plainScope(std::move(b)));
        }
      }
      bindRegs(res, ids);
      return true;
    }

    body.push_back(ctx.hardBarrier(false));
    for (int r = 0, n = regCount(src); r < n; ++r)
      body.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(buf), layout.flatTileOffset(srcTy, r)),
          scatterVal(srcNames[r])));
    body.push_back(ctx.hardBarrier(false));
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(res); r < n; ++r) {
      std::string id = fresh();
      msl::Expr *rd = gatherVal(
          ctx.subscript(ctx.var(buf), layout.flatTileOffset(resTy, r)));
      body.push_back(ctx.declStmt(ptrDeclTy, id, rd));
      ids.push_back(id);
    }
    bindRegs(res, ids);
    return true;
  }

  return std::nullopt;
}

// tt.dot / tt.map_elementwise / tt.histogram.
std::optional<bool> MSLEmitter::emitDotMap(Operation *op, msl::Block &body) {
  // tt.dot: simdgroup-matrix / scalar GEMM (all staging + fusion phases).
  if (auto dt = dyn_cast<tt::DotOp>(op))
    return emitDot(dt, body);

  // tt.map_elementwise: per-group inline of the scalar region. Single-block
  // groups AST-walk the region; multi-block groups use the state-machine
  // lowering (emitMapCFG).
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
          body.push_back(mapCaptureDecl(
              mslScalarType(elementScalarType(r.getType())), capture[i]));
        }
        emitMapCFG(region, capture, body);
        if (emitFailed)
          return false;
        for (int k = 0; k < nRes; ++k)
          for (int p = 0; p < pack; ++p)
            resIds[k].push_back(capture[k * pack + p]);
        continue;
      }
      for (Operation &o : blk.without_terminator()) {
        if (!emitOp(&o, body)) {
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
      bindRegs(mp->getResult(k), resIds[k]);
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
    body.push_back(atomic.histBinsDecl(bins));
    std::string zi = fresh();
    body.push_back(atomic.histZeroInit(bins, zi, nBins, threads));
    body.push_back(ctx.hardBarrier(false));

    auto &srcVals = names(hg.getSrc());
    SmallVector<std::string> *maskVals =
        hg.getMask() ? &names(hg.getMask()) : nullptr;
    std::string srcU =
        mslUnsignedType(elementScalarType(hg.getSrc().getType()));

    MLIRContext *mctx = hg.getContext();
    tt::LinearLayout srcLL = ttg::toLinearLayout(srcTy);
    auto kLane = StringAttr::get(mctx, "lane");
    auto kWarp = StringAttr::get(mctx, "warp");
    auto srcOut = llvm::to_vector(srcLL.getOutDimNames());
    uint32_t freeMask = 0;
    auto scanFree = [&](StringAttr in, int shift) {
      if (!srcLL.hasInDim(in))
        return;
      for (int b = 0, n = srcLL.getInDimSizeLog2(in); b < n; ++b) {
        bool moves = false;
        for (auto od : srcOut)
          if (srcLL.getBasis(in, b, od) != 0)
            moves = true;
        if (!moves)
          freeMask |= 1u << (shift + b);
      }
    };
    scanFree(kLane, 0);
    scanFree(kWarp, 5);
    // (tidId.x & freeMask u) == 0u
    msl::Expr *ownerGuard =
        freeMask == 0
            ? nullptr
            : ctx.binary(
                  B::Eq,
                  ctx.paren(ctx.binary(B::And, ctx.member(ctx.var(tidId), "x"),
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
      body.push_back(atomic.histFetchAdd(guard, bins, v));
    }
    body.push_back(ctx.hardBarrier(false));

    auto outDims = llvm::to_vector(ttg::toLinearLayout(resTy).getOutDimNames());
    msl::Type *resScTy = scalarType(resTy.getElementType());
    std::string resSc = mslScalarType(resTy.getElementType());
    int nResReg = regCount(hg.getResult());
    SmallVector<std::string> resIds;
    for (int r = 0; r < nResReg; ++r) {
      msl::Expr *idx = layout.layoutCoordExpr(resTy, r, outDims[0]);
      std::string id = fresh();
      // (resSc)atomic_load_explicit(&bins[idx], memory_order_relaxed)
      msl::Expr *load =
          ctx.cast(CS::CStyle, resScTy,
                   ctx.call("atomic_load_explicit",
                            {ctx.addrOf(ctx.subscript(ctx.var(bins), idx)),
                             ctx.lit("memory_order_relaxed")}));
      body.push_back(ctx.declStmt(resScTy, id, load));
      resIds.push_back(id);
    }
    bindRegs(hg.getResult(), resIds);
    return true;
  }

  return std::nullopt;
}

// tt.atomic_rmw: native fetch_* (AST) or fp-emulated CAS loop (captured),
// with redundant-thread guard + lane/warp replica broadcast.
std::optional<bool> MSLEmitter::emitAtomicRMW(Operation *op, msl::Block &body) {
  if (auto ar = dyn_cast<tt::AtomicRMWOp>(op)) {
    Value res = ar.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    msl::Type *scTy = scalarType(scalarTy);
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
    msl::Scalar atomicTy =
        isFloat ? msl::Scalar::F32
        : (kind == tt::RMWOp::UMAX || kind == tt::RMWOp::UMIN)
            ? msl::Scalar::U32
            : (bw == 64 ? msl::Scalar::I64 : msl::Scalar::I32);
    StringRef fn;
    if (!floatEmulated) {
      switch (kind) {
      case tt::RMWOp::ADD:
      case tt::RMWOp::FADD:
        fn = msl::builtin::atomic::FetchAdd;
        break;
      case tt::RMWOp::MAX:
      case tt::RMWOp::UMAX:
        fn = msl::builtin::atomic::FetchMax;
        break;
      case tt::RMWOp::MIN:
      case tt::RMWOp::UMIN:
        fn = msl::builtin::atomic::FetchMin;
        break;
      case tt::RMWOp::AND:
        fn = msl::builtin::atomic::FetchAnd;
        break;
      case tt::RMWOp::OR:
        fn = msl::builtin::atomic::FetchOr;
        break;
      case tt::RMWOp::XOR:
        fn = msl::builtin::atomic::FetchXor;
        break;
      case tt::RMWOp::XCHG:
        fn = msl::builtin::atomic::Exchange;
        break;
      default:
        return false;
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
      std::string wp =
          "((" + warpId + " & " + std::to_string(warpFree) + ") == 0)";
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
      body.push_back(ctx.declStmt(scTy, id, atomic.init0(sc)));
      msl::Expr *guard = nullptr;
      if (uniform)
        guard =
            ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0"));
      else if (!threadPred.empty())
        guard = ctx.var(threadPred);
      if (hasMask) {
        const std::string &m = (*mask)[mask->size() == 1 ? 0 : r];
        guard = guard ? static_cast<msl::Expr *>(
                            ctx.binary(B::LAnd, guard, ctx.var(m)))
                      : ctx.var(m);
      }
      msl::Block inner;
      if (floatEmulated) {
        int rmwOp = kind == tt::RMWOp::MAX                                ? 1
                    : kind == tt::RMWOp::MIN                              ? 2
                    : (kind == tt::RMWOp::ADD || kind == tt::RMWOp::FADD) ? 0
                                                                          : 3;
        msl::Expr *fv = ctx.cast(msl::Cast::Style::CStyle,
                                 ctx.scalar(msl::Scalar::F32), ctx.var(v));
        msl::Expr *call;
        if (bw == 16) {
          std::string wordPtr, isHigh;
          for (msl::Stmt *s : atomic.packed16Base(p, wordPtr, isHigh))
            inner.push_back(s);
          call = ctx.call(
              "tt_atomic_rmw_packed16<" + sc + ", tt_rtne_int_" + sc + ">",
              {ctx.var(wordPtr), ctx.var(isHigh), fv, ctx.i32lit(rmwOp)});
        } else {
          std::string wordPtr = fresh();
          msl::Type *aup = ctx.deviceAtomicPtr(msl::Scalar::U32);
          inner.push_back(ctx.declStmt(
              aup, wordPtr, ctx.cast(CS::CStyle, aup, ctx.paren(ctx.var(p)))));
          call = ctx.call("tt_atomic_rmw_f32",
                          {ctx.var(wordPtr), fv, ctx.i32lit(rmwOp)});
        }
        inner.push_back(ctx.assignStmt(ctx.var(id), call));
      } else {
        // Metal device atomics are relaxed-only; acquire/release/acq_rel are
        // not valid MSL memory orders, so the requested order is carried by
        // device-scope fences around the relaxed op instead.
        if (sem == tt::MemSemantic::RELEASE ||
            sem == tt::MemSemantic::ACQUIRE_RELEASE)
          inner.push_back(atomic.deviceFence());
        inner.push_back(
            ctx.assignStmt(ctx.var(id), atomic.rmwCall(fn, atomicTy, p, v,
                                                       "memory_order_relaxed",
                                                       /*memFlags=*/false)));
        if (sem == tt::MemSemantic::ACQUIRE ||
            sem == tt::MemSemantic::ACQUIRE_RELEASE)
          inner.push_back(atomic.deviceFence());
      }
      if (guard)
        body.push_back(ctx.ifScope(guard, std::move(inner)));
      else
        for (msl::Stmt *s : inner)
          body.push_back(s);
      if (!uniform && laneFree) {
        std::string src =
            "(uint)(" + laneId + " & " + std::to_string(~laneFree & 31) + ")";
        id = shuffle(msl::builtin::simd::Shuffle, sc, id, src, body);
      }
      ids[r] = id;
    }

    if (!uniform && warpFree) {
      auto ptrTy = cast<RankedTensorType>(ar.getPtr().getType());
      tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
      MLIRContext *c = ar.getContext();
      int64_t numWarps = ll.hasInDim(StringAttr::get(c, "warp"))
                             ? ll.getInDimSize(StringAttr::get(c, "warp"))
                             : 1;
      std::string bcbuf = fresh();
      body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup),
                                  bcbuf, poolRegion(0, sc)));
      body.push_back(ctx.hardBarrier(false));
      // ((warpId & warpFree) == 0)
      msl::Expr *wcanon = ctx.paren(ctx.binary(
          B::Eq,
          ctx.paren(ctx.binary(B::And, ctx.var(warpId), ctx.i32lit(warpFree))),
          ctx.i32lit(0)));
      // (warpId & (~warpFree & (numWarps-1)))
      msl::Expr *warpKey = ctx.paren(ctx.binary(
          B::And, ctx.var(warpId), ctx.i32lit(~warpFree & (numWarps - 1))));
      // ((warpKey * rc*32) + reg*32 + (laneId & (~laneFree & 31)))
      auto slotFor = [&](int reg) -> msl::Expr * {
        return ctx.paren(
            ctx.addChain({ctx.paren(ctx.mul(warpKey, ctx.i32lit(rc * 32))),
                          ctx.mul(ctx.i32lit(reg), ctx.i32lit(32)),
                          ctx.paren(ctx.binary(B::And, ctx.var(laneId),
                                               ctx.i32lit(~laneFree & 31)))}));
      };
      for (int r = 0; r < rc; ++r) {
        if (regFree && (r & regFree) != 0)
          continue;
        body.push_back(ctx.compactIf(
            wcanon, ctx.assignStmt(ctx.subscript(ctx.var(bcbuf), slotFor(r)),
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
    bindRegs(res, ids);
    return true;
  }

  return std::nullopt;
}

// tt.atomic_poll: elected-thread spin/probe on the aligned word + broadcast.
std::optional<bool> MSLEmitter::emitAtomicPoll(Operation *op,
                                               msl::Block &body) {
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
    msl::Scalar wordTy = bw == 16   ? msl::Scalar::U16
                         : bw == 64 ? msl::Scalar::U64
                                    : msl::Scalar::U32;

    // Bind wordPtr decls into `into`, returning the load-expr string.
    auto probe = [&](msl::Block &into) -> std::string {
      std::string loadExpr;
      if (bw == 16) {
        std::string wordPtr, isHigh;
        for (msl::Stmt *s : atomic.packed16Base(p, wordPtr, isHigh))
          into.push_back(s);
        loadExpr = "(ushort)((" + isHigh + ") ? (atomic_load_explicit(" +
                   wordPtr +
                   ", memory_order_relaxed) >> 16) : (atomic_load_explicit(" +
                   wordPtr + ", memory_order_relaxed) & 0xffffu))";
        return loadExpr;
      }
      std::string wordPtr = fresh();
      if (bw == 64) {
        msl::Type *vup =
            ctx.ptr(ctx.scalar(msl::Scalar::U64), msl::AddrSpace::Device,
                    /*coherent=*/false, /*vol=*/true, /*spaceStar=*/true);
        into.push_back(ctx.declStmt(
            vup, wordPtr, ctx.cast(CS::CStyle, vup, ctx.paren(ctx.var(p)))));
        loadExpr = "(*" + wordPtr + ")";
      } else {
        msl::Type *aup = ctx.deviceAtomicPtr(msl::Scalar::U32);
        into.push_back(ctx.declStmt(
            aup, wordPtr, ctx.cast(CS::CStyle, aup, ctx.paren(ctx.var(p)))));
        loadExpr =
            "atomic_load_explicit(" + wordPtr + ", memory_order_relaxed)";
      }
      return loadExpr;
    };

    std::string result = fresh();
    if (!pl.getTimeout()) {
      msl::Block ifBody;
      std::string loadExpr = probe(ifBody);
      std::string want = fresh();
      ifBody.push_back(
          ctx.declStmt(ctx.scalar(wordTy), want,
                       ctx.cast(CS::CStyle, ctx.scalar(wordTy), ctx.var(exp))));
      ifBody.push_back(ctx.whileScope(
          ctx.binary(B::Ne, ctx.raw(loadExpr), ctx.var(want)), msl::Block{}));
      body.push_back(ctx.ifScope(
          ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
          std::move(ifBody)));
      body.push_back(pollBarrier);
      body.push_back(
          ctx.declStmt(ctx.scalar(msl::Scalar::I1), result, ctx.lit("true")));
      bindRegs(pl.getResult(), {result});
      return true;
    }
    std::string flag = fresh();
    body.push_back(ctx.declStmt(ctx.named("threadgroup bool"), flag, nullptr));
    msl::Block ifBody;
    std::string loadExpr = probe(ifBody);
    std::string want = fresh(), loaded = fresh();
    ifBody.push_back(
        ctx.declStmt(ctx.scalar(wordTy), want,
                     ctx.cast(CS::CStyle, ctx.scalar(wordTy), ctx.var(exp))));
    ifBody.push_back(
        ctx.declStmt(ctx.scalar(wordTy), loaded, ctx.raw(loadExpr)));
    ifBody.push_back(ctx.assignStmt(
        ctx.var(flag),
        ctx.paren(ctx.binary(B::Eq, ctx.var(loaded), ctx.var(want)))));
    body.push_back(ctx.ifScope(
        ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
        std::move(ifBody)));
    body.push_back(pollBarrier);
    body.push_back(
        ctx.declStmt(ctx.scalar(msl::Scalar::I1), result, ctx.var(flag)));
    bindRegs(pl.getResult(), {result});
    return true;
  }

  return std::nullopt;
}

// tt.atomic_cas: int32/float32/packed16 compare-exchange + uniform spinlock.
std::optional<bool> MSLEmitter::emitAtomicCAS(Operation *op, msl::Block &body) {
  if (auto ca = dyn_cast<tt::AtomicCASOp>(op)) {
    Value res = ca.getResult();
    Type scalarTy = elementScalarType(res.getType());
    std::string sc = mslScalarType(scalarTy);
    msl::Type *scTy = scalarType(scalarTy);
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
    tt::MemSemantic casSem = ca.getSem();
    int rc = ptrs.size();
    if (casSem == tt::MemSemantic::RELEASE ||
        casSem == tt::MemSemantic::ACQUIRE_RELEASE)
      body.push_back(atomic.deviceFence());
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      const std::string &p = ptrs[r];
      const std::string &c = cmps[cmps.size() == 1 ? 0 : r];
      const std::string &v = vals[vals.size() == 1 ? 0 : r];
      std::string id = fresh();
      msl::Block casBody;
      // The CAS leaf declares `id` when !uniform; when uniform we pre-declare
      // it.
      if (packed16) {
        std::string wp, ih;
        for (msl::Stmt *s : atomic.packed16Base(p, wp, ih))
          casBody.push_back(s);
        for (msl::Stmt *s : atomic.packed16CAS(wp, ih, c, v, sc, id, !uniform))
          casBody.push_back(s);
      } else if (isFloat)
        for (msl::Stmt *s : atomic.float32CAS(p, c, v, id, !uniform))
          casBody.push_back(s);
      else
        for (msl::Stmt *s : atomic.int32CAS(p, c, v, sc, id, !uniform))
          casBody.push_back(s);

      if (uniform) {
        body.push_back(ctx.declStmt(scTy, id, ctx.var(c)));
        body.push_back(ctx.ifScope(
            ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
            std::move(casBody)));
        // Broadcast lane-0's result to every lane through a scratch slot.
        std::string bcast = fresh();
        body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup),
                                    bcast, poolRegion(0, sc)));
        body.push_back(ctx.compactIf(
            ctx.binary(B::Eq, ctx.member(ctx.var(tidId), "x"), ctx.lit("0")),
            ctx.assignStmt(ctx.subscript(ctx.var(bcast), ctx.lit("0")),
                           ctx.var(id))));
        body.push_back(ctx.hardBarrier(false));
        body.push_back(ctx.assignStmt(
            ctx.var(id), ctx.subscript(ctx.var(bcast), ctx.lit("0"))));
        body.push_back(ctx.hardBarrier(false));
        // Every lane (not just the CAS lane) must see the winner's device
        // writes, so the acquire fence follows the broadcast.
        if (casSem == tt::MemSemantic::ACQUIRE ||
            casSem == tt::MemSemantic::ACQUIRE_RELEASE)
          body.push_back(atomic.deviceFence());
      } else {
        for (msl::Stmt *s : casBody)
          body.push_back(s);
      }
      ids.push_back(id);
    }
    if (!uniform && (casSem == tt::MemSemantic::ACQUIRE ||
                     casSem == tt::MemSemantic::ACQUIRE_RELEASE))
      body.push_back(atomic.deviceFence());
    bindRegs(res, ids);
    return true;
  }

  return std::nullopt;
}

// tt.scan / tt.reduce: cross-lane/warp fold + prefix.
// tt.scan: per-run register fold + lane shuffle prefix + cross-warp carry +
// cross-run carry.
std::optional<bool> MSLEmitter::emitScan(Operation *op, msl::Block &body) {
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
      scTypes[k] = scalarType(elementScalarType(sn.getResult()[k].getType()));
      byteWidths[k] = byteWidth(elementScalarType(sn.getResult()[k].getType()));
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
    for (auto &pr : laneBits)
      axisLaneMask |= (1u << pr.first);
    unsigned axisLaneLow = axisLaneMask & (~axisLaneMask + 1);
    unsigned normMask = axisLaneMask / (axisLaneLow ? axisLaneLow : 1);
    if (axisLaneMask && (normMask & (normMask + 1)))
      return false; // unsupported lane layout: caller emits the error
    unsigned axisWarpMask = 0;
    for (auto &pr : warpBits)
      axisWarpMask |= (1u << pr.first);
    int numWarps = ll.hasInDim(kWarp) ? ll.getInDimSize(kWarp) : 1;

    int32_t laneWarpReach = 0;
    for (auto &pr : laneBits)
      laneWarpReach = std::max(laneWarpReach, pr.second);
    for (auto &pr : warpBits)
      laneWarpReach = std::max(laneWarpReach, pr.second);

    auto keyOf = [&](int reg) {
      SmallVector<int32_t> coords = layout.registerCoords(srcTy, reg);
      std::string key;
      for (int d = 0; d < (int)coords.size(); ++d)
        if (d != axis)
          key += std::to_string(coords[d]) + ",";
      return key;
    };
    SmallVector<int> runId(nReg, 0);
    for (int r = 0; r < nReg; ++r) {
      int32_t c = layout.registerCoords(srcTy, r)[axis];
      runId[r] = laneWarpReach ? (c / (2 * laneWarpReach)) : 0;
    }

    SmallVector<SmallVector<std::string>> accs(nOp,
                                               SmallVector<std::string>(nReg));
    for (int k = 0; k < nOp; ++k)
      for (int r = 0; r < nReg; ++r) {
        accs[k][r] = fresh();
        body.push_back(
            ctx.declStmt(scTypes[k], accs[k][r], ctx.var(srcNames[k][r])));
      }

    StringRef shuf =
        rev ? msl::builtin::simd::ShuffleDown : msl::builtin::simd::ShuffleUp;
    std::string axisTopLane =
        axisLaneMask == 0
            ? laneId
            : ("((" + laneId + " & " + std::to_string(~axisLaneMask) + ") | " +
               (rev ? "0" : std::to_string(axisLaneMask)) + ")");

    std::map<std::string, SmallVector<int>> keys;
    SmallVector<std::string> keyOrder;
    for (int r = 0; r < nReg; ++r) {
      std::string k = keyOf(r);
      if (keys.find(k) == keys.end())
        keyOrder.push_back(k);
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
          if (runId[r] == run)
            regs.push_back(r);
        llvm::sort(regs, [&](int a, int b) {
          int32_t ca = layout.registerCoords(srcTy, a)[axis];
          int32_t cb = layout.registerCoords(srcTy, b)[axis];
          return rev ? ca > cb : ca < cb;
        });

        for (size_t i = 1; i < regs.size(); ++i) {
          SmallVector<std::string> a(nOp), b(nOp);
          for (int k = 0; k < nOp; ++k) {
            a[k] = accs[k][regs[i - 1]];
            b[k] = accs[k][regs[i]];
          }
          SmallVector<std::string> out;
          if (!combineN(region, a, b, body, out))
            return false;
          for (int k = 0; k < nOp; ++k)
            body.push_back(
                ctx.assignStmt(ctx.var(accs[k][regs[i]]), ctx.var(out[k])));
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
            nb[k] = shuffle(shuf, scTys[k], laneScan[k],
                            std::to_string(delta) + "u", body);
          SmallVector<std::string> out;
          if (!combineN(region, nb, laneScan, body, out))
            return false;
          // (laneId & axisLaneMask) [<= mask-delta | >= delta]
          msl::Expr *local = ctx.paren(
              ctx.binary(B::And, ctx.var(laneId), ctx.i32lit(axisLaneMask)));
          msl::Expr *guard =
              rev ? ctx.binary(B::Le, local, ctx.i32lit(axisLaneMask - delta))
                  : ctx.binary(B::Ge, local, ctx.i32lit(delta));
          for (int k = 0; k < nOp; ++k)
            body.push_back(
                ctx.assignStmt(ctx.var(laneScan[k]),
                               ctx.paren(ctx.ternary(guard, ctx.var(out[k]),
                                                     ctx.var(laneScan[k])))));
        }
        if (!laneBits.empty()) {
          SmallVector<std::string> lanePrefix(nOp);
          for (int k = 0; k < nOp; ++k)
            lanePrefix[k] = shuffle(shuf, scTys[k], laneScan[k],
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
            for (int k = 0; k < nOp; ++k)
              ar[k] = accs[k][r];
            if (!combineN(region, lanePrefix, ar, body, out))
              return false;
            for (int k = 0; k < nOp; ++k)
              body.push_back(
                  ctx.assignStmt(ctx.var(accs[k][r]),
                                 ctx.paren(ctx.ternary(guard, ctx.var(out[k]),
                                                       ctx.var(accs[k][r])))));
          }
        }

        for (int k = 0; k < nOp; ++k) {
          runTotals[ri][k] = fresh();
          body.push_back(ctx.declStmt(scTypes[k], runTotals[ri][k], nullptr));
        }
        if (!scanWarpCarry(region, nOp, scTys, byteWidths, warpBits, regs, accs,
                           laneScan, axisTopLane, axisWarpMask, numWarps, rev,
                           runTotals[ri], body))
          return false;
      }

      for (size_t ri = 1; ri < runOrder.size(); ++ri) {
        SmallVector<std::string> carry = runTotals[0];
        for (size_t j = 1; j < ri; ++j) {
          SmallVector<std::string> out;
          if (!combineN(region, carry, runTotals[j], body, out))
            return false;
          carry = out;
        }
        int run = runOrder[ri];
        for (int r : keyRegs) {
          if (runId[r] != run)
            continue;
          SmallVector<std::string> ar(nOp);
          for (int k = 0; k < nOp; ++k)
            ar[k] = accs[k][r];
          SmallVector<std::string> out;
          if (!combineN(region, carry, ar, body, out))
            return false;
          for (int k = 0; k < nOp; ++k)
            body.push_back(
                ctx.assignStmt(ctx.var(accs[k][r]), ctx.var(out[k])));
        }
      }
    }
    for (int k = 0; k < nOp; ++k)
      bindRegs(sn.getResult()[k], accs[k]);
    return true;
  }

  return std::nullopt;
}

// tt.reduce: per-group register fold + lane-shuffle xor + optional cross-warp
// threadgroup combine.
std::optional<bool> MSLEmitter::emitReduce(Operation *op, msl::Block &body) {
  if (auto rd = dyn_cast<tt::ReduceOp>(op)) {
    int nOp = rd.getNumOperands();
    auto srcTy = cast<RankedTensorType>(rd.getOperand(0).getType());
    int axis = rd.getAxis();
    bool tensorResult = isa<RankedTensorType>(rd.getResult()[0].getType());
    SmallVector<std::string> scTys(nOp);
    SmallVector<msl::Type *> scTypes(nOp);
    for (int k = 0; k < nOp; ++k) {
      scTys[k] = mslScalarType(elementScalarType(rd.getResult()[k].getType()));
      scTypes[k] = scalarType(elementScalarType(rd.getResult()[k].getType()));
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
      SmallVector<int32_t> coords = layout.registerCoords(srcTy, reg);
      std::string key;
      for (int d = 0; d < (int)coords.size(); ++d)
        if (d != axis)
          key += std::to_string(coords[d]) + ",";
      return key;
    };
    auto fullKey = [&](int reg) {
      SmallVector<int32_t> coords = layout.registerCoords(srcTy, reg);
      std::string key;
      for (int32_t c : coords)
        key += std::to_string(c) + ",";
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
        body.push_back(
            ctx.declStmt(scTypes[k], accs[k], ctx.var(srcNames[k][regs[0]])));
      }
      for (size_t i = 1; i < regs.size(); ++i) {
        SmallVector<std::string> bVals(nOp);
        for (int k = 0; k < nOp; ++k)
          bVals[k] = srcNames[k][regs[i]];
        SmallVector<std::string> out;
        if (!combineN(region, accs, bVals, body, out))
          return false;
        for (int k = 0; k < nOp; ++k)
          body.push_back(ctx.assignStmt(ctx.var(accs[k]), ctx.var(out[k])));
      }
      for (int bit = 31; bit >= 0; --bit) {
        unsigned m = 1u << bit;
        if ((laneMask & m) == 0)
          continue;
        SmallVector<std::string> others(nOp);
        for (int k = 0; k < nOp; ++k)
          others[k] = shuffle(msl::builtin::simd::ShuffleXor, scTys[k], accs[k],
                              std::to_string(m) + "u", body);
        SmallVector<std::string> out;
        if (!combineN(region, accs, others, body, out))
          return false;
        for (int k = 0; k < nOp; ++k)
          body.push_back(ctx.assignStmt(ctx.var(accs[k]), ctx.var(out[k])));
      }
      if (warpMask != 0) {
        SmallVector<std::string> scratch(nOp);
        int64_t byteOff = 0;
        for (int k = 0; k < nOp; ++k) {
          scratch[k] = fresh();
          body.push_back(
              ctx.declStmt(ctx.ptr(scTypes[k], msl::AddrSpace::Threadgroup),
                           scratch[k], poolRegion(byteOff, scTys[k])));
          byteOff +=
              numWarps * 32 *
              std::max<int64_t>(
                  1,
                  bitsOf(elementScalarType(rd.getResult()[k].getType())) / 8);
        }
        body.push_back(ctx.hardBarrier(false));
        // scratch[k][warp * 32 + lane] = accs[k];
        for (int k = 0; k < nOp; ++k) {
          msl::Expr *idx = ctx.binary(
              B::Add, ctx.binary(B::Mul, ctx.var(warpId), ctx.lit("32")),
              ctx.var(laneId));
          body.push_back(ctx.assignStmt(ctx.subscript(ctx.var(scratch[k]), idx),
                                        ctx.var(accs[k])));
        }
        body.push_back(ctx.hardBarrier(false));
        SmallVector<int> redVals = subsetsOf(warpMask, numWarps);
        // base = ((warp & ~warpMask) * 32 + lane)
        msl::Expr *base = ctx.paren(ctx.binary(
            B::Add,
            ctx.binary(
                B::Mul,
                ctx.paren(ctx.binary(B::And, ctx.var(warpId),
                                     ctx.lit(std::to_string(~warpMask)))),
                ctx.lit("32")),
            ctx.var(laneId)));
        SmallVector<std::string> wacc(nOp);
        for (int k = 0; k < nOp; ++k) {
          wacc[k] = fresh();
          body.push_back(ctx.declStmt(
              scTypes[k], wacc[k], ctx.subscript(ctx.var(scratch[k]), base)));
        }
        for (size_t i = 1; i < redVals.size(); ++i) {
          SmallVector<std::string> wv(nOp);
          for (int k = 0; k < nOp; ++k) {
            wv[k] = fresh();
            msl::Expr *idx = ctx.binary(
                B::Add, base, ctx.lit(std::to_string(redVals[i] * 32)));
            body.push_back(ctx.declStmt(
                scTypes[k], wv[k], ctx.subscript(ctx.var(scratch[k]), idx)));
          }
          SmallVector<std::string> out;
          if (!combineN(region, wacc, wv, body, out))
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
      SmallVector<int32_t> rc = layout.registerCoords(resTy, r);
      std::string key;
      for (int32_t c : rc)
        key += std::to_string(c) + ",";
      auto it = groupResult.find(key);
      if (it == groupResult.end())
        return false;
      for (int k = 0; k < nOp; ++k)
        resIds[k].push_back(it->second[k]);
    }
    for (int k = 0; k < nOp; ++k)
      bindRegs(rd.getResult()[k], resIds[k]);
    return true;
  }

  return std::nullopt;
}

// tt.cat / tt.gather: threadgroup scatter/gather tile moves.
std::optional<bool> MSLEmitter::emitTensorMove(Operation *op,
                                               msl::Block &body) {
  // tt.cat: scatter both halves (rhs shifted past lhs flat size), gather
  // result.
  if (auto ct = dyn_cast<tt::CatOp>(op)) {
    Value lhs = ct.getLhs(), rhs = ct.getRhs(), res = ct.getResult();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    auto rhsTy = cast<RankedTensorType>(rhs.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    msl::Type *scTy = scalarType(resTy.getElementType());
    std::string sc = mslScalarType(resTy.getElementType());
    auto &lhsNames = names(lhs);
    auto &rhsNames = names(rhs);
    int64_t lhsFlat = tileSize(lhsTy);

    std::string buf = fresh();
    body.push_back(ctx.declStmt(ctx.ptr(scTy, msl::AddrSpace::Threadgroup), buf,
                                poolRegion(0, sc)));
    body.push_back(ctx.hardBarrier(false));
    for (int r = 0, n = regCount(lhs); r < n; ++r)
      body.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(buf), layout.flatTileOffset(lhsTy, r)),
          ctx.var(lhsNames[r])));
    for (int r = 0, n = regCount(rhs); r < n; ++r) {
      msl::Expr *off = ctx.binary(B::Add, layout.flatTileOffset(rhsTy, r),
                                  ctx.lit(std::to_string(lhsFlat)));
      body.push_back(ctx.assignStmt(ctx.subscript(ctx.var(buf), off),
                                    ctx.var(rhsNames[r])));
    }
    body.push_back(ctx.hardBarrier(false));
    SmallVector<std::string> ids;
    for (int r = 0, n = regCount(res); r < n; ++r) {
      std::string id = fresh();
      body.push_back(ctx.declStmt(
          scTy, id,
          ctx.subscript(ctx.var(buf), layout.flatTileOffset(resTy, r))));
      ids.push_back(id);
    }
    bindRegs(res, ids);
    return true;
  }

  // tt.gather: stage src tile, then read each result register at the
  // index-selected source offset (row-major fold, dim `axis` uses idx).
  if (auto ga = dyn_cast<tt::GatherOp>(op)) {
    Value src = ga.getSrc(), idx = ga.getIndices(), res = ga.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto resTy = cast<RankedTensorType>(res.getType());
    int axis = ga.getAxis();
    msl::Type *scTy = scalarType(elementScalarType(resTy));
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
                                poolRegion(0, sc)));
    body.push_back(ctx.hardBarrier(false));
    for (int r = 0; r < srcRc; ++r)
      body.push_back(ctx.assignStmt(
          ctx.subscript(ctx.var(buf), layout.flatTileOffset(srcTy, r)),
          ctx.var(srcNames[srcNames.size() == 1 ? 0 : r])));
    body.push_back(ctx.hardBarrier(false));

    SmallVector<std::string> outs;
    for (int r = 0; r < resRc; ++r) {
      msl::Expr *off = nullptr;
      int64_t stride = 1;
      for (int d = (int)srcShape.size() - 1; d >= 0; --d) {
        msl::Expr *c = (d == axis)
                           ? static_cast<msl::Expr *>(ctx.cast(
                                 CS::CStyle, ctx.scalar(msl::Scalar::I32),
                                 ctx.paren(ctx.var(
                                     idxNames[idxNames.size() == 1 ? 0 : r]))))
                           : layout.layoutCoordExpr(resTy, r, resOut[d]);
        msl::Expr *term =
            stride == 1 ? c
                        : ctx.paren(ctx.binary(
                              B::Mul, c, ctx.lit(std::to_string(stride))));
        off = off ? ctx.paren(ctx.binary(B::Add, off, term)) : term;
        stride *= srcShape[d];
      }
      if (!off)
        off = ctx.lit("0");
      std::string id = fresh();
      body.push_back(ctx.declStmt(scTy, id, ctx.subscript(ctx.var(buf), off)));
      outs.push_back(id);
    }
    bindRegs(res, outs);
    return true;
  }

  return std::nullopt;
}

// return / call / ub.poison / addptr / load / store.
std::optional<bool> MSLEmitter::emitCallReturn(Operation *op,
                                               msl::Block &body) {
  // tt.return
  if (auto ret = dyn_cast<tt::ReturnOp>(op)) {
    body.push_back(emitReturn(ret));
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
      msl::Type *scTy =
          scalarType(cast<RankedTensorType>(res.getType()).getElementType());
      std::string tmp = fresh();
      body.push_back(ctx.declStmt(deviceRetType(callee), tmp, call));
      int rc = regCount(res);
      SmallVector<std::string> idsV;
      for (int i = 0; i < rc; ++i) {
        std::string id = fresh();
        body.push_back(ctx.declStmt(
            scTy, id, ctx.member(ctx.var(tmp), "f" + std::to_string(i))));
        idsV.push_back(id);
      }
      bindRegs(res, idsV);
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
      body.push_back(
          ctx.declStmt(scalarType(cl.getResult(0).getType()), id, call));
      bindScalar(cl.getResult(0), id);
      return true;
    }
    std::string tmp = fresh();
    body.push_back(ctx.declStmt(deviceRetType(callee), tmp, call));
    for (auto [i, res] : llvm::enumerate(cl.getResults())) {
      std::string id = fresh();
      body.push_back(
          ctx.declStmt(scalarType(res.getType()), id,
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
    msl::Type *sc = isPtr ? storageType(res.getType())
                          : scalarType(elementScalarType(res.getType()));
    std::string scName = isPtr
                             ? mslStorageType(res.getType())
                             : mslScalarType(elementScalarType(res.getType()));
    if (scName.empty())
      return false;
    return declBind(op, sc, body, [&](int) -> msl::Expr * {
      if (isPtr)
        return ctx.lit("nullptr");
      return ctx.cast(CS::CStyle, sc, ctx.lit("0"));
    });
  }

  // addptr: `device sc* id = b + o;`
  if (auto ap = dyn_cast<tt::AddPtrOp>(op)) {
    msl::Type *sc =
        ctx.ptr(scalarType(elementScalarType(op->getResult(0).getType())),
                msl::AddrSpace::Device);
    auto &base = names(ap.getPtr());
    auto &offs = names(ap.getOffset());
    return declBind(op, sc, body, [&](int r) {
      return ctx.binary(B::Add, ctx.var(base[base.size() == 1 ? 0 : r]),
                        ctx.var(offs[offs.size() == 1 ? 0 : r]));
    });
  }

  // load: `sc id = init; [if (m)] id = *p;`
  if (auto ld = dyn_cast<tt::LoadOp>(op)) {
    Value res = ld.getResult();
    msl::Type *sc = scalarType(elementScalarType(res.getType()));
    std::string scName = mslScalarType(elementScalarType(res.getType()));
    auto &ptrs = names(ld.getPtr());
    bool hasMask = ld.getMask() != nullptr;
    SmallVector<std::string> *mask = hasMask ? &names(ld.getMask()) : nullptr;
    SmallVector<std::string> *other =
        ld.getOther() ? &names(ld.getOther()) : nullptr;
    int rc = regCount(res);
    int vw = loadVectorWidth(ld);
    if (vw < 2 || rc % vw != 0)
      vw = 1;

    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r)
      ids.push_back(fresh());
    auto maskOf = [&](int r) {
      return ctx.var((*mask)[mask->size() == 1 ? 0 : r]);
    };
    auto initOf = [&](int r) -> msl::Expr * {
      return other ? static_cast<msl::Expr *>(
                         ctx.var((*other)[other->size() == 1 ? 0 : r]))
                   : static_cast<msl::Expr *>(ctx.lit("0"));
    };
    auto scalarLoad = [&](int r) -> msl::Stmt * {
      msl::Stmt *a = ctx.assignStmt(ctx.var(ids[r]),
                                    derefPtr(ld.getPtr(), ptrs[r], scName));
      return hasMask ? static_cast<msl::Stmt *>(ctx.compactIf(maskOf(r), a))
                     : a;
    };

    for (int r = 0; r < rc; ++r)
      body.push_back(ctx.declStmt(sc, ids[r], initOf(r)));

    // A predicated device load stalls the memory pipe on its predicate, so a
    // run of individually-guarded loads issues serially. Peeling an
    // all-masks-true fast path lets the common (interior-tile) case issue as
    // one unconditional batch; the else arm keeps exact masked semantics.
    // Splitting the peel per vector run would re-serialize the runs against
    // each other, so the guard spans every register of the load and the wide
    // loads sit inside it -- they issue only when all masks hold.
    if (hasMask && rc >= kMaskFastPathMinRegs) {
      SmallVector<msl::Expr *> ms;
      llvm::StringSet<> seen;
      for (int r = 0; r < rc; ++r) {
        StringRef mn = (*mask)[mask->size() == 1 ? 0 : r];
        if (seen.insert(mn).second)
          ms.push_back(ctx.var(mn));
      }
      msl::Block hot, cold;
      for (int base = 0; base < rc; base += vw) {
        if (vw == 1) {
          hot.push_back(ctx.assignStmt(
              ctx.var(ids[base]), derefPtr(ld.getPtr(), ptrs[base], scName)));
          continue;
        }
        auto *scTy = cast<msl::ScalarType>(sc);
        msl::Type *vecTy = ctx.vector(scTy->s, vw);
        msl::Type *vecPtr = ctx.ptr(vecTy, msl::AddrSpace::Device);
        std::string vid = fresh();
        hot.push_back(ctx.declStmt(
            vecTy, vid,
            ctx.deref(ctx.cast(CS::CStyle, vecPtr, ctx.var(ptrs[base])))));
        for (int i = 0; i < vw; ++i)
          hot.push_back(ctx.assignStmt(
              ctx.var(ids[base + i]),
              ctx.subscript(ctx.var(vid), ctx.lit(std::to_string(i)))));
      }
      for (int r = 0; r < rc; ++r)
        cold.push_back(scalarLoad(r));
      body.push_back(ctx.ifElseScope(ctx.chain(B::And, ms), std::move(hot),
                                     std::move(cold)));
      bindRegs(res, ids);
      return true;
    }

    for (int base = 0; base < rc; base += vw) {
      if (vw == 1) {
        body.push_back(scalarLoad(base));
        continue;
      }
      auto *scTy = cast<msl::ScalarType>(sc);
      msl::Type *vecTy = ctx.vector(scTy->s, vw);
      msl::Type *vecPtr = ctx.ptr(vecTy, msl::AddrSpace::Device);
      std::string vid = fresh();
      msl::Block hot;
      hot.push_back(ctx.declStmt(
          vecTy, vid,
          ctx.deref(ctx.cast(CS::CStyle, vecPtr, ctx.var(ptrs[base])))));
      for (int i = 0; i < vw; ++i)
        hot.push_back(ctx.assignStmt(
            ctx.var(ids[base + i]),
            ctx.subscript(ctx.var(vid), ctx.lit(std::to_string(i)))));

      if (!hasMask) {
        for (msl::Stmt *s : hot)
          body.push_back(s);
        continue;
      }
      // The run's lanes are contiguous but their predicates are not provably
      // equal, so take the vector path only when all of them hold.
      SmallVector<msl::Expr *> ms;
      llvm::StringSet<> seenVec;
      for (int i = 0; i < vw; ++i) {
        StringRef mn = (*mask)[mask->size() == 1 ? 0 : base + i];
        if (seenVec.insert(mn).second)
          ms.push_back(ctx.var(mn));
      }
      msl::Block cold;
      for (int i = 0; i < vw; ++i)
        cold.push_back(scalarLoad(base + i));
      body.push_back(ctx.ifElseScope(ctx.chain(B::And, ms), std::move(hot),
                                     std::move(cold)));
    }
    bindRegs(res, ids);
    return true;
  }

  // store: `[if (guard)] *p = v;` (optionally wrapped by a directStore guard).
  if (auto st = dyn_cast<tt::StoreOp>(op)) {
    auto handled = directStoreHandled.find(st.getOperation());
    if (handled != directStoreHandled.end()) {
      msl::Block inner;
      storeBody(st, inner);
      // if (!<fullTile>) { <store body> }
      body.push_back(
          ctx.ifScope(ctx.unary(msl::UnOp::LNot, ctx.var(handled->second)),
                      std::move(inner)));
      return true;
    }
    storeBody(st, body);
    return true;
  }

  return std::nullopt;
}

// scf.if / scf.for (+ fused-GEMM route) / scf.while.
std::optional<bool> MSLEmitter::emitControlFlow(Operation *op,
                                                msl::Block &body) {
  // scf.if: predeclare result vars, then IfScope with then/else sub-blocks.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    SmallVector<SmallVector<std::string>> results;
    for (Value res : ifOp.getResults())
      results.push_back(declResultVars(res, body));

    unsigned d = (unsigned)indent + 1;
    msl::Block thenB = walkBlock(ifOp.getThenRegion().front(), d);
    if (!results.empty())
      for (msl::Stmt *s :
           yieldAssign(ifOp.thenBlock()->getTerminator(), results))
        thenB.push_back(s);
    if (ifOp.getElseRegion().empty()) {
      body.push_back(ctx.ifScope(ctx.var(names(ifOp.getCondition())[0]),
                                 std::move(thenB)));
    } else {
      msl::Block elseB = walkBlock(ifOp.getElseRegion().front(), d);
      if (!results.empty())
        for (msl::Stmt *s :
             yieldAssign(ifOp.elseBlock()->getTerminator(), results))
          elseB.push_back(s);
      body.push_back(ctx.ifElseScope(ctx.var(names(ifOp.getCondition())[0]),
                                     std::move(thenB), std::move(elseB)));
    }
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      bindRegs(res, results[i]);
    return true;
  }

  // scf.for. Fused GEMM K-loops route to emitFusedGemm; i64-IV loops take
  // the wide-IV shape below.
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (auto m = matchGemmDotLoop(forOp))
      return emitFusedGemm(forOp, m->first, m->second, body);
    Type ivType = forOp.getInductionVar().getType();
    bool wideIv = ivType.isInteger(64);

    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init, res] :
         llvm::enumerate(forOp.getInitArgs(), forOp.getResults())) {
      if (isDatalessType(res.getType())) {
        bindDataless(forOp.getRegionIterArg(i));
        bindDataless(res);
        carried.push_back({});
        continue;
      }
      auto &initNames = names(init);
      SmallVector<std::string> vars = declResultVars(res, body);
      for (size_t r = 0; r < vars.size(); ++r)
        body.push_back(
            ctx.assignStmt(ctx.var(vars[r]),
                           ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
      bindRegs(forOp.getRegionIterArg(i), vars);
      bindRegs(res, vars);
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
    msl::Block loopBody = walkBlock(forOp.getRegion().front(), d);
    for (msl::Stmt *s : yieldAssign(forOp.getBody()->getTerminator(), carried))
      loopBody.push_back(s);
    body.push_back(forNode(forOp, std::move(loopBody), iv, tc, ivTy, wideIv));
    return true;
  }

  // scf.while: `while (true) { <before> if (!(c)) { <fwd> break; } <after>
  // <yield> }`
  if (auto wh = dyn_cast<scf::WhileOp>(op)) {
    SmallVector<SmallVector<std::string>> carried;
    for (auto [i, init] : llvm::enumerate(wh.getInits())) {
      auto &initNames = names(init);
      SmallVector<std::string> vars = declResultVars(init, body);
      for (size_t r = 0; r < vars.size(); ++r)
        body.push_back(
            ctx.assignStmt(ctx.var(vars[r]),
                           ctx.var(initNames[initNames.size() == 1 ? 0 : r])));
      bindRegs(wh.getBeforeArguments()[i], vars);
      carried.push_back(vars);
    }
    SmallVector<SmallVector<std::string>> results;
    for (Value res : wh.getResults())
      results.push_back(declResultVars(res, body));

    unsigned d = (unsigned)indent + 1;
    msl::Block loopBody = walkBlock(wh.getBefore().front(), d);
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
        msl::UnOp::LNot, ctx.paren(ctx.var(scalarName(cond.getCondition()))));
    loopBody.push_back(ctx.ifScope(guard, std::move(brk)));

    for (auto [i, fwd] : llvm::enumerate(cond.getArgs()))
      bindAlias(wh.getAfterArguments()[i], fwd);

    for (msl::Stmt *s : walkBlock(wh.getAfter().front(), d))
      loopBody.push_back(s);
    for (msl::Stmt *s :
         yieldAssign(wh.getAfter().front().getTerminator(), carried))
      loopBody.push_back(s);

    body.push_back(ctx.whileScope(nullptr, std::move(loopBody)));
    for (auto [i, res] : llvm::enumerate(wh.getResults()))
      bindRegs(res, results[i]);
    return true;
  }

  return std::nullopt;
}

LogicalResult MSLEmitter::emitSplat(tt::SplatOp op) {
  std::string src = names(op.getSrc())[0];
  int rc = regCount(op.getResult());
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r)
    ids.push_back(src);
  bindRegs(op.getResult(), ids);
  return success();
}

LogicalResult MSLEmitter::emitReshapeLike(Value res, Value src, int axis,
                                          bool isExpand) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto resTy = cast<RankedTensorType>(res.getType());
  auto &srcNames = names(src);
  int srcRc = regCount(src);
  int resRc = regCount(res);

  llvm::DenseMap<uint64_t, int> srcByCoord;
  auto keyOf = [](ArrayRef<int32_t> c) -> uint64_t {
    uint64_t k = 0;
    for (int32_t v : c)
      k = k * 100003u + (uint32_t)v + 1;
    return k;
  };
  for (int r = 0; r < srcRc; ++r)
    srcByCoord[keyOf(layout.registerCoords(srcTy, r))] = r;

  auto srcShape = srcTy.getShape();
  SmallVector<std::string> ids;
  for (int r = 0; r < resRc; ++r) {
    SmallVector<int32_t> rc = layout.registerCoords(resTy, r);
    SmallVector<int32_t> sc;
    if (isExpand) {
      for (int d = 0; d < (int)rc.size(); ++d)
        if (d != axis)
          sc.push_back(rc[d]);
    } else {
      for (int d = 0; d < (int)rc.size(); ++d)
        sc.push_back(srcShape[d] == 1 ? 0 : rc[d]);
    }
    auto it = srcByCoord.find(keyOf(sc));
    if (it == srcByCoord.end()) {
      res.getDefiningOp()->emitError(
          "EmitMSL: reshape register coordinate has no source");
      return failure();
    }
    ids.push_back(srcNames[srcNames.size() == 1 ? 0 : it->second]);
  }
  bindRegs(res, ids);
  return success();
}

LogicalResult MSLEmitter::emitJoin(tt::JoinOp op) {
  Value res = op.getResult();
  auto resTy = cast<RankedTensorType>(res.getType());
  int trailing = resTy.getRank() - 1;
  SmallVector<SmallVector<std::string> *> srcNames = {&names(op.getLhs()),
                                                      &names(op.getRhs())};
  auto srcTy = cast<RankedTensorType>(op.getLhs().getType());
  int srcRc = regCount(op.getLhs());

  llvm::DenseMap<uint64_t, int> srcByCoord;
  for (int r = 0; r < srcRc; ++r)
    srcByCoord[layout.coordKey(layout.registerCoords(srcTy, r))] = r;

  int resRc = regCount(res);
  SmallVector<std::string> ids(resRc);
  for (int r = 0; r < resRc; ++r) {
    SmallVector<int32_t> rc = layout.registerCoords(resTy, r);
    int t = rc[trailing];
    rc.pop_back();
    auto it = srcByCoord.find(layout.coordKey(rc));
    if (it == srcByCoord.end() || t < 0 || t > 1) {
      op.emitError("EmitMSL: join register coordinate has no source");
      return failure();
    }
    auto &sn = *srcNames[t];
    ids[r] = sn[sn.size() == 1 ? 0 : it->second];
  }
  bindRegs(res, ids);
  return success();
}

LogicalResult MSLEmitter::emitSplit(tt::SplitOp op) {
  Value src = op.getOperand();
  auto srcTy = cast<RankedTensorType>(src.getType());
  int trailing = srcTy.getRank() - 1;
  auto &srcNames = names(src);
  int srcRc = regCount(src);

  llvm::DenseMap<uint64_t, int> srcByCoord;
  for (int r = 0; r < srcRc; ++r)
    srcByCoord[layout.coordKey(layout.registerCoords(srcTy, r))] = r;

  for (int k = 0; k < 2; ++k) {
    Value res = op.getResult(k);
    auto resTy = cast<RankedTensorType>(res.getType());
    int resRc = regCount(res);
    SmallVector<std::string> ids(resRc);
    for (int r = 0; r < resRc; ++r) {
      SmallVector<int32_t> rc = layout.registerCoords(resTy, r);
      rc.push_back(k);
      auto it = srcByCoord.find(layout.coordKey(rc));
      if (it == srcByCoord.end()) {
        op.emitError("EmitMSL: split register coordinate has no source");
        return failure();
      }
      (void)trailing;
      ids[r] = srcNames[srcNames.size() == 1 ? 0 : it->second];
    }
    bindRegs(res, ids);
  }
  return success();
}

bool MSLEmitter::isPureBarrierOp(Operation *op) {
  return isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp, ttg::BarrierOp,
             mlir::gpu::BarrierOp>(op);
}

} // namespace mlir::triton::applegpu
