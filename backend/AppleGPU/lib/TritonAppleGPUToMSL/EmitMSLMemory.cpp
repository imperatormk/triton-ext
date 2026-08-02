// EmitMSLMemory.cpp - memory / offset-expression lowering.
//
// AST builders that build the runtime offset/address expressions as
// typed msl::Expr nodes for the printer to emit.
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.

#include "MSLConstants.h"


#include "MSLEmitter.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {
// An initialiser that only computes an address: no subscript, deref or call.
bool isPureAddressExpr(const msl::Expr *e) {
  if (!e)
    return true;
  switch (e->kind) {
  case msl::Expr::Kind::VarRef:
  case msl::Expr::Kind::Literal:
    return true;
  case msl::Expr::Kind::Binary:
    return isPureAddressExpr(llvm::cast<msl::Binary>(e)->lhs) &&
           isPureAddressExpr(llvm::cast<msl::Binary>(e)->rhs);
  case msl::Expr::Kind::Unary:
    return isPureAddressExpr(llvm::cast<msl::Unary>(e)->x);
  case msl::Expr::Kind::Cast:
    return isPureAddressExpr(llvm::cast<msl::Cast>(e)->x);
  case msl::Expr::Kind::Paren:
    return isPureAddressExpr(llvm::cast<msl::Paren>(e)->x);
  default:
    return false;
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// Threadgroup pool region
//===----------------------------------------------------------------------===//

// ((threadgroup sc*)base), base = poolBuf or (poolBuf + off). `sc` may be a
// scalar or an atomic type name, so the pointee is a NamedType.
msl::Expr *MSLEmitter::poolRegion(int64_t byteOffset, StringRef sc) {
  msl::Expr *base =
      byteOffset == 0
          ? static_cast<msl::Expr *>(ctx.var(poolBuf))
          : ctx.paren(ctx.binary(msl::BinOp::Add, ctx.var(poolBuf),
                                 ctx.lit(std::to_string(byteOffset))));
  msl::Type *ptr = ctx.ptr(ctx.named(sc), msl::AddrSpace::Threadgroup);
  return ctx.paren(ctx.cast(msl::Cast::Style::CStyle, ptr, base));
}

//===----------------------------------------------------------------------===//
// Memdesc element address
//===----------------------------------------------------------------------===//

// Row strides for a memdesc, honouring the shared encoding's `order`. The
// fastest-varying dim gets stride 1 and the rest multiply up in order. Returns
// empty for a plain row-major layout, which memdescElemAddr handles directly.
SmallVector<int64_t> MSLEmitter::memdescStrides(ttg::MemDescType mt) {
  auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(mt.getEncoding());
  if (!shared)
    return {};
  auto order = shared.getOrder();
  int rank = mt.getRank();
  if ((int)order.size() != rank)
    return {};
  bool rowMajor = true;
  for (int i = 0; i < rank; ++i)
    if ((int)order[i] != rank - 1 - i)
      rowMajor = false;
  if (rowMajor)
    return {};
  SmallVector<int64_t> strides(rank, 1);
  int64_t acc = 1;
  for (int i = 0; i < rank; ++i) {
    int d = order[i];
    strides[d] = acc;
    acc *= mt.getShape()[d];
  }
  return strides;
}

// True when a barrier already covers the tail of `body`: the last statement is
// a barrier, or everything since one is pure address arithmetic. Declarations
// whose initialiser touches no memory cannot separate two synchronisation
// points, and the pipeliner emits a run of them (the next trip's operand
// pointers) between its wait and the staging copies.
bool MSLEmitter::barrierCoversTail(const msl::Block &body) {
  for (auto it = body.rbegin(); it != body.rend(); ++it) {
    if (isa<msl::BarrierStmt>(*it))
      return true;
    auto *d = dyn_cast<msl::DeclStmt>(*it);
    if (!d || !isPureAddressExpr(d->init))
      return false;
  }
  return false;
}

msl::Expr *MSLEmitter::memdescElemAddr(const MemDescInfo &info,
                                       RankedTensorType tileTy, int reg) {
  msl::Expr *off;
  if (info.bufStrides.empty()) {
    off = layout.flatTileOffset(tileTy, reg);
  } else {
    tt::LinearLayout ll = ttg::toLinearLayout(tileTy);
    auto outNames = llvm::to_vector(ll.getOutDimNames());
    off = nullptr;
    for (int d = 0; d < (int)outNames.size(); ++d) {
      msl::Expr *c = layout.layoutCoordExpr(tileTy, reg, outNames[d]);
      int64_t s = info.bufStrides[d];
      msl::Expr *term = s == 1
                            ? c
                            : ctx.paren(ctx.binary(msl::BinOp::Mul, c,
                                                   ctx.lit(std::to_string(s))));
      off = off ? ctx.paren(ctx.binary(msl::BinOp::Add, off, term)) : term;
    }
    if (!off)
      off = ctx.lit("0");
  }
  if (!info.baseOffset)
    return off;
  return ctx.paren(ctx.binary(msl::BinOp::Add, info.baseOffset, off));
}

int64_t MSLEmitter::poolBudget() const {
  int64_t b = kTGResidentBudgetBytes - liveTgBytes;
  return b < 0 ? 0 : b;
}

int64_t MSLEmitter::reshapeBandElems(int64_t totalElems, int64_t elemBytes,
                                     int64_t budget) {
  int64_t cap = budget / elemBytes;
  if (cap < 1)
    cap = 1;
  int64_t nBands = (totalElems + cap - 1) / cap;
  return (totalElems + nBands - 1) / nBands;
}

int64_t MSLEmitter::tileSize(RankedTensorType rt) {
  int64_t n = 1;
  for (int64_t d : rt.getShape())
    n *= d;
  return n;
}

int64_t MSLEmitter::tgScratchBytes(RankedTensorType ty, bool band2D) {
  Type e = ty.getElementType();
  int64_t elemBytes = byteWidth(e);
  int64_t bytes = tileSize(ty) * elemBytes;
  int64_t cap = poolBudget();
  if (bytes <= cap)
    return bytes;
  int rk = ty.getRank();
  if (band2D && rk >= 2) {
    int64_t N = ty.getShape()[rk - 1];
    int64_t bandRows = cap / (N * elemBytes);
    if (bandRows < 1)
      bandRows = 1;
    return bandRows * N * elemBytes;
  }
  return reshapeBandElems(tileSize(ty), elemBytes, cap) * elemBytes;
}

void MSLEmitter::scanPool(Operation *op) {
  if (auto c = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    if (convertLayoutIsDeadDotStage(c) || convertLayoutIsDeadDotStageSource(c))
      return;
    auto srcT = cast<RankedTensorType>(c.getSrc().getType());
    auto resT = cast<RankedTensorType>(c.getResult().getType());
    // Mirrors the emitter's two escapes: an identical layout rebinds and a
    // lane permutation shuffles, neither of which touches the pool.
    if (ttg::toLinearLayout(srcT) == ttg::toLinearLayout(resT))
      return;
    if (!isa<tt::PointerType>(resT.getElementType()))
      if (auto plan = planIntraWarpShuffle(srcT, resT))
        if (plan->uniformLanePerm && plan->lanePermLinear)
          return;
    auto st = srcT;
    poolBytes = std::max(poolBytes, tgScratchBytes(st, /*band2D=*/true));
  } else if (auto t = dyn_cast<tt::TransOp>(op)) {
    auto rt = cast<RankedTensorType>(t.getResult().getType());
    poolBytes = std::max(poolBytes, tgScratchBytes(rt, /*band2D=*/false));
  } else if (auto c = dyn_cast<tt::CatOp>(op)) {
    auto rt = cast<RankedTensorType>(c.getResult().getType());
    Type e = rt.getElementType();
    poolBytes = std::max(poolBytes, tileSize(rt) * byteWidth(e));
  } else if (auto rs = dyn_cast<tt::ReshapeOp>(op)) {
    auto rt = cast<RankedTensorType>(rs.getResult().getType());
    poolBytes = std::max(poolBytes, tgScratchBytes(rt, /*band2D=*/false));
  } else if (auto g = dyn_cast<tt::GatherOp>(op)) {
    auto st = cast<RankedTensorType>(g.getSrc().getType());
    Type e = st.getElementType();
    poolBytes = std::max(poolBytes, tileSize(st) * byteWidth(e));
  } else if (auto d = dyn_cast<tt::DotOp>(op)) {
    auto aTy = cast<RankedTensorType>(d.getA().getType());
    auto bTy = cast<RankedTensorType>(d.getB().getType());
    auto cTy = cast<RankedTensorType>(d.getResult().getType());
    int rk = cTy.getRank();
    int64_t M = cTy.getShape()[rk - 2];
    int64_t N = cTy.getShape()[rk - 1];
    int64_t Kd = aTy.getShape()[rk - 1];
    int64_t aEb = byteWidth(aTy.getElementType());
    int64_t bEb = byteWidth(bTy.getElementType());
    int64_t aPad, bPad;
    dotStageRowPads(M, N, Kd, aEb, bEb, aPad, bPad);
    int64_t aBy = M * (Kd + aPad) * aEb;
    int64_t bBy = Kd * (N + bPad) * bEb;
    Type cE = cTy.getElementType();
    int64_t need;
    int64_t dmaSlack = 0;
    if (isa<IntegerType>(cE)) {
      need = aBy + bBy;
    } else {
      int64_t accBytes = 4;
      int64_t elemBytes = byteWidth(aTy.getElementType());
      int64_t stagedA = aBy, stagedB = bBy;
      if (rk == 2 && aBy + bBy <= kTGResidentBudgetBytes) {
        if (dotOperandLocalLoad(d.getA(), M, Kd))
          stagedA = 0;
        if (dotOperandLocalLoad(d.getB(), Kd, N))
          stagedB = 0;
      }
      // Mirrors planDot: a device-direct A claims no pool. Both sides ask the
      // same phase-free predicate, so the reservation cannot drift from the
      // emission -- a mismatch here overwrites the staging underneath.
      if (stagedA && dotIsFusedGemmAcc(d)) {
        tt::LinearLayout ll = ttg::toLinearLayout(cTy);
        auto wd = StringAttr::get(d.getContext(), "warp");
        int64_t nw = ll.hasInDim(wd) ? ll.getInDimSize(wd) : 1;
        int64_t nf = (M / 8) * (N / 8);
        if (nw > nf)
          nw = nf;
        if (aDirectCandidate(d, M, Kd, nw, /*requireBound=*/false)) {
          stagedA = 0;
          aBy = 0;
        }
      }
      // Device-direct B staging keeps a second B tile so the copy feeding the
      // next K-trip can be in flight while this trip's MMAs read the other one.
      // Held aside from stagedAB so it does not disturb the panel/C decisions
      // below, and folded back into `need` once those have been made.
      // scanPool runs before any fused phase is set, so ask the phase-free
      // candidate test rather than dotDmaStage (which would see phase None and
      // under-reserve the second tile).
      // Mirrors planDot exactly: an unpadded destination pitch for the copy,
      // and the second tile only when it does not cost a residency step.
      if (stagedB && dmaStagingEnabled() && bDmaCandidate(d, false)) {
        int64_t bNoPad = Kd * N * bEb;
        int64_t ab = stagedA + bNoPad;
        if (ab + bNoPad <= kTGResidentBudgetBytes &&
            tgResidency(ab + bNoPad) >= tgResidency(ab)) {
          stagedB = bNoPad;
          bBy = bNoPad;
          dmaSlack = bNoPad;
        }
      }
      int64_t stagedAB = stagedA + stagedB;
      int64_t cFull = M * N * accBytes;
      int64_t cBudget = poolBudget() - dmaSlack;
      int64_t cBanded = std::min(cFull, cBudget);
      if (stagedAB == aBy + bBy &&
          dotNeedsPanel(M, N, Kd, elemBytes, accBytes)) {
        int64_t mp, np;
        dotPanelDims(M, N, Kd, elemBytes, accBytes, mp, np);
        need = mp * Kd * elemBytes + Kd * np * elemBytes + mp * np * accBytes;
      } else if (dotIsFusedGemmAcc(d)) {
        // Fused K-loop: C lands only in the post-loop epilogue, after a
        // barrier, so it overlays the dead A/B staging. When the epilogue's
        // relayout is a lane permutation the C tile never reaches threadgroup
        // memory at all; otherwise the readback stages it one band at a time,
        // so reserving the full tile would cap the block size for nothing.
        // A boundary-masked store keeps a threadgroup fallback arm for the
        // ragged tile, so C still needs the pool even when the direct path
        // exists; only an unmasked store bypasses threadgroup memory entirely.
        if (rk == 2 && fusedGemmCIsShuffled(d) && !fusedGemmCHasFallback(d)) {
          need = stagedAB;
        } else if (stagedAB == 0) {
          // Operands live in their own local_alloc buffers (the pipelined DMA
          // path), so stagedAB is 0 and there is nothing for C to overlay in
          // the pool. Reserving the whole C tile on top of those buffers is
          // what asked for 45-56KB against a 32KB cap and had the autotuner
          // drop the larger blocks; band it to what actually fits beside them.
          need = cBanded;
        } else {
          need = std::max(stagedAB, dmaSlack ? cBanded : cFull);
        }
        // With both operands read in place (pipelined into their own
        // allocations) stagedAB is zero. The readback still declares a
        // threadgroup C pointer whenever it emits a fallback arm, so the pool
        // has to cover that; an unmasked store has no such arm.
        if (need == 0 && fusedGemmCHasFallback(d))
          need = cBanded;
      } else if (stagedAB + cFull <= poolBudget()) {
        need = stagedAB + cFull;
      } else if (rk == 2 && dotIsFusedGemmAcc(d) && cStoresDirect(d)) {
        // C stores straight to device and never reaches the pool, so a band
        // reservation is pure waste. With both operands on DMA stagedAB is 0,
        // and that 2KB band was the whole reason BLOCK_K=64 did not fit.
        // fusedGemmCIsShuffled is stricter than needed here -- it also demands
        // the epilogue relayout be a lane permutation, which fails at larger
        // BLOCK_K even though the store itself is still direct.
        need = stagedAB;
      } else {
        int64_t band = dotCBandRows(M, N, cBudget, accBytes);
        need = std::max(stagedAB, band * N * accBytes);
      }
    }
    // A readback that still emits its threadgroup C pointer needs a buffer to
    // name; a provably-direct store has no such arm and needs nothing.
    if (need == 0 && rk == 2 && dotIsFusedGemmAcc(d) && !isa<IntegerType>(cE) &&
        !cStoresDirect(d))
      need = N * 4;
    poolBytes = std::max(poolBytes, need + dmaSlack);
  } else if (auto r = dyn_cast<tt::ReduceOp>(op)) {
    auto st = cast<RankedTensorType>(r.getOperand(0).getType());
    tt::LinearLayout ll = ttg::toLinearLayout(st);
    auto kWarp = StringAttr::get(op->getContext(), "warp");
    if (ll.hasInDim(kWarp)) {
      int64_t nw = ll.getInDimSize(kWarp);
      int64_t bytes = 0;
      for (Value res : r.getResult())
        bytes +=
            nw * 32 *
            std::max<int64_t>(1, byteWidth(elementScalarType(res.getType())));
      poolBytes = std::max(poolBytes, bytes);
    }
  } else if (auto h = dyn_cast<tt::HistogramOp>(op)) {
    auto rt = cast<RankedTensorType>(h.getResult().getType());
    poolBytes = std::max(poolBytes, tileSize(rt) * 4);
  } else if (auto ca = dyn_cast<tt::AtomicCASOp>(op)) {
    if (!isa<RankedTensorType>(ca.getPtr().getType()))
      poolBytes = std::max<int64_t>(poolBytes, 8);
  } else if (auto ar = dyn_cast<tt::AtomicRMWOp>(op)) {
    if (auto ptrTy = dyn_cast<RankedTensorType>(ar.getPtr().getType())) {
      tt::LinearLayout ll = ttg::toLinearLayout(ptrTy);
      MLIRContext *c = op->getContext();
      unsigned warpFree =
          ll.getFreeVariableMasks().lookup(StringAttr::get(c, "warp"));
      if (warpFree) {
        int64_t eb = std::max<int64_t>(
            1, byteWidth(elementScalarType(ar.getResult().getType())));
        int64_t rc = ll.getInDimSize(StringAttr::get(c, "register"));
        int64_t nw = ll.hasInDim(StringAttr::get(c, "warp"))
                         ? ll.getInDimSize(StringAttr::get(c, "warp"))
                         : 1;
        poolBytes = std::max<int64_t>(poolBytes, rc * 32 * nw * eb);
      }
    }
  } else if (auto s = dyn_cast<tt::ScanOp>(op)) {
    auto st = cast<RankedTensorType>(s.getOperand(0).getType());
    tt::LinearLayout ll = ttg::toLinearLayout(st);
    auto kWarp = StringAttr::get(op->getContext(), "warp");
    auto outDims = llvm::to_vector(ll.getOutDimNames());
    auto outDim = outDims[s.getAxis()];
    if (!axisBits(ll, kWarp, outDim).empty()) {
      int64_t nw = ll.getInDimSize(kWarp);
      int64_t bytes = 0;
      for (Value res : s.getResult())
        bytes += nw * 32 * byteWidth(elementScalarType(res.getType()));
      poolBytes = std::max(poolBytes, bytes);
    }
  }
  for (Region &reg : op->getRegions())
    for (Block &blk : reg)
      for (Operation &o : blk)
        scanPool(&o);
}

int64_t MSLEmitter::memdescFlatSize(ttg::MemDescType mt) {
  int64_t n = 1;
  for (int64_t d : mt.getShape())
    n *= d;
  return n;
}

LogicalResult MSLEmitter::emitMemDescIndex(ttg::MemDescIndexOp op) {
  auto srcMt = cast<ttg::MemDescType>(op.getSrc().getType());
  auto resMt = cast<ttg::MemDescType>(op.getResult().getType());
  MemDescInfo parent = memdescMap[op.getSrc()];
  int64_t sliceSize = memdescFlatSize(resMt);
  (void)srcMt;
  const std::string &idx = names(op.getIndex())[0];
  msl::Expr *scaled = ctx.binary(msl::BinOp::Mul, ctx.var(idx),
                                 ctx.lit(std::to_string(sliceSize)));
  msl::Expr *base =
      parent.baseOffset
          ? ctx.paren(ctx.binary(msl::BinOp::Add, parent.baseOffset, scaled))
          : ctx.paren(scaled);
  memdescMap[op.getResult()] = {parent.buf, base, parent.bufStrides};
  return success();
}

LogicalResult MSLEmitter::emitMemDescSubslice(ttg::MemDescSubsliceOp op) {
  auto srcMt = cast<ttg::MemDescType>(op.getSrc().getType());
  MemDescInfo parent = memdescMap[op.getSrc()];
  ArrayRef<int64_t> srcShape = srcMt.getShape();
  ArrayRef<int32_t> offsets = op.getOffsets();
  if (offsets.size() != srcShape.size())
    return op.emitError("EmitMSL: memdesc_subslice rank mismatch");

  SmallVector<int64_t> strides(srcShape.size());
  if (!parent.bufStrides.empty()) {
    strides.assign(parent.bufStrides.begin(), parent.bufStrides.end());
  } else {
    int64_t s = 1;
    for (int d = (int)srcShape.size() - 1; d >= 0; --d) {
      strides[d] = s;
      s *= srcShape[d];
    }
  }

  int64_t constOff = 0;
  for (int d = 0; d < (int)offsets.size(); ++d)
    constOff += (int64_t)offsets[d] * strides[d];

  msl::Expr *base;
  if (!parent.baseOffset)
    base = ctx.lit(std::to_string(constOff));
  else
    base = constOff == 0
               ? parent.baseOffset
               : ctx.paren(ctx.binary(msl::BinOp::Add, parent.baseOffset,
                                      ctx.lit(std::to_string(constOff))));

  memdescMap[op.getResult()] = {parent.buf, base, strides};
  return success();
}

} // namespace mlir::triton::applegpu
