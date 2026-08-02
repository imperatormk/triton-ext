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

#include <functional>

using namespace mlir;

namespace mlir::triton::applegpu {

namespace {
using B = msl::BinOp;
using CS = msl::Cast::Style;

template <typename T> T definingOp(Value v) {
  return dyn_cast_or_null<T>(v.getDefiningOp());
}
} // namespace

//===----------------------------------------------------------------------===//
// Device-direct operand staging (threadgroup DMA)
//===----------------------------------------------------------------------===//
//
// A dot operand normally reaches its threadgroup tile through registers: the
// tt.load materialises one register per element, then stageOperand scatters
// those registers into the pool. air.simdgroup_async_copy_2d can instead move
// the tile device->threadgroup directly, but only when the operand's pointer
// tensor really denotes a contiguous 2D tile: a uniform scalar base, a scalar
// row stride, and a unit-stride column range.
//
// The recogniser below matches exactly the shape Triton emits for
// `base + row[:,None]*stride + col[None,:]`:
//
//   addptr(broadcast(addptr(splat(%base), mul(rowIdx, splat(%stride)))),
//          broadcast(colRange))
//
// The destination needs no adjustment: stageOperand already writes a plain
// padded row-major tile (the XOR trees in its index are disjoint-bit
// composition, not a swizzle), so the DMA writes the identical layout and the
// consumer simdgroup_loads keep their existing offsets.

namespace {


// Peel the broadcast/expand_dims wrappers a pointer tensor accumulates.
Value peelBroadcast(Value v) {
  while (true) {
    if (auto b = definingOp<tt::BroadcastOp>(v)) {
      v = b.getSrc();
      continue;
    }
    if (auto e = definingOp<tt::ExpandDimsOp>(v)) {
      v = e.getSrc();
      continue;
    }
    // A layout convert on an index tensor relabels which lane holds which
    // element; the values are unchanged, so it cannot affect whether the
    // offset is a unit range or a uniform splat.
    if (auto c = definingOp<ttg::ConvertLayoutOp>(v)) {
      v = c.getSrc();
      continue;
    }
    return v;
  }
}

// True when a tensor value is uniform across the tile: `tt.splat` of a scalar,
// or a constant whose elements are all equal (`arith.constant dense<32>`, which
// is how a literal K-step advance is spelled).
bool isUniformTensor(Value v) {
  Value p = peelBroadcast(v);
  if (definingOp<tt::SplatOp>(p))
    return true;
  if (auto cst = definingOp<arith::ConstantOp>(p))
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      return dense.isSplat();
  return false;
}


// True when `v` indexes a unit-stride row: a 0..n-1 iota, optionally shifted by
// a uniform scalar (`splat(k) + iota`, as a tile's column offset carries the
// block's N-origin). Any such shift is uniform across the tile, so it moves the
// tile origin rather than breaking contiguity; it is returned in `shiftOut` to
// be folded into the base pointer.
bool isUnitRange(Value v, Value *shiftOut = nullptr,
                 int64_t tileRows = 0) {
  Value s = peelBroadcast(v);
  // `rm % M` keeps a ragged trailing tile in bounds by wrapping it. That wrap
  // is only harmless when it can never fall inside a tile: with a run-time
  // bound, or an extent the block does not divide, the last tile is split --
  // for N=30522/BLOCK_N=32 it runs 30496..30521 then 1..5, which is not a
  // rectangle in device memory and no 2D strided copy can express it. Only a
  // constant extent that the tile size divides is safe, and then the remainder
  // is an identity over every tile and can be peeled outright.
  // tt.contiguity is NOT sufficient here: it describes the typical tile, not
  // the trailing one.
  if (auto rem = definingOp<arith::RemSIOp>(s)) {
    if (!tileRows)
      return false;
    auto cst = definingOp<arith::ConstantOp>(peelBroadcast(rem.getRhs()));
    if (!cst)
      return false;
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return false;
    int64_t extent = dense.getSplatValue<APInt>().getSExtValue();
    if (extent <= 0 || extent % tileRows != 0)
      return false;
    return isUnitRange(rem.getLhs(), shiftOut, tileRows);
  }
  if (auto add = definingOp<arith::AddIOp>(s)) {
    Value a = add.getLhs(), b = add.getRhs();
    for (int i = 0; i < 2; ++i) {
      if (auto sp = definingOp<tt::SplatOp>(peelBroadcast(a)))
        if (isUnitRange(b, nullptr, tileRows)) {
          if (shiftOut)
            *shiftOut = sp.getSrc();
          return true;
        }
      std::swap(a, b);
    }
    return false;
  }
  auto r = definingOp<tt::MakeRangeOp>(s);
  return r && r.getStart() == 0;
}

// Match `mul(rowIndex, splat(stride))` (either operand order) and return the
// scalar stride. The row index may itself carry a uniform block-origin shift,
// which is reported in `shiftOut` (in rows, to be scaled by the stride).
// The stride is a splat of a kernel argument, or -- when the template's shapes
// are static, as inductor's are -- a folded `arith.constant dense<n>`. A match
// reports whichever form it found; `strideLit` is set only for the latter.
Value matchRowStride(Value v, Value *shiftOut = nullptr, int64_t tileRows = 0,
                     std::optional<int64_t> *strideLit = nullptr,
                     bool *matched = nullptr) {
  auto mul = definingOp<arith::MulIOp>(peelBroadcast(v));
  if (!mul)
    return nullptr;
  Value a = mul.getLhs(), b = mul.getRhs();
  for (int i = 0; i < 2; ++i) {
    if (isUnitRange(a, shiftOut, tileRows)) {
      Value pb = peelBroadcast(b);
      if (auto sp = definingOp<tt::SplatOp>(pb)) {
        if (matched)
          *matched = true;
        return sp.getSrc();
      }
      if (auto cst = definingOp<arith::ConstantOp>(pb))
        if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
          if (dense.isSplat() && strideLit) {
            *strideLit = dense.getSplatValue<APInt>().getSExtValue();
            if (matched)
              *matched = true;
            return nullptr;
          }
    }
    std::swap(a, b);
  }
  return nullptr;
}

} // namespace

// A GEMM's operand pointer is loop-carried: the tile pointer enters as an
// scf.for iter_arg and is advanced each trip by `addptr(arg, splat(delta))`.
// The tile stays contiguous with the same row stride throughout -- only its
// origin moves, by a uniform scalar -- so resolve the block argument to the
// loop's *initial* pointer and report the per-trip delta separately.
// Returns the init value, or null when the recurrence is not this shape.
static Value resolveLoopCarriedPtr(Value v, Value &deltaOut,
                                   std::optional<int64_t> &deltaLit,
                                   int *peeledSteps = nullptr) {
  // The pipeliner copies from the *advanced* pointer, `addptr(arg, splat(d))`,
  // one step beyond the iter-arg. Peel that so the recurrence below sees the
  // block argument; the extra step only moves the tile origin, uniformly.
  if (auto ap = definingOp<tt::AddPtrOp>(v))
    if (isa<BlockArgument>(ap.getPtr()))
      if (isUniformTensor(ap.getOffset())) {
        v = ap.getPtr();
        // The copy reads one step *ahead* of the iter-arg (it prefetches the
        // next trip). Dropping the step here would make trip 0 re-fetch the
        // tile the peeled prologue copy already staged.
        if (peeledSteps)
          ++*peeledSteps;
      }
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg)
    return v;
  auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!forOp || arg.getArgNumber() == 0)
    return nullptr;
  unsigned idx = arg.getArgNumber() - 1;
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  auto ap = definingOp<tt::AddPtrOp>(yield.getOperand(idx));
  // The recurrence must be exactly `next = addptr(thisArg, splat(scalar))`.
  if (!ap || ap.getPtr() != v)
    return nullptr;
  // Record the advance itself rather than assuming it equals the row stride:
  // B walks K down its rows (delta == a scalar stride), A walks K along its
  // columns (delta == a constant K-block).
  Value off = peelBroadcast(ap.getOffset());
  if (auto sp = definingOp<tt::SplatOp>(off)) {
    deltaOut = sp.getSrc();
  } else if (auto cst = definingOp<arith::ConstantOp>(off)) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return nullptr;
    deltaLit = dense.getSplatValue<APInt>().getSExtValue();
  } else {
    return nullptr;
  }
  return forOp.getInitArgs()[idx];
}

// Recognise a dot operand whose load is a contiguous device tile.
// `rows`/`cols` are the operand's logical tile shape.
std::optional<DirectStage> matchTilePointer(Value ptr, int64_t rows,
                                            int64_t cols);

std::optional<DirectStage> matchDirectStage(Value operand, int64_t rows,
                                            int64_t cols) {
  bool dbg = getenv("TRITON_MSL_DMA_PROBE") != nullptr;
  auto bail = [&](const char *why) {
    if (dbg)
      llvm::errs() << "[dma-probe]   bail: " << why << "\n";
    return std::nullopt;
  };

  auto ld = definingOp<tt::LoadOp>(operand);
  // A masked load would need the DMA to honour per-element predicates, which
  // the intrinsic cannot express; boundary tiles keep the register path.
  if (!ld)
    return bail("operand not defined by a tt.load");
  if (ld.getMask() || ld.getOther())
    return bail("masked load (boundary tile keeps the register path)");

  return matchTilePointer(ld.getPtr(), rows, cols);
}

// The pointer half of matchDirectStage: given the *pointer tensor* of a load or
// an async copy, recover the uniform base, row stride and block-origin shifts
// that air.simdgroup_async_copy_2d needs.
std::optional<DirectStage> matchTilePointer(Value ptr, int64_t rows,
                                            int64_t cols) {
  bool dbg = getenv("TRITON_MSL_DMA_PROBE") != nullptr;
  auto bail = [&](const char *why) {
    if (dbg)
      llvm::errs() << "[dma-probe]   bail: " << why << "\n";
    return std::nullopt;
  };

  Value delta;
  std::optional<int64_t> deltaLit;
  int peeled = 0;
  Value tilePtr = resolveLoopCarriedPtr(ptr, delta, deltaLit, &peeled);
  if (!tilePtr)
    return bail("loop-carried ptr not a uniform addptr recurrence");

  auto colAdd = definingOp<tt::AddPtrOp>(tilePtr);
  if (!colAdd)
    return bail("tile ptr not an addptr");

  // The two axis offsets reach the addptr either as a chain of two addptrs or,
  // as inductor's mm template spells it, already summed into one offset tensor.
  Value axis0, axis1;
  Value base;
  if (auto inner = definingOp<tt::AddPtrOp>(peelBroadcast(colAdd.getPtr()))) {
    axis0 = colAdd.getOffset();
    axis1 = inner.getOffset();
    base = inner.getPtr();
  } else if (auto sum =
                 definingOp<arith::AddIOp>(peelBroadcast(colAdd.getOffset()))) {
    axis0 = sum.getLhs();
    axis1 = sum.getRhs();
    base = colAdd.getPtr();
  } else {
    return bail("tile offset is neither a chained addptr nor a sum of axes");
  }

  auto splat = definingOp<tt::SplatOp>(peelBroadcast(base));
  if (!splat)
    return bail("base not a splat of a scalar pointer");

  // One axis offset walks contiguously (a unit range), the other is
  // iota*stride. Which *tensor* axis each varies along decides the orientation:
  // `tt.expand_dims axis=d` is constant along d, so it varies along 1-d. The
  // tile is row-major when the contiguous offset varies along the column axis
  // (dim 1), and column-major -- inductor hands B in with strides [1, N] --
  // when it varies along the row axis instead.
  std::function<bool(Value)> variesAlongCols = [&](Value v) -> bool {
    Value s = v;
    while (true) {
      if (auto b = definingOp<tt::BroadcastOp>(s)) {
        s = b.getSrc();
        continue;
      }
      if (auto c = definingOp<ttg::ConvertLayoutOp>(s)) {
        s = c.getSrc();
        continue;
      }
      break;
    }
    if (auto e = definingOp<tt::ExpandDimsOp>(s))
      return e.getAxis() == 0;
    if (auto mul = definingOp<arith::MulIOp>(s))
      return variesAlongCols(mul.getLhs()) || variesAlongCols(mul.getRhs());
    if (auto add = definingOp<arith::AddIOp>(s))
      return variesAlongCols(add.getLhs()) || variesAlongCols(add.getRhs());
    return false;
  };

  // Sort the two offsets by the tensor axis each varies along, so `rowAxis`
  // walks down the tile and `colAxis` across it regardless of the order the
  // template happened to sum them in.
  Value rowAxis = axis0, colAxis = axis1;
  if (variesAlongCols(axis0))
    std::swap(rowAxis, colAxis);
  if (variesAlongCols(rowAxis) || !variesAlongCols(colAxis))
    return bail("tile offsets do not vary along distinct axes");

  // A contiguous axis is spelled either as a bare iota or as iota*splat(s)
  // where s is 1 at runtime. The latter is inductor's form -- it multiplies
  // both axes by a stride even when one of them is unit -- so the contiguity
  // of a tile is not decidable here and the *pair* of strides is carried
  // instead; the copy picks the contiguous one.
  // A wrap on an axis is only safe when the tile size along *that* axis
  // divides the operand extent, so each axis is checked against its own dim.
  Value rowShift, colShift;
  std::optional<int64_t> rowStrideLit, colStrideLit;
  bool rowMatched = false, colMatched = false;
  Value rowStride =
      matchRowStride(rowAxis, &rowShift, rows, &rowStrideLit, &rowMatched);
  Value colStride =
      matchRowStride(colAxis, &colShift, cols, &colStrideLit, &colMatched);
  if (!rowMatched && !isUnitRange(rowAxis, &rowShift, rows))
    return bail("row offset neither iota nor iota*splat(stride)");
  if (!colMatched && !isUnitRange(colAxis, &colShift, cols))
    return bail("column offset neither iota nor iota*splat(stride)");

  // Row-major means the column axis is the contiguous one -- i.e. it carries no
  // stride of its own, or a stride of exactly 1. Anything else means the tile
  // walks down its columns and the row stride is the contiguous direction.
  bool colIsUnit = !colStride && (!colStrideLit || *colStrideLit == 1);
  bool transposed = !colIsUnit;
  Value stride = transposed ? colStride : rowStride;
  std::optional<int64_t> strideLit = transposed ? colStrideLit : rowStrideLit;
  if (!stride && !strideLit)
    return bail("no strided axis to describe the tile pitch");

  DirectStage out;
  out.basePtr = splat.getSrc();
  out.rowStride = stride;
  out.rowShift = rowShift;
  out.colShift = colShift;
  out.ptrDelta = delta;
  out.rowStrideLit = strideLit;
  out.ptrDeltaLit = deltaLit;
  out.aheadSteps = peeled;
  out.rows = rows;
  out.cols = cols;
  out.srcTransposed = transposed;
  return out;
}

// The DMA form of an async copy: the pointer tensor must denote a contiguous
// device tile whose destination is a plain unpadded row-major threadgroup
// buffer, which is exactly what the pipeliner allocates.

// Shared by the copy itself and the sibling walk, so the two cannot disagree.
bool MSLEmitter::dmaCopyEligible(ttg::AsyncCopyGlobalToLocalOp ac) {
  // The intrinsic takes no predicate, so a per-element mask would read past a
  // ragged operand; a uniform one just guards the whole call.
  if (Value m = ac.getMask())
    if (!definingOp<tt::SplatOp>(peelBroadcast(m)))
      return false;
  auto srcTy = dyn_cast<RankedTensorType>(ac.getSrc().getType());
  auto mt = dyn_cast<ttg::MemDescType>(ac.getResult().getType());
  if (!srcTy || !mt || srcTy.getRank() != 2 || mt.getRank() != 2)
    return false;
  Type e = mt.getElementType();
  if (!(e.isF32() || e.isF16() || e.isBF16()))
    return false;
  // A non-row-major destination cannot be expressed as (dstStride, dims).
  if (!memdescStrides(mt).empty())
    return false;
  // Asynchronous, so one buffer lets trip N+1's request overwrite the tile trip
  // N is still reading -- the register path survives that only by being
  // synchronous and barrier-ordered.
  Value buf = ac.getResult();
  while (auto idx = buf.getDefiningOp<ttg::MemDescIndexOp>())
    buf = idx.getSrc();
  auto bt = dyn_cast<ttg::MemDescType>(buf.getType());
  if (!bt || bt.getRank() < 3 || bt.getShape()[0] < 2)
    return false;
  int64_t rows = mt.getShape()[0], cols = mt.getShape()[1];
  return matchTilePointer(ac.getSrc(), rows, cols).has_value();
}

std::optional<DirectStage> MSLEmitter::asyncCopyDma(ttg::AsyncCopyGlobalToLocalOp ac) {
  if (!dmaStagingEnabled() || !dmaCopyEligible(ac))
    return std::nullopt;
  auto mt = cast<ttg::MemDescType>(ac.getResult().getType());
  auto ds = matchTilePointer(ac.getSrc(), mt.getShape()[0], mt.getShape()[1]);
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
  if (!bound(ds->basePtr) || !(ds->rowStrideLit || bound(ds->rowStride)) ||
      !bound(ds->rowShift) || !bound(ds->colShift))
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
      (!bound(ds->basePtr) || !(ds->rowStrideLit || bound(ds->rowStride)) || !bound(ds->rowShift) ||
       !bound(ds->colShift)))
    return std::nullopt;
  return ds;
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
// (iv / step) * ptrDelta. Empty when the dot is not in a loop, in which case the
// tile origin has no per-trip term.
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
  auto add = [&](msl::Expr *e) {
    off = off ? ctx.binary(B::Add, off, e) : e;
  };
  if (ds.rowShift)
    add(ctx.paren(ctx.binary(B::Mul, ctx.var(scalarName(ds.rowShift)),
                             dmaRowStride(ds))));
  if (ds.colShift)
    add(ctx.var(scalarName(ds.colShift)));
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
          t = ctx.paren(ctx.binary(
              B::Add, t, ctx.i32lit(ds.aheadSteps * ds.cols)));
        if (perK != 1)
          t = ctx.paren(ctx.binary(B::Mul, t, ctx.i32lit(perK)));
        add(ctx.paren(t));
      }
    } else if (ds.ptrDelta) {
      msl::Expr *t = ctx.var(tripVar);
      if (ds.aheadSteps)
        t = ctx.paren(
            ctx.binary(B::Add, t, ctx.i32lit(ds.aheadSteps * ds.rows)));
      add(ctx.paren(
          ctx.binary(B::Mul, t, dmaRowStride(ds))));
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
  msl::Expr *c = ctx.call(
      callee, {ctx.var(tgBuf), ctx.i32lit(pitch), src,
               dmaRowStride(ds), ctx.i32lit(ds.rows),
               ctx.i32lit(ds.cols)});
  return ctx.declStmt(ctx.named("ulong"), handle, c);
}

msl::Stmt *MSLEmitter::dmaWait(StringRef handle) {
  return ctx.exprStmt(ctx.call("__triton_tg_async_copy_wait",
                               {ctx.var(handle)}));
}

// `h = begin(tgBbase + parity*stagedB, ...)` for the trip after the current
// one. The destination is the tile the current MMAs are not reading, selected
// by the parity flag the caller has already flipped.
msl::Stmt *MSLEmitter::dmaBeginInto(StringRef handle, const DotPlan &plan,
                                    const DotEmitCtx &dc,
                                    const DirectStage &ds,
                                    RankedTensorType bStageTy, int64_t ldb,
                                    StringRef tripVar, bool nextTrip) {
  int64_t eb = byteWidth(bStageTy.getElementType());
  // The parity names the tile this trip READS. The priming copy fills that
  // tile; the in-loop copy fills the other one, for the next trip to read
  // after the top-of-trip flip.
  msl::Expr *slot =
      nextTrip ? ctx.paren(ctx.binary(B::Sub, ctx.i32lit(1),
                                      ctx.var(fusedDot.dmaParity)))
               : static_cast<msl::Expr *>(ctx.var(fusedDot.dmaParity));
  // air.simdgroup_async_copy_2d is issued per simdgroup, so letting every warp
  // request the whole tile would move it numWarps times over. Split it instead:
  // warp w takes the band [w*band, (w+1)*band).
  //
  // `_tr` swaps the source strides, so it also swaps which axis the device
  // walks contiguously. Banding the contiguous axis chops every warp's run into
  // a scalar strided gather -- 4.15x at 32x32 -- so a transposed source must be
  // banded on the opposite axis.
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
    src = ctx.binary(
        B::Add, src,
        ctx.paren(ctx.binary(B::Mul, ctx.i32lit(ds.rows),
                             dmaRowStride(ds))));
  if (split) {
    msl::Expr *wOff =
        ctx.paren(ctx.binary(B::Mul, ctx.var(warpId), ctx.i32lit(band)));
    dst = ctx.paren(ctx.binary(
        B::Add, dst,
        bandCols ? wOff
                 : ctx.paren(ctx.binary(B::Mul, wOff, ctx.i32lit(ldb)))));
    src = ctx.binary(
        B::Add, src, ctx.paren(ctx.binary(B::Mul, wOff, dmaRowStride(ds))));
  }
  msl::Expr *c = ctx.call(
      dmaCallee(eb, ds.srcTransposed),
      {dst, ctx.i32lit(ldb), src, dmaRowStride(ds),
       ctx.i32lit(bandCols ? ds.rows : band),
       ctx.i32lit(bandCols ? band : ds.cols)});
  return ctx.assignStmt(ctx.var(handle), c);
}

namespace {

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
  int64_t miCount = 1, niCount = 1; // subblock shape owned by one warp
  int64_t wGridN = 1;               // warps across the N axis
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
//
// The resulting sgload/MMA ratio is (mi+ni)/fragsPerWarp, minimised at the
// near-square split: 0.75 at fragsPerWarp=8, 0.50 at 16, 0.375 at 32. A kernel
// sitting at 0.75 is therefore at its geometric optimum, not on a fallback --
// only a larger per-warp accumulator tile lowers it. Verified 2026-08-01 across
// the 12-kernel inductor corpus: every config takes the 2D path.
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
  int n = regCount(stage);
  // Consecutive registers whose threadgroup slots are also consecutive can be
  // published as one vector store instead of vw scalar stores that each rebuild
  // the whole swizzle index. A column-major operand lays its registers down a
  // column (slots rowLen apart), so the run has to be checked, not assumed.
  int vw = stageVectorWidth(stageTy, n, rowPad);
  // A guarded run can still be widened, but only under one predicate, so every
  // register in it must carry the same one.
  auto render = [&](msl::Expr *e) {
    std::string s;
    llvm::raw_string_ostream os(s);
    msl::MSLPrinter(os).printExpr(e);
    return os.str();
  };
  if (guard && vw > 1) {
    for (int r = 0; r + vw <= n && vw > 1; r += vw) {
      std::string g0 = render(guard(r));
      for (int k = 1; k < vw; ++k)
        if (render(guard(r + k)) != g0) {
          vw = 1;
          break;
        }
    }
  }
  Type elem = elementScalarType(stageTy);
  for (int r = 0; r < n; r += (vw > 1 ? vw : 1)) {
    if (vw > 1) {
      msl::Type *vecTy =
          ctx.vector(cast<msl::ScalarType>(scalarType(elem))->s, vw);
      msl::Type *tgVecPtr = ctx.ptr(vecTy, msl::AddrSpace::Threadgroup);
      SmallVector<msl::Expr *> elems;
      for (int k = 0; k < vw; ++k)
        elems.push_back(ctx.var(names[r + k]));
      msl::Stmt *vasn = ctx.assignStmt(
          ctx.deref(ctx.paren(ctx.cast(
              msl::Cast::Style::CStyle, tgVecPtr,
              ctx.paren(ctx.binary(
                  msl::BinOp::Add, ctx.var(tgName),
                  layout.sliceFlatOffset(stageTy, r, rowPad)))))),
          ctx.call(mslScalarType(elem) + std::to_string(vw), elems));
      msl::Expr *vg = guard ? guard(r) : nullptr;
      body.push_back(vg ? ctx.compactIf(vg, vasn) : vasn);
      continue;
    }
    msl::Stmt *asn = ctx.assignStmt(
        ctx.subscript(ctx.var(tgName),
                      layout.sliceFlatOffset(stageTy, r, rowPad)),
        ctx.var(names[r]));
    msl::Expr *g = guard ? guard(r) : nullptr;
    body.push_back(g ? ctx.compactIf(g, asn) : asn);
  }
}

// The widest run of consecutive registers that lands on consecutive threadgroup
// slots, capped at 4 (MSL's widest vector). Returns 1 when no widening is safe.
int MSLEmitter::stageVectorWidth(RankedTensorType stageTy, int regs,
                                 int64_t rowPad) {
  if (regs < 2)
    return 1;
  auto shape = stageTy.getShape();
  int rk = shape.size();
  if (rk < 2)
    return 1;
  auto slotOf = [&](int r) -> int64_t {
    auto c = layout.registerCoords(stageTy, r);
    if ((int)c.size() != rk)
      return -1;
    int64_t off = 0;
    for (int d = 0; d < rk; ++d)
      off = off * (d == rk - 1 ? shape[d] + rowPad : shape[d]) + c[d];
    return off;
  };
  int64_t s0 = slotOf(0);
  if (s0 < 0)
    return 1;
  int vw = 1;
  for (int k = 1; k < 4 && k < regs; ++k) {
    int64_t sk = slotOf(k);
    if (sk < 0 || sk - s0 != k)
      break;
    vw = k + 1;
  }
  if (vw < 2 || regs % vw != 0)
    return 1;
  // Every run must repeat the pattern, not just the first one.
  for (int base = vw; base + vw <= regs; base += vw) {
    int64_t b0 = slotOf(base);
    if (b0 < 0)
      return 1;
    for (int k = 1; k < vw; ++k)
      if (slotOf(base + k) - b0 != k)
        return 1;
  }
  return vw;
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
    mslReject(op, "planDot", "integer-operands");
    return p;
  }
  p.rank = aTy.getRank();
  if (p.rank != 2 && p.rank != 3) {
    mslReject(op, "planDot", "rank-not-2-or-3");
    return p;
  }
  if (!isDotOperandElem(aElem) || !isDotOperandElem(bElem)) {
    mslReject(op, "planDot", "operand-elem-unsupported");
    return p;
  }
  if (aElem != bElem) {
    mslReject(op, "planDot", "mixed-ab-elem");
    return p;
  }
  if (!(cElem.isF32() || cElem.isF16())) {
    mslReject(op, "planDot", "acc-elem-not-f32-f16");
    return p;
  }
  p.Bd = p.rank == 3 ? cTy.getShape()[0] : 1;
  p.M = cTy.getShape()[p.rank - 2];
  p.N = cTy.getShape()[p.rank - 1];
  p.K = aTy.getShape()[p.rank - 1];
  if (p.M % 8 || p.N % 8 || p.K % 8) {
    mslReject(op, "planDot", "MNK-not-multiple-of-8");
    return p;
  }

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

  if (getenv("TRITON_MSL_DMA_PROBE")) {
    auto report = [&](const char *which, Value stage, int64_t r, int64_t c) {
      auto m = matchDirectStage(stage, r, c);
      llvm::errs() << "[dma-probe] " << which << " " << r << "x" << c << " "
                   << (!m                 ? "no-match"
                       : m->srcTransposed ? "MATCH-transposed(declined)"
                                          : "MATCH")
                   << " def=";
      if (Operation *d = stage.getDefiningOp())
        llvm::errs() << d->getName();
      else
        llvm::errs() << "<blockarg>";
      llvm::errs() << "\n";
    };
    report("A", p.aStage, p.M, p.K);
    report("B", p.bStage, p.K, p.N);
  }

  dotStageRowPads(p.M, p.N, p.K, byteWidth(aElem), byteWidth(bElem), p.aPad,
                  p.bPad);
  if (p.aInPlace)
    p.aPad = 0;
  if (p.bInPlace)
    p.bPad = 0;
  p.phase = fusedDot.phase;
  // A padded row stride makes the copy's destination pitch differ from its
  // width, which costs far more in the DMA engine than the bank padding
  // recovers in simdgroup_load. Apple's own kernel never pads a copy
  // destination: every one of its 3136 copies has dstStride == cols.
  bool bDma = p.phase != FusedDotPhase::None && !p.bInPlace &&
              dmaStagingEnabled() && bDmaCandidate(op, /*requireBound=*/false);
  if (bDma)
    p.bPad = 0;
  p.stagedA = p.aInPlace ? 0 : p.M * (p.K + p.aPad) * byteWidth(aElem);
  p.stagedB = p.bInPlace ? 0 : p.K * (p.N + p.bPad) * byteWidth(bElem);
  p.stagedAB = p.stagedA + p.stagedB;
  // A second B tile for the in-flight copy, sitting directly after the first,
  // so C (disjoint or banded) starts past both. Every phase must agree on this,
  // or they would disagree about where C begins -- hence the binding-free form
  // of the candidate test, which depends only on the IR.
  // The second B tile is only worth its footprint when it does not push the
  // pool past a residency step: measured on an M1 Pro, dropping from three
  // resident threadgroups to two costs ~20%, which is far more than the copy
  // saves. Without this the DMA path loses on exactly the tiles it fits.
  p.dmaB = bDma && p.stagedB &&
           p.stagedAB + p.stagedB <= kTGResidentBudgetBytes &&
           tgResidency(p.stagedAB + p.stagedB) >= tgResidency(p.stagedAB);
  if (p.dmaB)
    p.stagedAB += p.stagedB;
  // The fused epilogue writes C only after the K-loop, behind a barrier, so
  // its accumulators can reuse the (dead) A/B staging instead of claiming a
  // disjoint region. Keeping C disjoint there doubles the threadgroup
  // footprint and costs residency.
  bool fusedEpilogueC = p.phase != FusedDotPhase::None;
  p.disjointC =
      !fusedEpilogueC && p.stagedAB + p.M * p.N * accBytes <= poolBudget();
  p.bandRows = p.M;
  if (!p.disjointC) {
    // C overlays the dead A/B staging in the fused epilogue, so a band that
    // fits inside what staging already claimed is free. Sizing it to the whole
    // budget instead would inflate the pool to the 32KB cap and drop residency
    // to one threadgroup per core, which costs far more than the extra bands.
    int64_t budget = fusedEpilogueC
                         ? std::max(p.stagedAB, (int64_t)8 * p.N * accBytes)
                         : poolBudget();
    p.bandRows =
        dotCBandRows(p.M, p.N, std::min(budget, poolBudget()), accBytes);
  }

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
  if (p.kind == DotPlan::Kind::Panel)
    mslReject(op, "planDot", "panel-AB-over-TG-budget");
  else if (p.kind == DotPlan::Kind::Direct)
    mslReject(op, "planDot", "direct-not-fused");
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

  // The pipeline's parity/handle names must exist before dotPoolPtrs, which
  // derives the read-side B pointer from the parity. Mint them on the first
  // phase that sees this dot; their declarations are emitted in Decl.
  if (plan.dmaB && fusedDot.dmaParity.empty()) {
    fusedDot.dmaParity = fresh();
    fusedDot.dmaHandle = fresh();
  }

  // Rotate the tile pair before the read-side pointer is derived: the copy the
  // previous trip issued targeted the tile this trip reads.
  if (plan.dmaB && plan.phase == FusedDotPhase::MMA)
    body.push_back(ctx.assignStmt(
        ctx.var(fusedDot.dmaParity),
        ctx.binary(B::Sub, ctx.i32lit(1), ctx.var(fusedDot.dmaParity))));

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
  // Default the read-side B pointer to the single-tile case; the double-
  // buffered path overrides it below. Phases that skip the A/B declarations
  // still hand this name to the fragment loads.
  dc.tgBCur = dc.tgB;
  if (plan.needAB) {
    body.push_back(ctx.declStmt(tgPtr(dc.opScalar), dc.tgA,
                                plan.aInPlace ? inPlaceBase(*plan.aInPlace)
                                              : poolRegion(0, dc.opScalar)));
    msl::Expr *bBase = plan.bInPlace ? inPlaceBase(*plan.bInPlace)
                                     : poolRegion(plan.stagedA, dc.opScalar);
    body.push_back(ctx.declStmt(tgPtr(dc.opScalar), dc.tgB, bBase));
    // With two B tiles, the one this trip's MMAs read is selected by the parity
    // flag. dc.tgB stays the base of the pair (the copy indexes off it), and
    // tgBCur is what the simdgroup_loads use.
    if (plan.dmaB && !fusedDot.dmaParity.empty()) {
      dc.tgBCur = fresh();
      int64_t eb = byteWidth(elementScalarType(plan.bStage.getType()));
      body.push_back(ctx.declStmt(
          tgPtr(dc.opScalar), dc.tgBCur,
          ctx.binary(B::Add, ctx.var(dc.tgB),
                     ctx.paren(ctx.binary(B::Mul,
                                          ctx.var(fusedDot.dmaParity),
                                          ctx.i32lit(plan.stagedB / eb))))));
    } else {
      dc.tgBCur = dc.tgB;
    }
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
    // so it always lands inside the row dim: only a band covering the whole
    // dim makes the guard a tautology. A partial band must keep it, or the
    // registers outside the band get clobbered by this band's gather.
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
      msl::Expr *batchEq =
          ctx.binary(B::Eq, layout.batchCoordExpr(cTy, r), ctx.i32lit(bi));
      guard = guard ? ctx.paren(ctx.binary(B::LAnd, batchEq, guard))
                    : ctx.paren(batchEq);
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
  // A fragment (mi, ki) is shared by every column this warp owns, and B (ki,
  // ni) by every row, so reloading per (mi, ni) pair costs nT-1 (resp. mT-1)
  // redundant simdgroup_loads each. The cache is cleared per warp block, which
  // is the region the fragments stay live across.
  llvm::DenseMap<std::pair<int64_t, int64_t>, std::string> aFrag, bFrag;
  auto clearFragCache = [&]() {
    aFrag.clear();
    bFrag.clear();
  };
  auto fragMMA = [&](int64_t mi, int64_t ni, StringRef acc, msl::Block &into) {
    for (int64_t ki = 0; ki < kT; ++ki) {
      auto &fa = aFrag[{mi, ki}];
      if (fa.empty()) {
        fa = fresh();
        into.push_back(fragDecl(dc.opFrag, fa));
        into.push_back(sgLoad(fa, dc.tgA, mi * 8 * lda + ki * 8, lda));
      }
      auto &fb = bFrag[{ki, ni}];
      if (fb.empty()) {
        fb = fresh();
        into.push_back(fragDecl(dc.opFrag, fb));
        into.push_back(sgLoad(fb, dc.tgBCur, ki * 8 * ldb + ni * 8, ldb));
      }
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
    // Each batch restages A/B into the same buffers, so fragments cached from
    // the previous batch name stale values.
    clearFragCache();
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
        clearFragCache();
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
      clearFragCache();
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
    // Software-pipeline state. The prologue copy itself is issued at the top of
    // the first MMA trip (the pool pointers are declared per-phase, so the
    // threadgroup tile names are not in scope here); a null handle marks "no
    // copy in flight yet" so that first trip skips the wait.
    if (plan.dmaB) {
      body.push_back(
          ctx.declStmt(ctx.named("int"), fusedDot.dmaParity, ctx.i32lit(0)));
      body.push_back(
          ctx.declStmt(ctx.named("ulong"), fusedDot.dmaHandle, ctx.lit("0")));
    }
    return true;
  }

  if (fusedDot.phase == FusedDotPhase::MMA) {
    // B may be staged by threadgroup DMA instead of through registers.
    //
    // Two B tiles alternate: this trip's MMAs read one while the copy feeding
    // the next trip fills the other. Both the issue and the wait for a given
    // tile therefore sit a full MMA block apart, which is the overlap -- a
    // single-buffered copy has only A's register scatter to hide behind and
    // measures ~30% slower than staging B through registers.
    //
    // The rotation is carried in threadgroup memory, not registers, so the
    // loop-carried state is just the parity flag and the event token.
    std::optional<DirectStage> bDma;
    if (plan.dmaB)
      bDma = dotDmaStage(op);

    bool stagesHere = !plan.aInPlace || !plan.bInPlace;
    if (stagesHere)
      barrier();
    if (bDma) {
      // First trip: nothing is in flight yet, so fill the tile this trip reads
      // and wait for it. Later trips consume the copy issued a block earlier.
      msl::Block prime;
      prime.push_back(dmaBeginInto(fusedDot.dmaHandle, plan, dc, *bDma,
                                   bStageTy, ldb, dotDmaTripVar(op),
                                   /*nextTrip=*/false));
      body.push_back(ctx.ifScope(
          ctx.binary(B::Eq, ctx.var(fusedDot.dmaHandle), ctx.lit("0")),
          std::move(prime)));
    }
    stageOperand(body, dc.tgA, dc.aStage, aStageTy, dc.aNames,
                 (bool)plan.aInPlace, nullptr, plan.aPad);
    stageOperand(body, dc.tgB, dc.bStage, bStageTy, dc.bNames,
                 (bool)plan.bInPlace || bDma.has_value(), nullptr, plan.bPad);
    // A's register scatter is independent of B's tile, so it overlaps the
    // transfer; the wait only has to precede the publish barrier.
    if (bDma)
      body.push_back(dmaWait(fusedDot.dmaHandle));
    barrier();
    // Issue the next trip's B copy into the tile this trip is *not* reading.
    // It lands after the publish barrier so the previous copy has been waited
    // on and both tiles are quiescent, and before the MMA block so the whole
    // block overlaps the transfer.
    // The read-side pointer was derived from the parity at the top of this
    // trip, so the parity must not move until the MMAs are done with it. Issue
    // the next trip's copy into the other tile (parity, not 1-parity) and flip
    // only at the start of the next trip.
    if (bDma)
      body.push_back(dmaBeginInto(fusedDot.dmaHandle, plan, dc, *bDma,
                                  bStageTy, ldb, dotDmaTripVar(op),
                                  /*nextTrip=*/true));
    // A prefetched operand arrives as a loop-carried block argument: its load
    // was issued for the *next* iteration, so the value consumed here is
    // already in registers and its global load can sink past this barrier to
    // overlap the MMAs below. Emit those loads now, after the publish barrier.
    emitPendingPrefetchLoads(body);
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
                         {ctx.var(fb), ctx.binary(B::Add, ctx.var(dc.tgBCur), off),
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
        msl::Expr *col = ctx.paren(
            ctx.binary(B::Rem, ctx.var(warpId), ctx.i32lit(wt.wGridN)));
        return ctx.paren(ctx.mul(col, ctx.i32lit(wt.niCount)));
      };
      std::string colKey = "(" + warpId + " % " + std::to_string(wt.wGridN) +
                           ")*" + std::to_string(wt.niCount);
      auto emitRow = [&](int64_t wr, msl::Block &into) {
        SmallVector<Slot> slots;
        for (int64_t r = 0; r < wt.miCount; ++r)
          for (int64_t c = 0; c < wt.niCount; ++c) {
            msl::Expr *ni =
                c ? ctx.paren(ctx.add(niBase(), ctx.i32lit(c))) : niBase();
            slots.push_back(
                {wr * wt.miCount + r, colKey + "+" + std::to_string(c), ni});
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
    std::string base = scalarName(d.basePtr).str();
    std::string rowB = scalarName(d.rowBase).str(),
                colB = scalarName(d.colBase).str();
    auto uniform = [&](const UniformInt &u) -> msl::Expr * {
      return u.lit ? static_cast<msl::Expr *>(ctx.i32lit(*u.lit))
                   : ctx.var(scalarName(u.val));
    };
    // simdgroup_store requires the destination scalar to match the fragment's
    // element type, so a narrow output needs a narrow fragment: each thread
    // owns two elements, converted through thread_elements().
    auto narrowed = [&](msl::Block &blk, StringRef accName) -> std::string {
      if (!d.narrowTo)
        return accName.str();
      msl::Type *mt =
          ctx.matrix(d.narrowTo.isF16() ? msl::MatrixType::Elem::Half
                                        : msl::MatrixType::Elem::Bfloat);
      msl::Type *sc = scalarType(d.narrowTo);
      std::string id = fresh();
      blk.push_back(ctx.declStmt(mt, id, nullptr));
      for (int e = 0; e < 2; ++e) {
        auto elems = [&](StringRef nm) {
          return ctx.subscript(
              ctx.member(ctx.var(nm), msl::builtin::sg::ThreadElements),
              ctx.i32lit(e));
        };
        blk.push_back(ctx.assignStmt(elems(id),
                                     ctx.cast(CS::CStyle, sc, elems(accName))));
      }
      return id;
    };
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
                     uniform(d.ldc)),
             ctx.paren(ctx.add(ctx.var(colB), ctx.i32lit(ni * 8)))});
        std::string sv = narrowed(inner, fusedDot.accNames[j]);
        inner.push_back(ctx.exprStmt(ctx.call(
            msl::builtin::sg::Store, {ctx.var(sv), off, uniform(d.ldc)})));
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
    if (d.alwaysFullTile) {
      // Unmasked store: the fallback arm is dead. Emitting it anyway keeps a
      // threadgroup C pointer live and forces a pool reservation the kernel
      // never touches.
      for (msl::Stmt *s : ifBody)
        body.push_back(s);
    } else {
      body.push_back(ctx.ifElseScope(ctx.var(d.fullTileVar), std::move(ifBody),
                                     std::move(elseBody)));
    }
    return true;
  }

  // Non-direct readback: pool store + gather. The drain overlays the dead A/B
  // staging, so it has to happen in one pass -- a second band's stores would
  // land on the region the first is still gathering from.
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
// flag does NOT produce a transposed product: measured against a probe
// validated to 0.0 on A@B, enabling it matches none of the twelve orientation
// combinations, so it cannot absorb a column-major operand.
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

// The integer a uniform tensor holds, when it is a splat arith.constant.
// A kernel-invariant stride or bound reaches the matchers either as a
// tt::SplatOp or as a splat constant, so both spellings must be accepted.
static std::optional<int64_t> splatConstInt(Value v) {
  DenseElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)) || !attr.isSplat())
    return std::nullopt;
  auto ty = dyn_cast<IntegerType>(attr.getElementType());
  if (!ty)
    return std::nullopt;
  return attr.getSplatValue<APInt>().getSExtValue();
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

UniformInt MSLEmitter::matchUniformInt(Value x) {
  UniformInt u;
  Value peeled = peelBroadcastExpand(x);
  if (auto sp = dyn_cast_or_null<tt::SplatOp>(peeled.getDefiningOp())) {
    u.val = sp.getSrc();
    return u;
  }
  u.lit = splatConstInt(peeled);
  return u;
}

bool MSLEmitter::matchRowMajorOffset(Value off, Value &rowBase, UniformInt &ldc,
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
    Value rIdxA = mul.getLhs(), rIdxB = mul.getRhs();
    Value rb = matchTileIndex(rIdxA);
    UniformInt stride = matchUniformInt(rIdxB);
    if (!rb || !stride) {
      rb = matchTileIndex(rIdxB);
      stride = matchUniformInt(rIdxA);
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

// `rowBase`/`colBase` are the tile origins the store's own offset used, so each
// half of the conjunction is assigned to the axis whose origin it compares
// against. Matching on operand order instead would silently swap the two bounds
// whenever the IR happened to emit them the other way round, which only shows
// up when both axes are ragged AND M != N.
bool MSLEmitter::matchBoundaryMask(Value m, Value rowBase, Value colBase,
                                   UniformInt &boundM, UniformInt &boundN) {
  auto conj = definingOp<arith::AndIOp>(m);
  if (!conj)
    return false;
  auto cmpBound = [&](Value side, Value &axis) -> UniformInt {
    Value c = peelBroadcastExpand(side);
    auto cmp = definingOp<arith::CmpIOp>(c);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      return UniformInt();
    axis = matchTileIndex(cmp.getLhs());
    if (!axis)
      return UniformInt();
    return matchUniformInt(cmp.getRhs());
  };
  Value axisL, axisR;
  UniformInt bl = cmpBound(conj.getLhs(), axisL);
  UniformInt br = cmpBound(conj.getRhs(), axisR);
  if (!bl || !br)
    return false;
  if (axisL == rowBase && axisR == colBase) {
    boundM = bl;
    boundN = br;
    return true;
  }
  if (axisR == rowBase && axisL == colBase) {
    boundM = br;
    boundN = bl;
    return true;
  }
  return false;
}

std::optional<DirectStore> MSLEmitter::matchDirectStore(Value forResult) {
  Operation *site = forResult.getDefiningOp();
  auto rej = [&](StringRef why) {
    if (site)
      mslReject(site, "matchDirectStore", why);
    return std::nullopt;
  };
  if (!forResult.hasOneUse())
    return rej("acc-not-single-use");
  // The f32 accumulator may be narrowed to the output dtype before the layout
  // convert. simdgroup_store cannot narrow, so the fragment is converted
  // elementwise through thread_elements() and stored as a narrow fragment.
  Value chain = forResult;
  Type narrowTo;
  arith::TruncFOp narrowOp;
  if (auto tf = dyn_cast<arith::TruncFOp>(*chain.user_begin())) {
    if (!tf.getResult().hasOneUse())
      return rej("narrow-not-single-use");
    narrowTo = cast<RankedTensorType>(tf.getType()).getElementType();
    if (!narrowTo.isF16() && !narrowTo.isBF16())
      return rej("narrow-target-unsupported");
    narrowOp = tf;
    chain = tf.getResult();
  }
  auto cvt = dyn_cast<ttg::ConvertLayoutOp>(*chain.user_begin());
  if (!cvt)
    return rej("user-not-convert_layout");
  if (!cvt.getResult().hasOneUse())
    return rej("convert-not-single-use");
  auto store = dyn_cast<tt::StoreOp>(*cvt.getResult().user_begin());
  if (!store || store.getValue() != cvt.getResult())
    return rej("convert-user-not-store");
  auto cTy = dyn_cast<RankedTensorType>(cvt.getResult().getType());
  if (!cTy)
    return rej("store-value-not-tensor");
  if (narrowTo) {
    if (cTy.getElementType() != narrowTo)
      return rej("store-elem-mismatches-narrow");
  } else if (!cTy.getElementType().isF32()) {
    return rej("store-elem-not-f32");
  }
  auto ptr = definingOp<tt::AddPtrOp>(store.getPtr());
  if (!ptr)
    return rej("ptr-not-addptr");
  Value ptrBase = ptr.getPtr();
  Value rowOff;
  if (auto inner = definingOp<tt::AddPtrOp>(peelBroadcast(ptrBase))) {
    rowOff = inner.getOffset();
    ptrBase = inner.getPtr();
  }
  auto splat = definingOp<tt::SplatOp>(peelBroadcast(ptrBase));
  if (!splat)
    return rej("ptr-base-not-splat");
  if (!isa<BlockArgument>(splat.getSrc()))
    return rej("ptr-base-not-block-arg");
  Value rowBase, colBase;
  UniformInt ldc;
  if (rowOff) {
    Value cb = matchTileIndex(ptr.getOffset());
    auto mul = definingOp<arith::MulIOp>(peelBroadcastExpand(rowOff));
    if (!cb || !mul)
      return rej("split-offset-shape");
    Value rb = matchTileIndex(mul.getLhs());
    UniformInt stride = matchUniformInt(mul.getRhs());
    if (!rb || !stride) {
      rb = matchTileIndex(mul.getRhs());
      stride = matchUniformInt(mul.getLhs());
    }
    if (!rb || !stride)
      return rej("split-offset-rowterm");
    rowBase = rb;
    colBase = cb;
    ldc = stride;
  } else if (!matchRowMajorOffset(ptr.getOffset(), rowBase, ldc, colBase)) {
    return rej("offset-not-row-major");
  }
  UniformInt boundM, boundN;
  if (store.getMask()) {
    if (!matchBoundaryMask(store.getMask(), rowBase, colBase, boundM, boundN))
      return rej("mask-not-boundary");
  }
  DirectStore ds;
  ds.store = store;
  ds.basePtr = splat.getSrc();
  ds.ldc = ldc;
  ds.rowBase = rowBase;
  ds.colBase = colBase;
  ds.boundM = boundM;
  ds.boundN = boundN;
  ds.narrowTo = narrowTo;
  ds.narrowOp = narrowOp;
  ds.cvt = cvt;
  return ds;
}

// True when nothing routes the C tile through threadgroup memory: the
// accumulator fragments go straight to device memory (DirectStore), and every
// convert_layout it feeds is a lane permutation. Without the DirectStore half
// the fused readback still stages C through the pool via simdgroup_store, so
// the reservation has to stay.
// True when the direct C store keeps a threadgroup fallback arm: a boundary
// mask means the ragged tile still goes through the pool, so its space cannot
// be reclaimed even though the full-tile path stores straight to device.
// True when this dot's C goes straight to device with no threadgroup fallback
// arm, so the readback never touches the pool.
bool MSLEmitter::cStoresDirect(tt::DotOp d) {
  auto forOp = dyn_cast<scf::ForOp>(d->getParentOp());
  if (!forOp)
    return false;
  auto m = matchGemmDotLoop(forOp);
  if (!m || m->first != d)
    return false;
  auto ds = matchDirectStore(forOp.getResult(m->second));
  if (!ds)
    return false;
  if (!ds->boundM)
    return true;
  auto cTy = cast<RankedTensorType>(d.getResult().getType());
  int rk = cTy.getRank();
  return directStoreNeverRagged(*ds, cTy.getShape()[rk - 2],
                                cTy.getShape()[rk - 1]);
}

// `origin` is a multiple of `blk` when it is spelled `x * blk` (or `blk * x`),
// which is how every tile origin reaches here: `pid * BLOCK`. A bare scalar
// carries no such proof and is rejected.
static bool isMultipleOfBlock(Value origin, int64_t blk) {
  if (blk <= 0)
    return false;
  auto mul = dyn_cast_or_null<arith::MulIOp>(origin.getDefiningOp());
  if (!mul)
    return false;
  APInt c;
  for (Value side : {mul.getRhs(), mul.getLhs()})
    if (matchPattern(side, m_ConstantInt(&c)) && c.getSExtValue() != 0 &&
        c.getSExtValue() % blk == 0)
      return true;
  return false;
}

bool MSLEmitter::directStoreNeverRagged(const DirectStore &ds, int64_t M,
                                        int64_t N) {
  if (!ds.boundM || !ds.boundN)
    return false;
  if (!ds.boundM.lit || !ds.boundN.lit)
    return false;
  if (M <= 0 || N <= 0 || *ds.boundM.lit % M != 0 || *ds.boundN.lit % N != 0)
    return false;
  return isMultipleOfBlock(ds.rowBase, M) && isMultipleOfBlock(ds.colBase, N);
}

bool MSLEmitter::fusedGemmCHasFallback(tt::DotOp d) {
  auto forOp = dyn_cast<scf::ForOp>(d->getParentOp());
  if (!forOp)
    return false;
  auto m = matchGemmDotLoop(forOp);
  if (!m || m->first != d)
    return false;
  auto ds = matchDirectStore(forOp.getResult(m->second));
  if (!ds || !ds->boundM)
    return false;
  auto cTy = cast<RankedTensorType>(d.getResult().getType());
  int rk = cTy.getRank();
  return !directStoreNeverRagged(*ds, cTy.getShape()[rk - 2],
                                 cTy.getShape()[rk - 1]);
}

bool MSLEmitter::fusedGemmCIsShuffled(tt::DotOp d) {
  auto forOp = dyn_cast<scf::ForOp>(d->getParentOp());
  if (!forOp)
    return false;
  auto m = matchGemmDotLoop(forOp);
  if (!m || m->first != d)
    return false;
  Value acc = forOp.getResult(m->second);
  if (!matchDirectStore(acc))
    return false;

  SmallVector<Value> work{acc};
  SmallVector<Value> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (llvm::is_contained(seen, v))
      continue;
    seen.push_back(v);
    if (seen.size() > 64)
      return false; // give up rather than walk an unbounded epilogue
    for (Operation *user : v.getUsers()) {
      if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(user)) {
        auto st = cast<RankedTensorType>(cl.getSrc().getType());
        auto rt = cast<RankedTensorType>(cl.getResult().getType());
        if (ttg::toLinearLayout(st) == ttg::toLinearLayout(rt))
          continue;
        auto plan = planIntraWarpShuffle(st, rt);
        if (!plan || !plan->uniformLanePerm || !plan->lanePermLinear)
          return false;
        continue;
      }
      // Elementwise ops keep the accumulator in its layout, so the convert
      // downstream of them is still the C relayout. Anything else (reduce,
      // trans, reshape, dot) may itself need the pool: bail.
      if (!user->hasTrait<OpTrait::Elementwise>() &&
          !isa<arith::TruncFOp, arith::ExtFOp, arith::SelectOp, tt::StoreOp>(
              user))
        return false;
      for (Value r : user->getResults())
        work.push_back(r);
    }
  }
  return true;
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
  if (!found)
    return std::nullopt;
  if (nDots != 1) {
    mslReject(op, "matchGemmDotLoop", "nDots!=1");
    return std::nullopt;
  }

  auto cArg = dyn_cast<BlockArgument>(found.getC());
  if (!cArg || cArg.getOwner() != body) {
    mslReject(op, "matchGemmDotLoop", "acc-not-iter-arg");
    return std::nullopt;
  }
  unsigned idx = cArg.getArgNumber();
  if (idx == 0)
    return std::nullopt; // arg 0 is the induction var
  unsigned iterIdx = idx - 1;
  if (yield.getOperand(iterIdx) != found.getResult()) {
    mslReject(op, "matchGemmDotLoop", "yield-not-dot-result");
    return std::nullopt;
  }
  // The iter-arg feeds the dot's C and nothing else; the dot result feeds the
  // yield and nothing else. This keeps the accumulator purely register-carried.
  for (Operation *u : cArg.getUsers())
    if (u != found.getOperation()) {
      mslReject(op, "matchGemmDotLoop", "acc-extra-user");
      return std::nullopt;
    }
  for (Operation *u : found.getResult().getUsers())
    if (u != yield.getOperation()) {
      mslReject(op, "matchGemmDotLoop", "dot-result-extra-user");
      return std::nullopt;
    }

  auto cTy = dyn_cast<RankedTensorType>(found.getResult().getType());
  if (!cTy || cTy.getRank() != 2) {
    mslReject(op, "matchGemmDotLoop", "acc-rank!=2");
    return std::nullopt;
  }
  Type aElem = cast<RankedTensorType>(found.getA().getType()).getElementType();
  Type cElem = cTy.getElementType();
  if (isa<IntegerType>(aElem) || !(cElem.isF32() || cElem.isF16())) {
    mslReject(op, "matchGemmDotLoop", "elem-type-unsupported");
    return std::nullopt;
  }
  int64_t M = cTy.getShape()[0], N = cTy.getShape()[1];
  int64_t K = cast<RankedTensorType>(found.getA().getType()).getShape()[1];
  if (M % 8 || N % 8 || K % 8) {
    mslReject(op, "matchGemmDotLoop", "MNK-not-multiple-of-8");
    return std::nullopt;
  }

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
  // The fused epilogue overlays C's accumulators on the dead A/B staging and
  // writes it one row band at a time, so C never needs the whole tile - only a
  // single 8-row band has to fit alongside the staging.
  int64_t cBand =
      std::min(cFull, std::max(stagedA + stagedB, (int64_t)8 * N * 4));
  // Operands staged into their own local_alloc buffers are dead by the
  // epilogue exactly like pool staging is, so the C band overlays them too.
  // Counting the band *on top* of those buffers made poolBudget() zero and
  // rejected the whole fused path -- which then forced C back through the
  // pool, the very reservation this check was worried about.
  // Credit only *this dot's* operand tiles, not liveTgBytes: that counts every
  // local_alloc in the function, so on a multi-dot kernel it hands out far more
  // headroom than the epilogue can actually reuse and the kernel then asks for
  // more threadgroup memory than the device has.
  int64_t liveOperandBytes =
      (stagedA == 0 && stagedB == 0)
          ? std::min(liveTgBytes, aBytes + bBytes)
          : 0;
  if (std::max(stagedA + stagedB, cBand) > poolBudget() + liveOperandBytes) {
    mslReject(op, "matchGemmDotLoop", "pool-over-budget");
    return std::nullopt;
  }
  tt::LinearLayout cLL = ttg::toLinearLayout(cTy);
  auto kWarpDim = StringAttr::get(op.getContext(), "warp");
  int64_t numWarps = cLL.hasInDim(kWarpDim) ? cLL.getInDimSize(kWarpDim) : 1;
  int64_t nFrag = (M / 8) * (N / 8);
  if (numWarps > nFrag)
    numWarps = nFrag;
  int64_t fragsPerWarp = (nFrag + numWarps - 1) / numWarps;
  if (fragsPerWarp > 32) {
    mslReject(op, "matchGemmDotLoop", "fragsPerWarp>32");
    return std::nullopt;
  }

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
