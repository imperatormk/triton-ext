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

namespace {

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
bool isUnitRange(Value v, Value *shiftOut = nullptr, int64_t tileRows = 0,
                 std::optional<int64_t> *shiftLitOut = nullptr) {
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
    return isUnitRange(rem.getLhs(), shiftOut, tileRows, shiftLitOut);
  }
  if (auto add = definingOp<arith::AddIOp>(s)) {
    Value a = add.getLhs(), b = add.getRhs();
    for (int i = 0; i < 2; ++i) {
      Value pa = peelBroadcast(a);
      if (auto sp = definingOp<tt::SplatOp>(pa))
        if (isUnitRange(b, nullptr, tileRows)) {
          if (shiftOut)
            *shiftOut = sp.getSrc();
          return true;
        }
      if (auto cst = definingOp<arith::ConstantOp>(pa))
        if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
          if (dense.isSplat() && isUnitRange(b, nullptr, tileRows)) {
            if (shiftLitOut)
              *shiftLitOut = dense.getSplatValue<APInt>().getSExtValue();
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
                     bool *matched = nullptr,
                     std::optional<int64_t> *shiftLitOut = nullptr) {
  auto mul = definingOp<arith::MulIOp>(peelBroadcast(v));
  if (!mul)
    return nullptr;
  Value a = mul.getLhs(), b = mul.getRhs();
  for (int i = 0; i < 2; ++i) {
    if (isUnitRange(a, shiftOut, tileRows, shiftLitOut)) {
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

static bool axisNeverRagged(const UniformInt &bound, Value origin, int64_t blk);

std::optional<DirectStage> matchDirectStage(Value operand, int64_t rows,
                                            int64_t cols) {
  bool dbg = getenv("TRITON_MSL_DMA_PROBE") != nullptr;
  auto bail = [&](const char *why) {
    if (dbg)
      llvm::errs() << "[dma-probe]   bail: " << why << "\n";
    return std::nullopt;
  };

  // A transposed operand is the same device tile read with its axes swapped,
  // which simdgroup_load expresses directly, so the transpose is peeled and the
  // underlying tile matched at its own shape.
  bool transposed = false;
  if (auto tr = definingOp<tt::TransOp>(operand)) {
    auto ord = tr.getOrder();
    if (ord.size() != 2 || ord[0] != 1 || ord[1] != 0)
      return bail("trans order not a 2D swap");
    transposed = true;
    operand = tr.getSrc();
    std::swap(rows, cols);
  }

  auto ld = definingOp<tt::LoadOp>(operand);
  if (!ld)
    return bail("operand not defined by a tt.load");

  // The intrinsic cannot honour per-element predicates. A mask bounding only
  // the row axis is a whole-tile property, so it is carried out for the caller
  // to prove; anything else keeps the register path.
  auto ds = matchTilePointer(ld.getPtr(), rows, cols);
  if (ds) {
    ds->rowMask = ld.getMask();
    ds->fragTransposed = transposed;
  }
  return ds;
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
  std::optional<int64_t> rowShiftLit, colShiftLit;
  bool rowMatched = false, colMatched = false;
  Value rowStride = matchRowStride(rowAxis, &rowShift, rows, &rowStrideLit,
                                   &rowMatched, &rowShiftLit);
  Value colStride = matchRowStride(colAxis, &colShift, cols, &colStrideLit,
                                   &colMatched, &colShiftLit);
  if (!rowMatched && !isUnitRange(rowAxis, &rowShift, rows, &rowShiftLit))
    return bail("row offset neither iota nor iota*splat(stride)");
  if (!colMatched && !isUnitRange(colAxis, &colShift, cols, &colShiftLit))
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
  out.rowShiftLit = rowShiftLit;
  out.colShiftLit = colShiftLit;
  out.ptrDelta = delta;
  out.rowStrideLit = strideLit;
  out.ptrDeltaLit = deltaLit;
  out.aheadSteps = peeled;
  out.rows = rows;
  out.cols = cols;
  out.srcTransposed = transposed;
  return out;
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
// only a larger per-warp accumulator tile lowers it.
WarpTiling planWarpTiling(int64_t mT, int64_t nT, int64_t numWarps,
                          int64_t nFrag, int64_t fragsPerWarp,
                          bool preferMSplit) {
  WarpTiling t;
  if (numWarps < 1 || fragsPerWarp < 2)
    return t;
  // Every warp must be full and the grid exactly covered.
  if (mT * nT != nFrag || numWarps * fragsPerWarp != nFrag)
    return t;

  // A device-direct A operand needs each warp to own a contiguous band of M and
  // the full N extent, so that the rows a warp reads are private to it. That is
  // the ni == nT split; it costs more simdgroup_loads than the near-square one,
  // but those loads now come from device memory and the staging they replace
  // cost more than they do.
  if (preferMSplit && nT <= fragsPerWarp && fragsPerWarp % nT == 0) {
    int64_t mi = fragsPerWarp / nT;
    if (mT % mi == 0 && (mT / mi) == numWarps) {
      t.miCount = mi;
      t.niCount = nT;
      t.wGridN = 1;
      t.twoD = true;
      return t;
    }
  }

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
          ctx.deref(ctx.paren(
              ctx.cast(msl::Cast::Style::CStyle, tgVecPtr,
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
const DotFacts &MSLEmitter::dotFacts(tt::DotOp op) {
  auto it = dotFactsCache.find(op.getOperation());
  if (it != dotFactsCache.end())
    return it->second;

  DotFacts f;
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  auto cTy = cast<RankedTensorType>(op.getResult().getType());
  Type aElem = aTy.getElementType();
  Type bElem = bTy.getElementType();
  Type cElem = cTy.getElementType();

  f.rank = cTy.getRank();
  f.M = cTy.getShape()[f.rank - 2];
  f.N = cTy.getShape()[f.rank - 1];
  f.K = aTy.getShape()[f.rank - 1];
  f.aElemBytes = byteWidth(aElem);
  f.bElemBytes = byteWidth(bElem);
  f.intOperands = isa<IntegerType>(aElem) || isa<IntegerType>(bElem) ||
                  isa<IntegerType>(cElem);
  f.usable = !f.intOperands && (f.rank == 2 || f.rank == 3) &&
             isDotOperandElem(aElem) && isDotOperandElem(bElem) &&
             aElem == bElem && (cElem.isF32() || cElem.isF16()) && !(f.M % 8) &&
             !(f.N % 8) && !(f.K % 8);

  f.abResident =
      f.rank == 2 && f.M * f.K * f.aElemBytes + f.K * f.N * f.bElemBytes <=
                         kTGResidentBudgetBytes;
  if (f.abResident) {
    f.aInPlace = dotOperandInPlaceBuf(op.getA(), f.M, f.K);
    f.bInPlace = dotOperandInPlaceBuf(op.getB(), f.K, f.N);
    f.aNoStage =
        f.aInPlace.has_value() || dotOperandInLocalAlloc(op.getA(), f.M, f.K);
    f.bNoStage =
        f.bInPlace.has_value() || dotOperandInLocalAlloc(op.getB(), f.K, f.N);
  }
  if (!f.aInPlace)
    f.aDirect = dotADirect(op);
  // The phase-free candidate test: scanPool runs before any fused phase is
  // set, so asking dotDmaStage here would see phase None and disagree.
  f.bDma = dmaStagingEnabled() && bDmaCandidate(op, false).has_value();

  return dotFactsCache.insert({op.getOperation(), f}).first->second;
}

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
    op.emitError("tt.dot operand element type is not supported on Apple GPUs: "
                 "got ")
        << aElem << " x " << bElem
        << "; simdgroup matrix operands must be f32, f16 or bf16";
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

  int64_t accBytes = 4;

  // In-place aliasing is only offered when the whole A+B tile is resident, so
  // this tests the unstaged footprint against the hard 32KB cap, not the
  // (smaller) live pool budget that sizes the staged regions below. Taken from
  // dotFacts so scanPool cannot answer it differently.
  const DotFacts &facts = dotFacts(op);
  p.aInPlace = facts.aInPlace;
  p.bInPlace = facts.bInPlace;
  p.aNoStage = facts.aNoStage;
  p.bNoStage = facts.bNoStage;
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

  p.phase = fusedDot.phase;
  // A padded row stride makes the copy's destination pitch differ from its
  // width, which costs far more in the DMA engine than the bank padding
  // recovers in simdgroup_load. Apple's own kernel never pads a copy
  // destination: every one of its 3136 copies has dstStride == cols.
  bool bDma = p.phase != FusedDotPhase::None && !p.bInPlace &&
              dmaStagingEnabled() && bDmaCandidate(op, /*requireBound=*/false);
  // A device-direct A claims no pool at all. Taken from dotFacts, which is
  // also what scanPool reserves against, so the two cannot disagree about it.
  p.aDirect = facts.aDirect;
  if (!p.aInPlace && !p.aDirect)
    p.aFrag = aFragEligible(op, p);
  // Other consumers of A still read the staged tile (flash reads P for its l_i
  // reduction), so staging is only dropped when the dot is the sole reader.
  p.aFragOnly = p.aFrag && op.getA().hasOneUse();
  // aNoStage/bNoStage rather than aInPlace/bInPlace: the in-place answer needs
  // a memdescMap entry the body walk has not made yet in the Decl phase, so
  // asking it here sizes A's pad differently per phase and the pool the Decl
  // phase reserved no longer covers what the MMA phase lays down.
  dotStageRowPads(p.M, p.N, p.K, byteWidth(aElem), byteWidth(bElem), p.aPad,
                  p.bPad, p.aNoStage || p.aDirect.has_value() || p.aFragOnly,
                  p.bNoStage || bDma);
  // The panel path stages B against its own K x N offset arithmetic, so the
  // pre-transpose order is only available to the whole-tile paths.
  if (!p.bInPlace && p.rank == 2 &&
      !dotNeedsPanel(p.M, p.N, p.K, byteWidth(aElem), accBytes))
    if (auto tr = p.bStage.getDefiningOp<tt::TransOp>()) {
      auto ord = tr.getOrder();
      auto srcTy = dyn_cast<RankedTensorType>(tr.getSrc().getType());
      if (ord.size() == 2 && ord[0] == 1 && ord[1] == 0 && srcTy &&
          srcTy.getShape()[0] == p.N && srcTy.getShape()[1] == p.K) {
        p.bStage = tr.getSrc();
        p.bStageTransposed = true;
      }
    }
  // The pipelined copy may store a column-major B in its own N x K shape; the
  // fragment loads then have to read it back with the transpose flag. Decided
  // from the copy, not the phase, so every phase of one dot agrees.
  if (!p.bStageTransposed && p.rank == 2)
    if (auto ac = dotBFillCopy(op, p.K, p.N))
      if (auto ds = asyncCopyDma(ac))
        if (ds->srcTransposed && dmaStoreTransposed(ac))
          p.bStageTransposed = true;
  // The transposed staging addresses B at pitch K exactly, so it cannot carry
  // a row pad.
  if (p.bStageTransposed)
    p.bPad = 0;
  p.stagedA = (p.aInPlace || p.aNoStage || p.aDirect || p.aFragOnly)
                  ? 0
                  : p.M * (p.K + p.aPad) * byteWidth(aElem);
  p.stagedB = (p.bInPlace || p.bNoStage) ? 0
              : p.bStageTransposed       ? p.N * p.K * byteWidth(bElem)
                                   : p.K * (p.N + p.bPad) * byteWidth(bElem);
  p.stagedAB = p.stagedA + p.stagedB;
  // A second B tile for the in-flight copy, sitting directly after the first,
  // so C (disjoint or banded) starts past both. Every phase must agree on this,
  // or they would disagree about where C begins -- hence the binding-free form
  // of the candidate test, which depends only on the IR.
  // The second B tile is only worth its footprint when it does not push the
  // pool past a residency step, which costs far more than the copy saves.
  p.dmaB = bDma && p.stagedB &&
           p.stagedAB + p.stagedB <= kTGResidentBudgetBytes &&
           tgResidency(p.stagedAB + p.stagedB) >= tgResidency(p.stagedAB);
  if (getenv("TRITON_MSL_DMA_PROBE"))
    llvm::errs() << "[dma-probe] dmaB bDma=" << (bool)bDma
                 << " stagedA=" << p.stagedA << " stagedB=" << p.stagedB
                 << " fits="
                 << (p.stagedAB + p.stagedB <= kTGResidentBudgetBytes)
                 << " resid="
                 << (tgResidency(p.stagedAB + p.stagedB) >=
                     tgResidency(p.stagedAB))
                 << " => " << p.dmaB << "\n";
  if (p.dmaB)
    p.stagedAB += p.stagedB;
  // The fused epilogue writes C only after the K-loop, behind a barrier, so
  // its accumulators can reuse the (dead) A/B staging instead of claiming a
  // disjoint region. Keeping C disjoint there doubles the threadgroup
  // footprint and costs residency.
  bool fusedEpilogueC = p.phase != FusedDotPhase::None;
  auto it = dotCReserved.find(op);
  int64_t cRoom = it == dotCReserved.end()
                      ? std::max<int64_t>(poolBytes - p.stagedAB, 0)
                      : it->second;
  p.disjointC = !fusedEpilogueC && p.M * p.N * accBytes <= cRoom;
  p.bandRows = p.M;
  if (!p.disjointC) {
    // A non-disjoint C is published at pool offset 0 once A and B are dead, so
    // the staging it overlays is its room and a band fitting inside it is free.
    // cRoom describes only the space past the staging, which is what a disjoint
    // C would need; clamping to it collapses the band to the 8-row floor.
    int64_t budget = std::max(p.stagedAB, (int64_t)8 * p.N * accBytes);
    p.bandRows = dotCBandRows(p.M, p.N, std::max(budget, cRoom), accBytes);
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

  bool needsPanel = dotNeedsPanel(p.M, p.N, p.K, byteWidth(aElem), accBytes) &&
                    p.stagedAB > kTGResidentBudgetBytes;
  p.kind = needsPanel                       ? DotPlan::Kind::Panel
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
  dc.bStageTransposed = plan.bStageTransposed;
  if (plan.bStageTransposed)
    dc.bStageLd = plan.K;

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

  dotPoolPtrs(body, op, plan, dc);
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
// Overlap between two simultaneously live pool regions is a miscompile no
// count-based census can see: the MMA and accumulator counts stay correct while
// one region's stores land inside another's tile.
void MSLEmitter::checkPoolRegions(Operation *op, ArrayRef<PoolRegion> live) {
  // A failure here is our own layout arithmetic disagreeing with the size
  // scanPool reserved, not a gate declining to handle an op, so it is reported
  // whether or not reject logging is on. Silently, the caller only sees the
  // enclosing op fail to emit, which names the wrong operation entirely.
  auto fail = [&](const std::string &why) {
    mslReject(op, "poolRegions", why);
    if (!mslLogReject())
      llvm::errs() << "MSL-POOL\t" << why << '\t' << *op << '\n';
    emitFailed = true;
  };
  for (const PoolRegion &r : live) {
    if (!r.bytes || r.begin + r.bytes <= poolBytes)
      continue;
    fail(std::string(r.name) + "[" + std::to_string(r.begin) + "," +
         std::to_string(r.begin + r.bytes) + ") past pool " +
         std::to_string(poolBytes));
  }
  for (size_t i = 0; i < live.size(); ++i)
    for (size_t j = i + 1; j < live.size(); ++j) {
      if (!live[i].bytes || !live[j].bytes)
        continue;
      int64_t ie = live[i].begin + live[i].bytes;
      int64_t je = live[j].begin + live[j].bytes;
      if (live[i].begin < je && live[j].begin < ie) {
        fail(std::string(live[i].name) + "[" + std::to_string(live[i].begin) +
             "," + std::to_string(ie) + ") overlaps " + live[j].name + "[" +
             std::to_string(live[j].begin) + "," + std::to_string(je) + ")");
      }
    }
}

void MSLEmitter::checkDotPoolRegions(tt::DotOp op, const DotPlan &plan) {
  // The panel path walks sub-tiles and carries its own region check; the
  // whole-tile extents below are never laid down for it.
  if (plan.kind == DotPlan::Kind::Panel)
    return;
  int64_t aBytes = plan.stagedA;
  int64_t bBytes = plan.stagedB * (plan.dmaB ? 2 : 1);
  // A C that overlays the staging is published only once A and B are dead, so
  // it is not live alongside them.
  int64_t cBytes =
      (plan.needC && plan.disjointC) ? plan.bandRows * plan.N * 4 : 0;
  checkPoolRegions(op, {{"A", 0, aBytes},
                        {"B", plan.stagedA, bBytes},
                        {"C", plan.stagedAB, cBytes}});
}

// unconditionally even when a phase suppresses the matching decl - the fused
// phases share one id numbering, so skipping a fresh() here would renumber
// every later name.
void MSLEmitter::dotPoolPtrs(msl::Block &body, tt::DotOp op,
                             const DotPlan &plan, DotEmitCtx &dc) {
  checkDotPoolRegions(op, plan);
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
  if (plan.needAB && plan.aDirect) {
    // `devA = Abase + <tile origin> + warpId * (M/numWarps) * lda`: the first
    // row of the band this warp owns, at this trip's K block.
    int64_t bandRows = plan.M / plan.numWarps;
    dc.devALda = dmaRowStride(*plan.aDirect);
    dc.devA = fresh();
    // The IV may count trips rather than K elements, so the per-trip advance is
    // rescaled by the loop's step.
    DirectStage stat = *plan.aDirect;
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(op->getParentOp())) {
      APInt stepVal;
      if (matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) &&
          stepVal.getSExtValue() > 0 && stat.ptrDeltaLit)
        stat.ptrDeltaLit =
            *stat.ptrDeltaLit * stat.cols / stepVal.getSExtValue();
    }
    msl::Expr *origin = dmaTileOrigin(stat, dotDmaTripVar(op));
    msl::Expr *base = origin;
    if (plan.kind != DotPlan::Kind::Direct) {
      msl::Expr *bandOff = ctx.paren(ctx.binary(
          B::Mul,
          ctx.paren(ctx.binary(B::Mul, ctx.var(warpId), ctx.i32lit(bandRows))),
          dc.devALda));
      base = ctx.binary(B::Add, origin, bandOff);
    }
    body.push_back(
        ctx.declStmt(ctx.ptr(ctx.named(dc.opScalar), msl::AddrSpace::Device),
                     dc.devA, base));
  }
  if (plan.needAB) {
    if (!plan.aDirect)
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
                     ctx.paren(ctx.binary(B::Mul, ctx.var(fusedDot.dmaParity),
                                          ctx.i32lit(plan.stagedB / eb))))));
    } else {
      dc.tgBCur = dc.tgB;
    }
  }
  // A direct, unmasked store never reaches threadgroup memory, so scanPool
  // reserves nothing; naming a pool that was never declared emits a cast of
  // nothing.
  if (plan.needC && !poolBuf.empty())
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
  bool wholeBand = r0 == 0 && r1 >= cTy.getShape()[cTy.getRank() - 2];
  for (int r = 0, nRes = ids.size(); r < nRes; ++r) {
    // A register reaches a fixed set of rows, so one outside this band can only
    // produce a false guard.
    bool regInBand = false;
    if (!wholeBand) {
      int32_t lo = 0, hi = 0;
      layout.coordRange(cTy, r, rowDim, lo, hi);
      if (hi < r0 || lo >= r1)
        continue;
      regInBand = lo >= r0 && hi < r1;
    }
    std::string b = base[base.size() == 1 ? 0 : r];
    // ((rowExpr - r0) * N + colExpr)
    msl::Expr *bandOff = ctx.paren(
        ctx.add(ctx.mul(coordMinus(cTy, r, rowDim, r0), ctx.i32lit(N)),
                layout.layoutCoordExpr(cTy, r, colDim)));
    // (rowExpr >= r0 && rowExpr < r1), needed only when the register straddles
    // the band edge: one wholly inside it can never fail, and one wholly
    // outside was skipped above.
    msl::Expr *guard = nullptr;
    if (!wholeBand && !regInBand)
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
  // redundant simdgroup_loads each. Caching them across a whole warp block
  // removes those reloads but keeps nT*kT B fragments live at once; past a
  // handful of live fragments that costs far more than the reloads it saves,
  // so the column axis is walked in chunks and the cache spans one chunk.
  llvm::DenseMap<std::pair<int64_t, int64_t>, std::string> aFrag, bFrag;
  std::map<std::string, std::string> bFragExpr, aFragExpr;
  auto clearFragCache = [&]() {
    aFrag.clear();
    bFrag.clear();
    bFragExpr.clear();
    aFragExpr.clear();
  };
  // Columns per cache scope. Every disjointC arm asks here so the bound is
  // decided once; the accumulators those arms build are short-lived (declared,
  // filled and stored inside the chunk), so the live set is chunk*kT B
  // fragments plus the chunk's accumulators.
  const int64_t colChunk = dotColChunk(nT, kT);
  auto loadAFrag = [&](int64_t mi, int64_t ki, StringRef fa, msl::Block &into) {
    if (plan.aFrag) {
      for (int e = 0; e < 2; ++e)
        into.push_back(ctx.assignStmt(
            ctx.subscript(
                ctx.member(ctx.var(fa), msl::builtin::sg::ThreadElements),
                ctx.i32lit(e)),
            ctx.var(dc.aNames[2 * ki + e])));
      return;
    }
    if (dc.devA.empty()) {
      into.push_back(sgLoad(fa, dc.tgA, mi * 8 * lda + ki * 8, lda));
      return;
    }
    msl::Expr *off = ctx.paren(
        ctx.add(ctx.paren(ctx.binary(B::Mul, ctx.i32lit(mi * 8), dc.devALda)),
                ctx.i32lit(ki * 8)));
    into.push_back(ctx.exprStmt(ctx.call(
        msl::builtin::sg::Load,
        {ctx.var(fa), ctx.binary(B::Add, ctx.var(dc.devA), off), dc.devALda})));
  };
  auto fragMMABand = [&](StringRef miKey, msl::Expr *miExpr, int64_t ni,
                         StringRef acc, msl::Block &into) {
    for (int64_t ki = 0; ki < kT; ++ki) {
      std::string key = miKey.str() + ":" + std::to_string(ki);
      auto &fa = aFragExpr[key];
      if (fa.empty()) {
        fa = fresh();
        into.push_back(fragDecl(dc.opFrag, fa));
        msl::Expr *off = ctx.paren(ctx.add(
            ctx.paren(ctx.binary(
                B::Mul, ctx.paren(ctx.mul(miExpr, ctx.i32lit(8))), dc.devALda)),
            ctx.i32lit(ki * 8)));
        into.push_back(ctx.exprStmt(
            ctx.call(msl::builtin::sg::Load,
                     {ctx.var(fa), ctx.binary(B::Add, ctx.var(dc.devA), off),
                      dc.devALda})));
      }
      auto &fb = bFrag[{ki, ni}];
      if (fb.empty()) {
        fb = fresh();
        into.push_back(fragDecl(dc.opFrag, fb));
        into.push_back(bFragLoad(dc, ki, nullptr, ni, fb, ldb));
      }
      into.push_back(sgMultiplyAccumulate(acc, fa, fb));
    }
  };
  auto fragMMA = [&](int64_t mi, int64_t ni, StringRef acc, msl::Block &into) {
    for (int64_t ki = 0; ki < kT; ++ki) {
      auto &fa = aFrag[{mi, ki}];
      if (fa.empty()) {
        fa = fresh();
        into.push_back(fragDecl(dc.opFrag, fa));
        loadAFrag(mi, ki, fa, into);
      }
      auto &fb = bFrag[{ki, ni}];
      if (fb.empty()) {
        fb = fresh();
        into.push_back(fragDecl(dc.opFrag, fb));
        into.push_back(bFragLoad(dc, ki, nullptr, ni, fb, ldb));
      }
      into.push_back(sgMultiplyAccumulate(acc, fa, fb));
    }
  };
  // A whole cache window's MMAs with the k-steps rolled into a real loop: the
  // window's fragments are k-uniform apart from operand offsets affine in ki,
  // so one body under a counter is the same program with kT times fewer
  // fragment allocas. The AGX backend's SROA throws bad_alloc past roughly ten
  // thousand of those in one function.
  struct WinSlot {
    int64_t mi, ni;
    std::string acc;
  };
  auto rollableWindow = [&](ArrayRef<WinSlot> ws) {
    if (kT <= 1 || plan.aFrag || ws.empty())
      return false;
    llvm::DenseSet<int64_t> mis, nis;
    for (auto &s : ws) {
      mis.insert(s.mi);
      nis.insert(s.ni);
    }
    return rollKSteps(mis.size() + nis.size());
  };
  auto fragMMAWindowRolled = [&](ArrayRef<WinSlot> ws, msl::Block &into) {
    std::string kv = fresh();
    msl::Block inner;
    DenseMap<int64_t, std::string> fa, fb;
    for (auto &s : ws) {
      if (!fa.count(s.mi)) {
        std::string n = fresh();
        fa[s.mi] = n;
        inner.push_back(fragDecl(dc.opFrag, n));
        if (dc.devA.empty()) {
          inner.push_back(sgLoadExpr(
              n, dc.tgA,
              ctx.paren(ctx.add(ctx.i32lit(s.mi * 8 * lda), ctx.var(kv))),
              lda));
        } else {
          msl::Expr *off = ctx.paren(ctx.add(
              ctx.paren(ctx.binary(B::Mul, ctx.i32lit(s.mi * 8), dc.devALda)),
              ctx.var(kv)));
          inner.push_back(ctx.exprStmt(ctx.call(
              msl::builtin::sg::Load,
              {ctx.var(n), ctx.binary(B::Add, ctx.var(dc.devA), off),
               dc.devALda})));
        }
      }
      if (!fb.count(s.ni)) {
        std::string n = fresh();
        fb[s.ni] = n;
        inner.push_back(fragDecl(dc.opFrag, n));
        inner.push_back(
            bFragLoad(dc, 0, nullptr, s.ni, n, ldb, ctx.var(kv)));
      }
    }
    for (auto &s : ws)
      inner.push_back(sgMultiplyAccumulate(s.acc, fa[s.mi], fb[s.ni]));
    into.push_back(ctx.forScope(
        ctx.declStmt(ctx.named("int"), kv, ctx.i32lit(0)),
        ctx.binary(B::Lt, ctx.var(kv), ctx.i32lit(kT * 8)),
        ctx.assignStmt(ctx.var(kv),
                       ctx.binary(B::Add, ctx.var(kv), ctx.i32lit(8))),
        std::move(inner)));
  };

  auto fragMMAExpr = [&](int64_t mi, StringRef niKey, msl::Expr *niExpr,
                         StringRef acc, msl::Block &into) {
    for (int64_t ki = 0; ki < kT; ++ki) {
      auto &fa = aFrag[{mi, ki}];
      if (fa.empty()) {
        fa = fresh();
        into.push_back(fragDecl(dc.opFrag, fa));
        loadAFrag(mi, ki, fa, into);
      }
      std::string key = std::to_string(ki) + ":" + niKey.str();
      auto &fb = bFragExpr[key];
      if (fb.empty()) {
        fb = fresh();
        into.push_back(fragDecl(dc.opFrag, fb));
        into.push_back(bFragLoad(dc, ki, niExpr, 0, fb, ldb));
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
    stageOperand(body, dc.tgA, dc.aStage, aStageTy, dc.aNames,
                 (bool)plan.aInPlace || plan.aDirect.has_value(),
                 rank == 3 ? llvm::function_ref<msl::Expr *(int)>(guardA)
                           : nullptr,
                 plan.aPad);
    stageOperand(
        body, dc.tgB, dc.bStage, bStageTy, dc.bNames, (bool)plan.bInPlace,
        rank == 3 ? llvm::function_ref<msl::Expr *(int)>(guardB) : nullptr,
        plan.bPad);
    barrier();

    if (plan.disjointC) {
      if (plan.aDirect && plan.mT % numWarps == 0 && nT <= numWarps &&
          nFrag % numWarps == 0 && Bd == 1) {
        int64_t mB = plan.mT / numWarps;
        std::string miKey = "(" + warpId + "*" + std::to_string(mB) + ")";
        for (int64_t n0 = 0; n0 < nT; n0 += colChunk) {
          clearFragCache();
          int64_t n1 = std::min<int64_t>(n0 + colChunk, nT);
          for (int64_t r = 0; r < mB; ++r) {
            msl::Expr *miExpr = ctx.paren(ctx.add(
                ctx.mul(ctx.var(warpId), ctx.i32lit(mB)), ctx.i32lit(r)));
            for (int64_t ni = n0; ni < n1; ++ni) {
              std::string acc = fresh();
              body.push_back(accFragDecl(dc.accFragTy, acc));
              fragMMABand(miKey + "+" + std::to_string(r), miExpr, ni, acc,
                          body);
              msl::Expr *off = ctx.paren(
                  ctx.add(ctx.paren(ctx.binary(
                              B::Mul, ctx.paren(ctx.mul(miExpr, ctx.i32lit(8))),
                              ctx.i32lit(N))),
                          ctx.i32lit(ni * 8)));
              body.push_back(ctx.exprStmt(ctx.call(
                  msl::builtin::sg::Store,
                  {ctx.var(acc), ctx.binary(B::Add, ctx.var(dc.tgC), off),
                   ctx.i32lit(N)})));
            }
          }
        }
      } else if (numWarps == nT && nFrag % numWarps == 0 && Bd == 1) {
        clearFragCache();
        std::string niKey = warpId;
        msl::Expr *niExpr = ctx.var(warpId);
        for (int64_t j = 0; j * numWarps < nFrag; ++j) {
          int64_t mi = (j * numWarps) / nT;
          std::string acc = fresh();
          body.push_back(accFragDecl(dc.accFragTy, acc));
          fragMMAExpr(mi, niKey, niExpr, acc, body);
          msl::Expr *off = ctx.paren(
              ctx.add(ctx.i32lit(mi * 8 * N), ctx.mul(niExpr, ctx.i32lit(8))));
          body.push_back(ctx.exprStmt(
              ctx.call(msl::builtin::sg::Store,
                       {ctx.var(acc), ctx.binary(B::Add, ctx.var(dc.tgC), off),
                        ctx.i32lit(N)})));
        }
      } else if (nT % numWarps == 0 && nFrag % numWarps == 0 && Bd == 1) {
        // Each warp owns nT/numWarps columns; chunk that axis, not nT.
        const int64_t cols = nT / numWarps;
        const int64_t cChunk = dotColChunk(cols, kT);
        for (int64_t c0 = 0; c0 < cols; c0 += cChunk) {
          clearFragCache();
          int64_t c1 = std::min<int64_t>(c0 + cChunk, cols);
          for (int64_t mi = 0; mi < plan.mT; ++mi)
            for (int64_t c = c0; c < c1; ++c) {
              std::string niKey =
                  "(" + warpId + "+" + std::to_string(c * numWarps) + ")";
              msl::Expr *niExpr =
                  ctx.paren(ctx.add(ctx.var(warpId), ctx.i32lit(c * numWarps)));
              std::string acc = fresh();
              body.push_back(accFragDecl(dc.accFragTy, acc));
              fragMMAExpr(mi, niKey, niExpr, acc, body);
              msl::Expr *off = ctx.paren(ctx.add(
                  ctx.i32lit(mi * 8 * N), ctx.mul(niExpr, ctx.i32lit(8))));
              body.push_back(ctx.exprStmt(ctx.call(
                  msl::builtin::sg::Store,
                  {ctx.var(acc), ctx.binary(B::Add, ctx.var(dc.tgC), off),
                   ctx.i32lit(N)})));
            }
        }
      } else {
        // f strides by numWarps, so a window of colChunk*numWarps consecutive
        // f values spans at most colChunk distinct columns.
        const int64_t fWindow = colChunk * numWarps;
        for (int64_t w = 0; w < numWarps; ++w) {
          msl::Block inner;
          for (int64_t f0 = w; f0 < nFrag; f0 += fWindow) {
            clearFragCache();
            SmallVector<WinSlot> ws;
            for (int64_t f = f0; f < std::min(f0 + fWindow, nFrag);
                 f += numWarps)
              ws.push_back({f / nT, f % nT, fresh()});
            bool roll = rollableWindow(ws);
            for (auto &s : ws)
              inner.push_back(accFragDecl(dc.accFragTy, s.acc));
            if (roll) {
              fragMMAWindowRolled(ws, inner);
            } else {
              for (auto &s : ws)
                fragMMA(s.mi, s.ni, s.acc, inner);
            }
            for (auto &s : ws)
              inner.push_back(
                  sgStore(s.acc, dc.tgC, s.mi * 8 * N + s.ni * 8, N));
          }
          warpIf(w, std::move(inner));
        }
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
    const int64_t fWindow = colChunk * numWarps;
    for (int64_t w = 0; w < numWarps; ++w) {
      msl::Block inner;
      for (int64_t f0 = w; f0 < nFrag; f0 += fWindow) {
        clearFragCache();
        SmallVector<WinSlot> ws;
        for (int64_t f = f0; f < std::min(f0 + fWindow, nFrag); f += numWarps)
          ws.push_back({f / nT, f % nT, accName(f / nT, f % nT)});
        if (rollableWindow(ws)) {
          fragMMAWindowRolled(ws, inner);
        } else {
          for (auto &s : ws)
            fragMMA(s.mi, s.ni, s.acc, inner);
        }
      }
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
  if (fusedDot.phase == FusedDotPhase::Decl)
    fusedDot.warpTilingMSplit = plan.aDirect.has_value();
  WarpTiling wt = planWarpTiling(
      plan.mT, nT, numWarps, nFrag, fragsPerWarp,
      fusedDot.warpTilingMSplit.value_or(plan.aDirect.has_value()));
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
    // single-buffered copy has only A's register scatter to hide behind.
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
                 (bool)plan.aInPlace || plan.aDirect.has_value() ||
                     plan.aFragOnly,
                 nullptr, plan.aPad);
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
      body.push_back(dmaBeginInto(fusedDot.dmaHandle, plan, dc, *bDma, bStageTy,
                                  ldb, dotDmaTripVar(op),
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
    // The AGX backend compiler runs SROA over every simdgroup fragment alloca
    // and throws bad_alloc past ~10k of them in one function, so above a
    // threshold the k-steps are rolled into a real loop instead of unrolled.
    // The body is k-uniform apart from the operand offsets, which are affine in
    // ki, so one body with a loop counter is the same program.
    auto emitOneKStep = [&](ArrayRef<Slot> slots, msl::Block &into,
                            int64_t ki, msl::Expr *kiOff) {
      DenseMap<int64_t, std::string> aFrag;
      std::map<std::string, std::string> bFrag;
      for (auto &pr : slots) {
        int64_t mi = pr.mi;
        const std::string &niKey = pr.niKey;
        if (!aFrag.count(mi)) {
          std::string fa = fresh();
          aFrag[mi] = fa;
          into.push_back(fragDecl(dc.opFrag, fa));
          msl::Expr *kTerm = kiOff ? kiOff : (msl::Expr *)ctx.i32lit(ki * 8);
          if (!dc.devA.empty()) {
            // devA already points at this warp's band, so the fragment row is
            // band-relative; the row stride is A's device pitch.
            int64_t bandFrags = plan.M / (numWarps * 8);
            int64_t rowInBand = bandFrags ? mi % bandFrags : mi;
            msl::Expr *off = ctx.paren(
                ctx.add(ctx.paren(ctx.binary(
                            B::Mul, ctx.i32lit(rowInBand * 8), dc.devALda)),
                        kTerm));
            into.push_back(ctx.exprStmt(ctx.call(
                msl::builtin::sg::Load,
                {ctx.var(fa), ctx.binary(B::Add, ctx.var(dc.devA), off),
                 dc.devALda})));
          } else if (kiOff) {
            into.push_back(sgLoadExpr(
                fa, dc.tgA,
                ctx.paren(ctx.add(ctx.i32lit(mi * 8 * lda), kTerm)), lda));
          } else {
            into.push_back(sgLoad(fa, dc.tgA, mi * 8 * lda + ki * 8, lda));
          }
        }
        if (!bFrag.count(niKey)) {
          std::string fb = fresh();
          bFrag[niKey] = fb;
          into.push_back(fragDecl(dc.opFrag, fb));
          into.push_back(bFragLoad(dc, ki, pr.niExpr, 0, fb, ldb, kiOff));
        }
      }
      for (auto [j, mn] : llvm::enumerate(slots)) {
        const std::string &acc = fusedDot.accNames[j];
        into.push_back(
            sgMultiplyAccumulate(acc, aFrag[mn.mi], bFrag[mn.niKey]));
      }
    };
    auto emitSlots = [&](ArrayRef<Slot> slots, msl::Block &into) {
      if (kT > 1 && rollKSteps(slots.size())) {
        std::string kv = fresh();
        msl::Block inner;
        emitOneKStep(slots, inner, 0, ctx.var(kv));
        into.push_back(ctx.forScope(
            ctx.declStmt(ctx.named("int"), kv, ctx.i32lit(0)),
            ctx.binary(B::Lt, ctx.var(kv), ctx.i32lit(kT * 8)),
            ctx.assignStmt(ctx.var(kv),
                           ctx.binary(B::Add, ctx.var(kv), ctx.i32lit(8))),
            std::move(inner)));
        return;
      }
      for (int64_t ki = 0; ki < kT; ++ki)
        emitOneKStep(slots, into, ki, nullptr);
    };
    if (wt.twoD) {
      // mi is a compile-time constant only within a warp row, so the 2D form
      // branches per warp-row and keeps ni as a warpId expression.
      int64_t wGridM = numWarps / wt.wGridN;
      // ni = (warpId % wGridN) * niCount + c
      auto niBase = [&]() -> msl::Expr * {
        if (wt.wGridN == 1)
          return ctx.i32lit(0);
        msl::Expr *col = ctx.paren(
            ctx.binary(B::Rem, ctx.var(warpId), ctx.i32lit(wt.wGridN)));
        return ctx.paren(ctx.mul(col, ctx.i32lit(wt.niCount)));
      };
      std::string colKey =
          wt.wGridN == 1 ? std::string("0")
                         : "(" + warpId + " % " + std::to_string(wt.wGridN) +
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
      // devA is band-relative, so every warp row emits identical addresses and
      // accumulator slots once the band is miCount fragments tall.
      int64_t bandFrags = plan.M / (numWarps * 8);
      bool rowInvariant = !dc.devA.empty() && bandFrags == wt.miCount;
      if (wGridM == 1 || rowInvariant) {
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
    if (d.baseOff) {
      msl::Type *et =
          d.narrowTo ? scalarType(d.narrowTo) : ctx.scalar(msl::Scalar::F32);
      std::string shifted = fresh();
      body.push_back(
          ctx.declStmt(ctx.ptr(et, msl::AddrSpace::Device), shifted,
                       ctx.binary(B::Add, ctx.var(base), uniform(d.baseOff))));
      base = shifted;
    }
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
    // The column a lane's element e occupies inside an 8x8 fragment is
    // `e | ((lane&1)<<1) | (((lane>>3)&1)<<2)` -- independent of the row (see
    // AppleMmaLayoutConversions.cpp). So a bias that is constant down a column
    // is uniform over a fragment's rows and costs two scalar loads per
    // fragment, not one per accumulator element.
    std::string biasLaneCol;
    if (d.biasPtr) {
      biasLaneCol = fresh();
      body.push_back(ctx.declStmt(
          ctx.scalar(msl::Scalar::I32), biasLaneCol,
          ctx.binary(
              B::Or,
              ctx.paren(ctx.binary(
                  B::Shl,
                  ctx.paren(ctx.binary(B::And, ctx.var(laneId), ctx.i32lit(1))),
                  ctx.i32lit(1))),
              ctx.paren(
                  ctx.binary(B::Shl,
                             ctx.paren(ctx.binary(
                                 B::And,
                                 ctx.paren(ctx.binary(B::Shr, ctx.var(laneId),
                                                      ctx.i32lit(3))),
                                 ctx.i32lit(1))),
                             ctx.i32lit(2))))));
    }
    // A pure elementwise epilogue applied to the fragment's two slots, so the
    // tile never leaves register layout on its way to device memory.
    auto elementwised = [&](msl::Block &blk, StringRef accName) -> std::string {
      if (d.elementwise.empty())
        return accName.str();
      std::string id = fresh();
      blk.push_back(ctx.declStmt(dc.accFragTy, id, ctx.var(accName)));
      Value accVal = d.elementwiseAcc;
      for (int e = 0; e < 2; ++e) {
        msl::Expr *slot = ctx.subscript(
            ctx.member(ctx.var(id), msl::builtin::sg::ThreadElements),
            ctx.i32lit(e));
        // The region is a DAG, so each member's result gets its own name and
        // operands resolve through this map rather than a single slot.
        DenseMap<Value, msl::Expr *> bound;
        bound[accVal] = slot;
        std::string last;
        for (Operation *op : d.elementwise) {
          // Operand order is preserved, so a non-commutative form (sub, div)
          // keeps its sides.
          auto resolve = [&](Value o) -> msl::Expr * {
            if (auto it = bound.find(o); it != bound.end())
              return it->second;
            return uniformSplatScalar(o);
          };
          msl::Expr *cur = resolve(op->getOperand(0));
          msl::Expr *rhs =
              op->getNumOperands() > 1 ? resolve(op->getOperand(1)) : nullptr;
          std::string nm = fresh();
          blk.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::F32), nm,
                                     scalarEpilogueExpr(op, cur, rhs)));
          bound[op->getResult(0)] = ctx.var(nm);
          last = nm;
        }
        blk.push_back(ctx.assignStmt(slot, ctx.var(last)));
      }
      return id;
    };
    // acc + bias[biasCol + ni*8 + laneCol + e], into a scratch fragment so the
    // accumulator stays clean for the fallback arm.
    // The bias is constant down a column, so its two values depend only on the
    // fragment's ni -- not on which row fragment consumes them. Load each once
    // and reuse it across every mi, or the same address is fetched once per
    // accumulator.
    DenseMap<int64_t, std::array<std::string, 2>> biasVal;
    // Cached names are only in scope within the block that declared them, so
    // each emission site starts fresh.
    auto resetBias = [&] { biasVal.clear(); };
    // `warpCol` is the emitting warp's own column offset when one block covers
    // every warp; the bias is indexed from its own tile origin, so it has to
    // carry the same offset the store does or every warp reads warp 0's slice.
    auto biased = [&](msl::Block &blk, StringRef accName, int64_t ni,
                      msl::Expr *warpCol) -> std::string {
      if (!d.biasPtr)
        return accName.str();
      auto it = biasVal.find(ni);
      if (it == biasVal.end()) {
        std::array<std::string, 2> vs;
        for (int e = 0; e < 2; ++e) {
          vs[e] = fresh();
          SmallVector<msl::Expr *> parts{ctx.var(scalarName(d.biasPtr)),
                                         ctx.var(scalarName(d.biasCol))};
          if (warpCol)
            parts.push_back(warpCol);
          parts.push_back(ctx.i32lit(ni * 8 + e));
          parts.push_back(ctx.var(biasLaneCol));
          msl::Expr *addr = ctx.addChain(parts);
          blk.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::F32), vs[e],
                                     ctx.deref(ctx.paren(addr))));
        }
        it = biasVal.insert({ni, vs}).first;
      }
      std::string id = fresh();
      blk.push_back(ctx.declStmt(dc.accFragTy, id, ctx.var(accName)));
      for (int e = 0; e < 2; ++e) {
        msl::Expr *slot = ctx.subscript(
            ctx.member(ctx.var(id), msl::builtin::sg::ThreadElements),
            ctx.i32lit(e));
        blk.push_back(ctx.assignStmt(
            slot, ctx.binary(B::Add, slot, ctx.var(it->second[e]))));
      }
      return id;
    };
    // Warp w's slot j sits at an offset affine in w, so one block parameterised
    // by warpId covers every warp. Emitting a guarded copy per warp instead
    // duplicates the whole fragment epilogue numWarps times for nothing.
    bool mergeWarps = wt.twoD || numWarps == 1;
    for (int64_t w = 0; w < numWarps && mergeWarps; ++w)
      for (int64_t j = 0; j < fragsPerWarp; ++j) {
        int64_t mi, ni, mi0, ni0;
        wt.frag(w, j, nT, numWarps, mi, ni);
        wt.frag(0, j, nT, numWarps, mi0, ni0);
        if (mi != mi0 + (w / wt.wGridN) * wt.miCount ||
            ni != ni0 + (w % wt.wGridN) * wt.niCount ||
            (mi * nT + ni >= nFrag) != (mi0 * nT + ni0 >= nFrag))
          mergeWarps = false;
      }
    msl::Block ifBody;
    if (mergeWarps) {
      msl::Block inner;
      msl::Expr *wRow = ctx.mul(
          ctx.paren(ctx.binary(B::Div, ctx.var(warpId), ctx.i32lit(wt.wGridN))),
          ctx.i32lit(wt.miCount * 8));
      msl::Expr *wCol = ctx.mul(
          ctx.paren(ctx.binary(B::Rem, ctx.var(warpId), ctx.i32lit(wt.wGridN))),
          ctx.i32lit(wt.niCount * 8));
      for (int64_t j = 0; j < fragsPerWarp; ++j) {
        int64_t mi, ni;
        wt.frag(0, j, nT, numWarps, mi, ni);
        if (mi * nT + ni >= nFrag)
          continue;
        msl::Expr *off =
            ctx.addChain({ctx.var(base),
                          ctx.mul(ctx.paren(ctx.addChain({ctx.var(rowB), wRow,
                                                          ctx.i32lit(mi * 8)})),
                                  uniform(d.ldc)),
                          ctx.paren(ctx.addChain(
                              {ctx.var(colB), wCol, ctx.i32lit(ni * 8)}))});
        std::string sv = narrowed(
            inner,
            elementwised(inner, biased(inner, fusedDot.accNames[j], ni, wCol)));
        inner.push_back(ctx.exprStmt(ctx.call(
            msl::builtin::sg::Store, {ctx.var(sv), off, uniform(d.ldc)})));
      }
      for (msl::Stmt *s : inner)
        ifBody.push_back(s);
    } else {
      for (int64_t w = 0; w < numWarps; ++w) {
        msl::Block inner;
        resetBias();
        for (int64_t j = 0; j < fragsPerWarp; ++j) {
          int64_t mi, ni;
          wt.frag(w, j, nT, numWarps, mi, ni);
          if (mi * nT + ni >= nFrag)
            continue;
          // simdgroup_store(acc, base + (rowB + mi*8)*ldc + (colB + ni*8),
          // ldc);
          msl::Expr *off = ctx.addChain(
              {ctx.var(base),
               ctx.mul(ctx.paren(ctx.add(ctx.var(rowB), ctx.i32lit(mi * 8))),
                       uniform(d.ldc)),
               ctx.paren(ctx.add(ctx.var(colB), ctx.i32lit(ni * 8)))});
          std::string sv = narrowed(
              inner, elementwised(inner, biased(inner, fusedDot.accNames[j], ni,
                                                nullptr)));
          inner.push_back(ctx.exprStmt(ctx.call(
              msl::builtin::sg::Store, {ctx.var(sv), off, uniform(d.ldc)})));
        }
        ifBody.push_back(ctx.ifScope(
            ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
            std::move(inner)));
      }
    }
    // The pool image of C is the plain row-major tile (that is what the
    // simdgroup_store above writes), so the ragged arm can stream it to device
    // itself under the same bounds the store's mask carries.
    // The ragged arm drains a fragment at a time using an offset affine in the
    // warp id, so it needs the same tiling property the merged fast arm does.
    bool raggedDirect = !d.alwaysFullTile && d.boundM && d.boundN && mergeWarps;
    if (raggedDirect) {
      directStoreRaggedHandled.insert(
          const_cast<DirectStore &>(d).store.getOperation());
      if (d.narrowOp)
        directStoreRaggedHandled.insert(d.narrowOp);
      if (d.cvt)
        directStoreRaggedHandled.insert(d.cvt);
      if (d.biasAdd)
        directStoreRaggedHandled.insert(d.biasAdd);
      if (d.biasCvt)
        directStoreRaggedHandled.insert(d.biasCvt);
      for (Operation *e : d.elementwise)
        directStoreRaggedHandled.insert(e);
    }
    // if (fullTileVar) { <ifBody> } else { <pool store + gather> }
    msl::Block elseBody;
    // Save `body` size to splice the pool path into elseBody instead.
    // Build pool store + gather into elseBody:
    {
      msl::Block &tgt = elseBody;
      // Draining one fragment at a time through a per-warp 8x8 slot needs
      // numWarps*64 elements of scratch instead of the whole MxN tile, which is
      // what otherwise sets the pool size and caps threadgroup residency.
      // The per-warp drain addresses a fragment as warp-offset + constant,
      // which is only the right mapping when the warp tiling is affine in the
      // warp id -- the same condition the merged fast arm needs.
      if (raggedDirect) {
        // The drain slots overlay the A/B staging, so every warp must be past
        // its last staging read before any slot write lands.
        tgt.push_back(ctx.hardBarrier(false));
        std::string slot = fresh();
        tgt.push_back(ctx.declStmt(
            ctx.ptr(ctx.scalar(msl::Scalar::F32), msl::AddrSpace::Threadgroup),
            slot,
            ctx.add(ctx.var(dc.tgC),
                    ctx.mul(ctx.var(warpId), ctx.i32lit(64)))));
        for (int64_t j = 0; j < fragsPerWarp; ++j) {
          int64_t mi, ni;
          wt.frag(0, j, nT, numWarps, mi, ni);
          if (mi * nT + ni >= nFrag)
            continue;
          // rows/cols this fragment owns, offset by the warp's own subblock
          msl::Expr *wRowOff =
              ctx.mul(ctx.paren(ctx.binary(B::Div, ctx.var(warpId),
                                           ctx.i32lit(wt.wGridN))),
                      ctx.i32lit(wt.miCount * 8));
          msl::Expr *wColOff =
              ctx.mul(ctx.paren(ctx.binary(B::Rem, ctx.var(warpId),
                                           ctx.i32lit(wt.wGridN))),
                      ctx.i32lit(wt.niCount * 8));
          std::string fr = fresh(), fc = fresh();
          tgt.push_back(ctx.declStmt(
              ctx.scalar(msl::Scalar::I32), fr,
              ctx.addChain({ctx.var(rowB), ctx.i32lit(mi * 8), wRowOff})));
          tgt.push_back(ctx.declStmt(
              ctx.scalar(msl::Scalar::I32), fc,
              ctx.addChain({ctx.var(colB), ctx.i32lit(ni * 8), wColOff})));
          // A fragment wholly inside the bounds stores straight to device and
          // one wholly outside is skipped; only a straddling fragment takes
          // the slot round-trip. The conditions are warp-uniform, not
          // threadgroup-uniform, so the straddle arm may only use simdgroup
          // barriers -- which suffice, since the slot is per-warp.
          msl::Expr *fullIn =
              ctx.binary(B::Le, ctx.paren(ctx.add(ctx.var(fc), ctx.i32lit(8))),
                         uniform(d.boundN));
          msl::Expr *anyIn = ctx.binary(B::Lt, ctx.var(fc), uniform(d.boundN));
          if (!directStoreRowNeverRagged(d, M)) {
            fullIn = ctx.binary(
                B::LAnd,
                ctx.binary(B::Le,
                           ctx.paren(ctx.add(ctx.var(fr), ctx.i32lit(8))),
                           uniform(d.boundM)),
                fullIn);
            anyIn = ctx.binary(
                B::LAnd, ctx.binary(B::Lt, ctx.var(fr), uniform(d.boundM)),
                anyIn);
          }
          msl::Block dir;
          resetBias();
          std::string dv =
              narrowed(dir, elementwised(dir, biased(dir, fusedDot.accNames[j],
                                                     ni, wColOff)));
          dir.push_back(ctx.exprStmt(ctx.call(
              msl::builtin::sg::Store,
              {ctx.var(dv),
               ctx.addChain({ctx.var(base),
                             ctx.mul(ctx.paren(ctx.var(fr)), uniform(d.ldc)),
                             ctx.var(fc)}),
               uniform(d.ldc)})));
          msl::Block drain;
          drain.push_back(ctx.simdBarrier());
          drain.push_back(ctx.exprStmt(ctx.call(
              msl::builtin::sg::Store,
              {ctx.var(fusedDot.accNames[j]), ctx.var(slot), ctx.lit("8")})));
          drain.push_back(ctx.simdBarrier());
          msl::Expr *fRow = ctx.var(fr);
          msl::Expr *fCol = ctx.var(fc);
          std::string e = fresh(), rr = fresh(), cc2 = fresh();
          std::string gr = fresh(), gc = fresh();
          msl::Block loop;
          loop.push_back(
              ctx.declStmt(ctx.scalar(msl::Scalar::I32), rr,
                           ctx.binary(B::Div, ctx.var(e), ctx.i32lit(8))));
          loop.push_back(
              ctx.declStmt(ctx.scalar(msl::Scalar::I32), cc2,
                           ctx.binary(B::Rem, ctx.var(e), ctx.i32lit(8))));
          loop.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), gr,
                                      ctx.add(fRow, ctx.var(rr))));
          loop.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I32), gc,
                                      ctx.add(fCol, ctx.var(cc2))));
          msl::Expr *val = ctx.subscript(
              ctx.var(slot),
              ctx.add(ctx.mul(ctx.var(rr), ctx.i32lit(8)), ctx.var(cc2)));
          if (d.biasPtr) {
            // The bias is indexed from its own tile origin, not from C's.
            msl::Expr *bCol = ctx.addChain(
                {ctx.var(scalarName(d.biasCol)), ctx.i32lit(ni * 8),
                 ctx.mul(ctx.paren(ctx.binary(B::Rem, ctx.var(warpId),
                                              ctx.i32lit(wt.wGridN))),
                         ctx.i32lit(wt.niCount * 8)),
                 ctx.var(cc2)});
            val = ctx.binary(B::Add, val,
                             ctx.deref(ctx.paren(ctx.add(
                                 ctx.var(scalarName(d.biasPtr)), bCol))));
          }
          // The fast arm folds the epilogue into the fragments; this arm reads
          // the raw accumulator out of the slot, so it applies the same region
          // scalar-side to land on the identical value. The accumulator is
          // bound to a name first: a DAG reads it more than once, and `val` is
          // an expression tree that would otherwise be recomputed per read.
          if (!d.elementwise.empty()) {
            std::string accNm = fresh();
            loop.push_back(
                ctx.declStmt(ctx.scalar(msl::Scalar::F32), accNm, val));
            val = ctx.var(accNm);
            DenseMap<Value, msl::Expr *> bound;
            bound[d.elementwiseAcc] = val;
            for (Operation *eop : d.elementwise) {
              auto resolve = [&](Value o) -> msl::Expr * {
                if (auto it = bound.find(o); it != bound.end())
                  return it->second;
                return uniformSplatScalar(o);
              };
              msl::Expr *lhs = resolve(eop->getOperand(0));
              msl::Expr *rhs = eop->getNumOperands() > 1
                                   ? resolve(eop->getOperand(1))
                                   : nullptr;
              msl::Expr *r = scalarEpilogueExpr(eop, lhs, rhs);
              if (!r)
                break;
              std::string nm = fresh();
              loop.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::F32), nm, r));
              bound[eop->getResult(0)] = ctx.var(nm);
              val = ctx.var(nm);
            }
          }
          if (d.narrowTo)
            val = ctx.cast(CS::CStyle, scalarType(d.narrowTo), val);
          msl::Block guarded;
          guarded.push_back(ctx.assignStmt(
              ctx.subscript(
                  ctx.var(base),
                  ctx.add(ctx.mul(ctx.var(gr), uniform(d.ldc)), ctx.var(gc))),
              val));
          msl::Expr *inb = ctx.binary(B::Lt, ctx.var(gc), uniform(d.boundN));
          if (!directStoreRowNeverRagged(d, M))
            inb = ctx.binary(B::LAnd,
                             ctx.binary(B::Lt, ctx.var(gr), uniform(d.boundM)),
                             inb);
          loop.push_back(ctx.ifScope(inb, std::move(guarded)));
          drain.push_back(ctx.forScope(
              ctx.declStmt(ctx.scalar(msl::Scalar::I32), e, ctx.var(laneId)),
              ctx.binary(B::Lt, ctx.var(e), ctx.i32lit(64)),
              ctx.assignStmt(ctx.var(e),
                             ctx.binary(B::Add, ctx.var(e), ctx.i32lit(32))),
              std::move(loop)));
          msl::Block offEdge;
          offEdge.push_back(ctx.ifScope(anyIn, std::move(drain)));
          tgt.push_back(
              ctx.ifElseScope(fullIn, std::move(dir), std::move(offEdge)));
        }
      } else {
        tgt.push_back(ctx.hardBarrier(false));
        if (mergeWarps) {
          msl::Expr *wOff = ctx.paren(
              ctx.add(ctx.mul(ctx.paren(ctx.binary(B::Div, ctx.var(warpId),
                                                   ctx.i32lit(wt.wGridN))),
                              ctx.i32lit(wt.miCount * 8 * N)),
                      ctx.mul(ctx.paren(ctx.binary(B::Rem, ctx.var(warpId),
                                                   ctx.i32lit(wt.wGridN))),
                              ctx.i32lit(wt.niCount * 8))));
          for (int64_t j = 0; j < fragsPerWarp; ++j) {
            int64_t mi, ni;
            wt.frag(0, j, nT, numWarps, mi, ni);
            if (mi * nT + ni >= nFrag)
              continue;
            tgt.push_back(ctx.exprStmt(
                ctx.call(msl::builtin::sg::Store,
                         {ctx.var(fusedDot.accNames[j]),
                          ctx.addChain({ctx.var(dc.tgC),
                                        ctx.i32lit(mi * 8 * N + ni * 8), wOff}),
                          ctx.lit(std::to_string(N))})));
          }
        } else {
          for (int64_t w = 0; w < numWarps; ++w) {
            msl::Block inner;
            for (int64_t j = 0; j < fragsPerWarp; ++j) {
              int64_t mi, ni;
              wt.frag(w, j, nT, numWarps, mi, ni);
              if (mi * nT + ni >= nFrag)
                continue;
              inner.push_back(sgStore(fusedDot.accNames[j], dc.tgC,
                                      mi * 8 * N + ni * 8, N));
            }
            tgt.push_back(ctx.ifScope(
                ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
                std::move(inner)));
          }
        }
        tgt.push_back(ctx.hardBarrier(false));
        readbackInto(tgt, 0, 0, M);
        tgt.push_back(ctx.hardBarrier(false));
      }
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
  // staging, so each band must be gathered before the next band's stores land
  // on it -- hence the barrier on both sides of every band.
  for (int64_t r0 = 0; r0 < M; r0 += plan.bandRows) {
    int64_t r1 = std::min<int64_t>(r0 + plan.bandRows, M);
    barrier();
    for (int64_t w = 0; w < numWarps; ++w) {
      msl::Block inner;
      for (int64_t j = 0; j < fragsPerWarp; ++j) {
        int64_t mi, ni;
        wt.frag(w, j, nT, numWarps, mi, ni);
        if (mi * nT + ni >= nFrag)
          continue;
        if (mi * 8 < r0 || mi * 8 >= r1)
          continue;
        inner.push_back(sgStore(fusedDot.accNames[j], dc.tgC,
                                (mi * 8 - r0) * N + ni * 8, N));
      }
      body.push_back(ctx.ifScope(
          ctx.binary(B::Eq, ctx.var(warpId), ctx.lit(std::to_string(w))),
          std::move(inner)));
    }
    barrier();
    readbackInto(body, 0, r0, r1);
  }
  barrier();
  return true;
}

// The (a >= lo0 && a < hi0 && b >= lo1 && b < hi1) predicate guarding one
// staged register, with each half dropped when the register's reachable
// coordinates already satisfy it. Sets `dead` when they cannot, and returns
// null for a predicate that is wholly redundant.
msl::Expr *MSLEmitter::panelStageGuard(RankedTensorType ty, int reg,
                                       StringAttr dimA, int64_t lo0,
                                       int64_t hi0, StringAttr dimB,
                                       int64_t lo1, int64_t hi1, bool &dead) {
  SmallVector<msl::Expr *> terms;
  auto axis = [&](StringAttr d, int64_t lo, int64_t hi) {
    int32_t cl = 0, ch = 0;
    layout.coordRange(ty, reg, d, cl, ch);
    if (ch < lo || cl >= hi) {
      dead = true;
      return;
    }
    if (cl < lo)
      terms.push_back(
          ctx.binary(B::Ge, layout.layoutCoordExpr(ty, reg, d), ctx.i32lit(lo)));
    if (ch >= hi)
      terms.push_back(
          ctx.binary(B::Lt, layout.layoutCoordExpr(ty, reg, d), ctx.i32lit(hi)));
  };
  dead = false;
  axis(dimA, lo0, hi0);
  if (dead)
    return nullptr;
  axis(dimB, lo1, hi1);
  if (dead || terms.empty())
    return nullptr;
  return ctx.paren(ctx.chain(B::LAnd, terms));
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
  DotPanel pan = dotPanelPlan(M, N, K, elemBytes, accBytes);
  const int64_t mp = pan.mp, np = pan.np, kp = pan.kp;
  int64_t aPanelBytes = pan.aBytes, bPanelBytes = pan.bBytes;
  auto aOut = llvm::to_vector(ttg::toLinearLayout(aStageTy).getOutDimNames());
  StringAttr aRowDim = aOut[rank - 2], aColDim = aOut[rank - 1];
  auto bOut = llvm::to_vector(ttg::toLinearLayout(bStageTy).getOutDimNames());
  StringAttr bColDim = bOut[rank - 1], bRowDim = bOut[rank - 2];
  int nRes = regCount(op.getResult());

  auto tgPtr = [&](StringRef s) {
    return ctx.ptr(ctx.named(s), msl::AddrSpace::Threadgroup);
  };
  std::string pA = fresh(), pB = fresh(), pC = fresh();
  checkPoolRegions(op, {{"A", 0, aPanelBytes},
                        {"B", aPanelBytes, bPanelBytes},
                        {"C", aPanelBytes + bPanelBytes, pan.cBytes}});
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
      for (int64_t n0 = 0; n0 < N; n0 += np) {
        int64_t n1 = std::min<int64_t>(n0 + np, N), npCur = n1 - n0;
        int64_t pmT = mpCur / 8, pnT = npCur / 8;
        for (int64_t k0 = 0; k0 < K; k0 += kp) {
          int64_t k1 = std::min<int64_t>(k0 + kp, K), kpCur = k1 - k0;
          barrier();
          for (int r = 0; r < nARegs; ++r) {
            // (row >= m0 && row < m1 && col >= k0 && col < k1)
            bool dead = false;
            msl::Expr *guard = panelStageGuard(aStageTy, r, aRowDim, m0, m1,
                                               aColDim, k0, k1, dead);
            if (dead)
              continue;
            if (rank == 3) {
              msl::Expr *batchEq = ctx.binary(
                  B::Eq, layout.batchCoordExpr(aStageTy, r), ctx.i32lit(bi));
              guard = guard ? ctx.paren(ctx.binary(B::LAnd, batchEq, guard))
                            : ctx.paren(batchEq);
            }
            // ((row - m0) * kpCur + (col - k0))
            msl::Expr *off =
                ctx.paren(ctx.add(ctx.mul(coordMinus(aStageTy, r, aRowDim, m0),
                                          ctx.i32lit(kpCur)),
                                  coordMinus(aStageTy, r, aColDim, k0)));
            msl::Stmt *asn = ctx.assignStmt(ctx.subscript(ctx.var(pA), off),
                                            ctx.var(dc.aNames[r]));
            body.push_back(guard ? ctx.compactIfBare(guard, asn) : asn);
          }
          {
            barrier();
            for (int r = 0; r < nBRegs; ++r) {
              // (col >= n0 && col < n1 && row >= k0 && row < k1)
              bool dead = false;
              msl::Expr *guard = panelStageGuard(bStageTy, r, bColDim, n0, n1,
                                                 bRowDim, k0, k1, dead);
              if (dead)
                continue;
              if (rank == 3) {
                msl::Expr *batchEq = ctx.binary(
                    B::Eq, layout.batchCoordExpr(bStageTy, r), ctx.i32lit(bi));
                guard = guard ? ctx.paren(ctx.binary(B::LAnd, batchEq, guard))
                              : ctx.paren(batchEq);
              }
              // ((row - k0) * npCur + (col - n0))
              msl::Expr *off = ctx.paren(
                  ctx.add(ctx.mul(coordMinus(bStageTy, r, bRowDim, k0),
                                  ctx.i32lit(npCur)),
                          coordMinus(bStageTy, r, bColDim, n0)));
              msl::Stmt *asn = ctx.assignStmt(ctx.subscript(ctx.var(pB), off),
                                              ctx.var(dc.bNames[r]));
              body.push_back(guard ? ctx.compactIfBare(guard, asn) : asn);
            }
            barrier();
            int64_t pnFrag = pmT * pnT;
            int64_t pWarps = numWarps > pnFrag ? pnFrag : numWarps;
            // Under this predicate mi is warp-invariant and ni differs by
            // exactly w, so one warpId-relative block covers every warp.
            bool affineWarps = (pnFrag % pWarps == 0) && (pnT % pWarps == 0);
            for (int64_t w = 0; w < (affineWarps ? 1 : pWarps); ++w) {
              msl::Block inner;
              // A's fragment depends only on (mi, ki) and B's on (ki, ni), so
              // walking the (f, ki) grid reloads each of them once per the axis
              // it does not vary along. Nothing writes pA/pB between the
              // barriers bracketing this block, so one load per address serves
              // every use.
              auto colOff = [&](int64_t ni, int64_t scale) -> msl::Expr * {
                msl::Expr *base = ctx.i32lit(ni * 8 * scale);
                if (!affineWarps)
                  return base;
                return ctx.paren(ctx.add(
                    base, ctx.mul(ctx.var(warpId), ctx.i32lit(8 * scale))));
              };
              auto accAddr = [&](int64_t mi, int64_t ni) -> msl::Expr * {
                return ctx.paren(
                    ctx.add(ctx.i32lit(mi * 8 * npCur), colOff(ni, 1)));
              };
              const int64_t kSteps = kpCur / 8;
              SmallVector<std::pair<int64_t, int64_t>> slots;
              for (int64_t f = w; f < pnFrag; f += pWarps)
                slots.push_back({f / pnT, f % pnT});
              if (kSteps > 1 && rollKSteps(slots.size())) {
                SmallVector<std::string> accs;
                for (auto [mi, ni] : slots) {
                  std::string acc = fresh();
                  accs.push_back(acc);
                  if (k0 == 0) {
                    inner.push_back(accFragDecl(
                        ctx.matrix(msl::MatrixType::Elem::Float), acc));
                  } else {
                    inner.push_back(fragDecl(
                        ctx.matrix(msl::MatrixType::Elem::Float), acc));
                    inner.push_back(
                        sgLoadExpr(acc, pC, accAddr(mi, ni), npCur));
                  }
                }
                std::string kv = fresh();
                msl::Block loopB;
                DenseMap<int64_t, std::string> fa, fb;
                for (auto [mi, ni] : slots) {
                  if (!fa.count(mi)) {
                    std::string n = fresh();
                    fa[mi] = n;
                    loopB.push_back(fragDecl(dc.opFrag, n));
                    loopB.push_back(
                        sgLoadExpr(n, pA,
                                   ctx.paren(ctx.add(ctx.i32lit(mi * 8 * kpCur),
                                                     ctx.var(kv))),
                                   kpCur));
                  }
                  if (!fb.count(ni)) {
                    std::string n = fresh();
                    fb[ni] = n;
                    loopB.push_back(fragDecl(dc.opFrag, n));
                    loopB.push_back(sgLoadExpr(
                        n, pB,
                        ctx.paren(ctx.add(
                            ctx.paren(ctx.mul(ctx.var(kv), ctx.i32lit(npCur))),
                            colOff(ni, 1))),
                        npCur));
                  }
                }
                for (auto [j, mn] : llvm::enumerate(slots))
                  loopB.push_back(sgMultiplyAccumulate(accs[j], fa[mn.first],
                                                       fb[mn.second]));
                inner.push_back(ctx.forScope(
                    ctx.declStmt(ctx.named("int"), kv, ctx.i32lit(0)),
                    ctx.binary(B::Lt, ctx.var(kv), ctx.i32lit(kpCur)),
                    ctx.assignStmt(ctx.var(kv), ctx.binary(B::Add, ctx.var(kv),
                                                           ctx.i32lit(8))),
                    std::move(loopB)));
                for (auto [j, mn] : llvm::enumerate(slots))
                  inner.push_back(sgStoreExpr(
                      accs[j], pC, accAddr(mn.first, mn.second), npCur));
              } else {
                DenseMap<int64_t, std::string> aCache, bCache;
                for (auto [mi, ni] : slots) {
                  std::string acc = fresh();
                  if (k0 == 0) {
                    inner.push_back(accFragDecl(
                        ctx.matrix(msl::MatrixType::Elem::Float), acc));
                  } else {
                    inner.push_back(fragDecl(
                        ctx.matrix(msl::MatrixType::Elem::Float), acc));
                    inner.push_back(
                        sgLoadExpr(acc, pC, accAddr(mi, ni), npCur));
                  }
                  for (int64_t ki = 0; ki < kSteps; ++ki) {
                    int64_t aOff = mi * 8 * kpCur + ki * 8;
                    std::string &fa = aCache[aOff];
                    if (fa.empty()) {
                      fa = fresh();
                      inner.push_back(fragDecl(dc.opFrag, fa));
                      inner.push_back(sgLoad(fa, pA, aOff, kpCur));
                    }
                    int64_t bOff = ki * 8 * npCur + ni * 8;
                    std::string &fb = bCache[bOff];
                    if (fb.empty()) {
                      fb = fresh();
                      inner.push_back(fragDecl(dc.opFrag, fb));
                      inner.push_back(sgLoadExpr(
                          fb, pB,
                          ctx.paren(ctx.add(ctx.i32lit(ki * 8 * npCur),
                                            colOff(ni, 1))),
                          npCur));
                    }
                    inner.push_back(sgMultiplyAccumulate(acc, fa, fb));
                  }
                  inner.push_back(sgStoreExpr(acc, pC, accAddr(mi, ni), npCur));
                }
              }
              if (affineWarps) {
                for (msl::Stmt *s : inner)
                  body.push_back(s);
                break;
              }
              body.push_back(ctx.ifScope(ctx.binary(B::Eq, ctx.var(warpId),
                                                    ctx.lit(std::to_string(w))),
                                         std::move(inner)));
            }
            barrier();
            for (int r = 0; k1 == K && r < nRes; ++r) {
              std::string base = dc.cInit[dc.cInit.size() == 1 ? 0 : r];
              // ((rowExpr - m0) * npCur + (colExpr - n0))
              msl::Expr *off = ctx.paren(ctx.add(
                  ctx.mul(coordMinus(cTy, r, dc.rowDim, m0), ctx.i32lit(npCur)),
                  coordMinus(cTy, r, dc.colDim, n0)));
              // (rowExpr >= m0 && rowExpr < m1 && colExpr >= n0 && colExpr <
              // n1)
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
                  guard, ctx.assignStmt(
                             ctx.var(dc.ids[r]),
                             ctx.binary(B::Add, ctx.subscript(ctx.var(pC), off),
                                        ctx.var(base)))));
            }
          }
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

// The simdgroup-matrix fragment sub-builders (sgFragType .. readbackValue)
// live in EmitMSLDotFragments.cpp.

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

// The tensor axis a broadcast index term varies along, or -1 when no
// expand_dims establishes it.
static int broadcastVaryingAxis(Value v) {
  while (v) {
    Operation *def = v.getDefiningOp();
    if (auto b = dyn_cast_or_null<tt::BroadcastOp>(def)) {
      v = b.getSrc();
      continue;
    }
    if (auto e = dyn_cast_or_null<tt::ExpandDimsOp>(def))
      return e.getAxis() == 1 ? 0 : 1;
    return -1;
  }
  return -1;
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
    // A column-major store spells the same shape with the axes swapped
    // (`m*1 + n*ldc`), so matching on which term carries the multiply would
    // bind rowBase to N and store the fragments transposed.
    int rowAxis = broadcastVaryingAxis(rowT),
        colAxis = broadcastVaryingAxis(colT);
    if (rowAxis == 1 || colAxis == 0)
      return false;
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
                                   UniformInt &boundM, UniformInt &boundN,
                                   Value &tileGuard) {
  auto conj = definingOp<arith::AndIOp>(m);
  if (!conj)
    return false;
  // bmm ands in a tile-uniform `idx_q < BATCH`, which guards the whole tile
  // rather than an axis; the row/col pair sits in the conjunction under it.
  auto peelUniform = [&](Value uni, Value rest) {
    auto sp = definingOp<tt::SplatOp>(uni);
    auto in = definingOp<arith::AndIOp>(rest);
    if (!sp || !in)
      return false;
    if (isa<RankedTensorType>(sp.getSrc().getType()))
      return false;
    tileGuard = sp.getSrc();
    conj = in;
    return true;
  };
  peelUniform(conj.getRhs(), conj.getLhs()) ||
      peelUniform(conj.getLhs(), conj.getRhs());
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

Value MSLEmitter::matchRowBroadcastBias(Value v, DirectStore &ds) {
  if (!v.hasOneUse())
    return Value();
  auto add = dyn_cast<arith::AddFOp>(*v.user_begin());
  if (!add)
    return Value();
  Value other = add.getLhs() == v ? add.getRhs() : add.getLhs();
  // The row broadcast can sit on either side of the load -- on the loaded
  // values (a 1 x N load broadcast down the rows) or on the pointer (a
  // broadcast pointer tensor loaded at full tile shape) -- and a layout convert
  // may wrap either. Both spellings denote the same tile, and neither op
  // changes a value, so both are peeled and elided along with the add.
  Operation *cvt = nullptr;
  auto peel = [&](Value x) {
    while (x) {
      if (auto c = definingOp<ttg::ConvertLayoutOp>(x)) {
        cvt = c;
        x = c.getSrc();
        continue;
      }
      if (auto b = definingOp<tt::BroadcastOp>(x)) {
        x = b.getSrc();
        continue;
      }
      break;
    }
    return x;
  };
  auto load = definingOp<tt::LoadOp>(peel(other));
  if (!load)
    return Value();
  auto lTy = dyn_cast<RankedTensorType>(load.getType());
  if (!lTy || !lTy.getElementType().isF32())
    return Value();
  auto ptr = definingOp<tt::AddPtrOp>(peel(load.getPtr()));
  if (!ptr)
    return Value();
  // One value per column replicated down the rows: whatever the broadcast
  // spelling, the addressed tensor must have a row extent of 1, or the bias is
  // not column-uniform and folding it per fragment column is wrong.
  auto ptrTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!ptrTy || ptrTy.getRank() != 2 || ptrTy.getShape()[0] != 1)
    return Value();
  auto splat = definingOp<tt::SplatOp>(peelBroadcast(ptr.getPtr()));
  if (!splat || !isa<BlockArgument>(splat.getSrc()))
    return Value();
  Value colBase = matchTileIndex(ptr.getOffset());
  if (!colBase)
    return Value();
  ds.biasPtr = splat.getSrc();
  ds.biasCol = colBase;
  ds.biasAdd = add;
  ds.biasCvt = cvt;
  return add.getResult();
}

// The scalar behind a tile-uniform operand: a splat of a scalar, or a splat
// constant. Null when the value varies across the tile.
msl::Expr *MSLEmitter::uniformSplatScalar(Value v) {
  if (!isa<RankedTensorType>(v.getType()))
    return nullptr;
  Value src = peelBroadcast(v);
  if (auto sp = definingOp<tt::SplatOp>(src)) {
    // A splat of a value the emitter bound dataless (convertLayout does this
    // for a dead dot-stage) has no register to name, so it is not a scalar the
    // caller can use.
    if (names(sp.getSrc()).empty())
      return nullptr;
    return ctx.var(scalarName(sp.getSrc()));
  }
  if (auto cst = src.getDefiningOp<arith::ConstantOp>())
    if (auto se = dyn_cast<SplatElementsAttr>(cst.getValue())) {
      auto f = dyn_cast<FloatAttr>(se.getSplatValue<Attribute>());
      if (!f)
        return nullptr;
      return floatLitExpr(f.getValue(), f.getType());
    }
  return nullptr;
}

// One elementwise epilogue op rendered on scalars. `a` and `b` are its operands
// in IR order, so a non-commutative form keeps its sides; `b` is null for the
// unary ops.
msl::Expr *MSLEmitter::scalarEpilogueExpr(Operation *op, msl::Expr *a,
                                          msl::Expr *b) {
  msl::Expr *cur = a;
  msl::Expr *rhs = b;
  if (!cur)
    return nullptr;
  if (isa<arith::AddFOp>(op) && rhs)
    return ctx.binary(B::Add, cur, rhs);
  if (isa<arith::SubFOp>(op) && rhs)
    return ctx.binary(B::Sub, cur, rhs);
  if (isa<arith::MulFOp>(op) && rhs)
    return ctx.binary(B::Mul, cur, rhs);
  if (isa<arith::DivFOp, tt::PreciseDivFOp>(op) && rhs)
    return ctx.binary(B::Div, cur, rhs);
  namespace bi = msl::builtin;
  StringRef n = op->getName().getStringRef();
  static const llvm::StringMap<StringRef> unary = {
      {"math.exp", bi::math::Exp},     {"math.exp2", bi::math::Exp2},
      {"math.log", bi::precise::Log},  {"math.sqrt", bi::math::Sqrt},
      {"math.rsqrt", bi::math::Rsqrt}, {"math.tanh", bi::precise::Tanh},
      {"math.erf", "tt_erf"},          {"math.absf", bi::math::Fabs},
      {"math.floor", bi::math::Floor}, {"math.ceil", bi::math::Ceil}};
  auto it = unary.find(n);
  if (it != unary.end())
    return ctx.call(it->second, {cur});
  return nullptr;
}

// True for a value that is the same for every element of the tile, so emission
// can read it once per fragment instead of per element.
static bool isTileUniform(Value v) {
  Value src = peelBroadcast(v);
  if (definingOp<tt::SplatOp>(src))
    return true;
  if (auto cst = src.getDefiningOp<arith::ConstantOp>())
    return isa<SplatElementsAttr>(cst.getValue());
  return false;
}

// An op scalarEpilogueExpr can render, in the accumulator's own layout. The two
// lists must not drift, or a matched op reaches emission with no expression.
static bool isElementwiseEpilogueOp(Operation *op, RankedTensorType accTy) {
  if (!op || op->getNumResults() != 1)
    return false;
  if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
           tt::PreciseDivFOp, math::ExpOp, math::Exp2Op, math::LogOp,
           math::SqrtOp, math::RsqrtOp, math::TanhOp, math::ErfOp, math::AbsFOp,
           math::FloorOp, math::CeilOp>(op))
    return false;
  auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resTy || resTy.getShape() != accTy.getShape() ||
      resTy.getEncoding() != accTy.getEncoding())
    return false;
  return true;
}

// Collect the elementwise region hanging off `acc`, in topological order.
//
// The region may be a DAG rather than a chain -- gelu reads the accumulator
// twice and rejoins -- but it must be closed: every operand of a member is
// either the accumulator, another member, or tile-uniform, and every member's
// result is consumed inside the region except the single one that leaves it.
// That keeps the whole thing expressible as arithmetic on a fragment's
// thread_elements(), with no cross-element or cross-lane dependency.
static bool collectElementwiseRegion(Value acc, RankedTensorType accTy,
                                     SmallVectorImpl<Operation *> &ordered,
                                     Value &result) {
  SmallVector<Operation *> worklist;
  SetVector<Operation *> members;
  for (Operation *u : acc.getUsers()) {
    if (!isElementwiseEpilogueOp(u, accTy))
      return false;
    worklist.push_back(u);
  }
  if (worklist.empty())
    return false;

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!members.insert(op))
      continue;
    for (Value o : op->getOperands()) {
      if (o == acc || isTileUniform(o))
        continue;
      Operation *def = o.getDefiningOp();
      if (!def || !isElementwiseEpilogueOp(def, accTy))
        return false;
      worklist.push_back(def);
    }
    for (Operation *u : op->getResult(0).getUsers())
      if (isElementwiseEpilogueOp(u, accTy))
        worklist.push_back(u);
  }

  // Exactly one member may escape the region, and it must escape exactly once.
  Operation *sink = nullptr;
  for (Operation *op : members) {
    bool escapes = false;
    for (Operation *u : op->getResult(0).getUsers())
      if (!members.contains(u))
        escapes = true;
    if (!escapes)
      continue;
    if (sink || !op->getResult(0).hasOneUse())
      return false;
    sink = op;
  }
  if (!sink)
    return false;

  // Topological order, so emission never reads a value it has not produced.
  DenseSet<Operation *> done;
  std::function<bool(Operation *)> visit = [&](Operation *op) -> bool {
    if (done.contains(op))
      return true;
    for (Value o : op->getOperands()) {
      if (o == acc || isTileUniform(o))
        continue;
      Operation *def = o.getDefiningOp();
      if (!def || !members.contains(def) || !visit(def))
        return false;
    }
    done.insert(op);
    ordered.push_back(op);
    return true;
  };
  for (Operation *op : members)
    if (!visit(op))
      return false;

  result = sink->getResult(0);
  return true;
}

std::optional<DirectStore> MSLEmitter::matchDirectStore(Value forResult) {
  Operation *site = forResult.getDefiningOp();
  auto rej = [&](StringRef why) {
    if (site)
      mslReject(site, "matchDirectStore", why);
    return std::nullopt;
  };
  // The direct path stores the raw MMA fragments, which hold only what this
  // loop accumulated. Anything the accumulator started from is folded in by the
  // pool readback, so a non-zero init (mm_plus_mm chains one loop's result into
  // the next) would be dropped silently.
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(forResult.getDefiningOp())) {
    Value init =
        forOp.getInitArgs()[cast<OpResult>(forResult).getResultNumber()];
    DenseElementsAttr initAttr;
    if (!matchPattern(init, m_Constant(&initAttr)) || !initAttr.isSplat())
      return rej("acc-init-not-zero");
    Attribute splat = initAttr.getSplatValue<Attribute>();
    auto fp = dyn_cast<FloatAttr>(splat);
    auto in = dyn_cast<IntegerAttr>(splat);
    if (!(fp && fp.getValue().isZero()) && !(in && in.getValue().isZero()))
      return rej("acc-init-not-zero");
  }
  // Several uses are fine when they all land inside one elementwise region --
  // gelu reads the accumulator twice and rejoins -- since the region is then
  // folded into the fragments and nothing else observes the raw accumulator.
  if (!forResult.hasOneUse()) {
    SmallVector<Operation *> probeOps;
    Value probeOut;
    auto accTy = dyn_cast<RankedTensorType>(forResult.getType());
    if (!accTy ||
        !collectElementwiseRegion(forResult, accTy, probeOps, probeOut))
      return rej("acc-not-single-use");
  }
  // The f32 accumulator may be narrowed to the output dtype before the layout
  // convert. simdgroup_store cannot narrow, so the fragment is converted
  // elementwise through thread_elements() and stored as a narrow fragment.
  Value chain = forResult;
  Type narrowTo;
  arith::TruncFOp narrowOp;
  DirectStore ds;
  // addmm's `acc + bias` sits between the accumulator and the output convert.
  // Folding it into the fragments is what keeps this store direct; without it
  // the accumulator's user is the add and the whole tile stages through the
  // pool. The fold reads the bias unmasked, which is sound only because it is
  // emitted under the full-tile predicate -- the ragged arm keeps the original
  // masked load.
  if (Value biased = matchRowBroadcastBias(chain, ds)) {
    // As for the accumulator itself, several uses are fine when they all land
    // inside one elementwise region -- addmm followed by gelu reads the biased
    // value twice and rejoins.
    if (!biased.hasOneUse()) {
      SmallVector<Operation *> probeOps;
      Value probeOut;
      auto biasedTy = dyn_cast<RankedTensorType>(biased.getType());
      if (!biasedTy ||
          !collectElementwiseRegion(biased, biasedTy, probeOps, probeOut))
        return rej("bias-add-not-single-use");
    }
    chain = biased;
  }
  // A pure elementwise epilogue keeps the tile in fragment layout: every op
  // maps one element to one element, so it runs on thread_elements() in place.
  // Operands other than the chain must be uniform across the tile (a splat or a
  // scalar constant), or the fragment would need a second per-element source.
  //
  // Only the unmasked store can carry it. A boundary-masked store keeps a
  // threadgroup fallback arm that streams the pool image written by the fast
  // arm, and that image is post-epilogue, so the ragged tile would have the
  // epilogue applied twice.
  // Whether the fold is legal cannot be decided here: it turns on the store's
  // bounds, which are only matched further down. The region is carried through
  // so the rest of the walk sees the store, and the fold is committed at the
  // end once raggedness is known.
  SmallVector<Operation *> regionOps;
  Value regionOut;
  Value regionAcc;
  if (auto accTy = dyn_cast<RankedTensorType>(chain.getType()))
    if (collectElementwiseRegion(chain, accTy, regionOps, regionOut)) {
      regionAcc = chain;
      chain = regionOut;
    }

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
  // bmm's batch term is uniform across the tile, so it belongs on the base
  // pointer; left in place neither term of the outer add matches a row or col.
  Value tileOff = ptr.getOffset();
  {
    if (auto outer = definingOp<arith::AddIOp>(tileOff)) {
      auto take = [&](const UniformInt &u) {
        if (!u)
          return false;
        ds.baseOff = u;
        return true;
      };
      if (UniformInt u = matchUniformInt(outer.getRhs()); take(u)) {
        tileOff = outer.getLhs();
      } else if (UniformInt u2 = matchUniformInt(outer.getLhs()); take(u2)) {
        tileOff = outer.getRhs();
      }
    }
  }
  if (rowOff) {
    Value cb = matchTileIndex(ptr.getOffset());
    auto mul = definingOp<arith::MulIOp>(peelBroadcastExpand(rowOff));
    if (!cb || !mul)
      return rej("split-offset-shape");
    if (broadcastVaryingAxis(rowOff) == 1 ||
        broadcastVaryingAxis(ptr.getOffset()) == 0)
      return rej("split-offset-axes-swapped");
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
  } else if (!matchRowMajorOffset(tileOff, rowBase, ldc, colBase)) {
    return rej("offset-not-row-major");
  }
  UniformInt boundM, boundN;
  if (store.getMask()) {
    if (!matchBoundaryMask(store.getMask(), rowBase, colBase, boundM, boundN,
                           ds.tileGuard))
      return rej("mask-not-boundary");
  }
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

  // The ragged fallback arm streams the pool image the fast arm wrote, and that
  // image is already post-epilogue, so a tile that can go ragged would apply
  // the epilogue twice. An unmasked store has no such arm, and a masked one
  // qualifies only when its bounds prove the tile is never ragged -- the same
  // proof the emission site uses to decide the arm is unreachable.
  if (!regionOps.empty()) {
    // A ragged tile is safe when the fallback arm drains the raw accumulator
    // itself and reapplies the region scalar-side; that arm needs both bounds.
    bool raggedHandles = ds.boundM && ds.boundN;
    bool neverRagged =
        !store.getMask() ||
        directStoreNeverRagged(ds, cTy.getShape()[0], cTy.getShape()[1]);
    if (!neverRagged && !raggedHandles)
      return rej("epilogue-store-may-be-ragged");
    ds.elementwise = regionOps;
    ds.elementwiseAcc = regionAcc;
  }
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
// True when the ragged C drain addresses a fragment as warp-offset plus a
// constant, which is what lets that arm stream the tile itself. Both the fold
// gate and the drain read this, so the two cannot disagree about which arm
// runs.
bool MSLEmitter::dotRaggedDrainAffine(tt::DotOp d) {
  auto cTy = dyn_cast<RankedTensorType>(d.getResult().getType());
  if (!cTy || cTy.getRank() != 2)
    return false;
  int64_t M = cTy.getShape()[0], N = cTy.getShape()[1];
  if (M % 8 || N % 8)
    return false;
  tt::LinearLayout ll = ttg::toLinearLayout(cTy);
  auto wd = StringAttr::get(d.getContext(), "warp");
  int64_t nw = ll.hasInDim(wd) ? ll.getInDimSize(wd) : 1;
  int64_t mT = M / 8, nT = N / 8, nFrag = mT * nT;
  if (nw > nFrag)
    nw = nFrag;
  if (nw == 1)
    return true;
  int64_t fragsPerWarp = (nFrag + nw - 1) / nw;
  WarpTiling wt = planWarpTiling(mT, nT, nw, nFrag, fragsPerWarp,
                                 dotADirect(d).has_value());
  if (!wt.twoD)
    return false;
  for (int64_t w = 0; w < nw; ++w)
    for (int64_t j = 0; j < fragsPerWarp; ++j) {
      int64_t mi, ni, mi0, ni0;
      wt.frag(w, j, nT, nw, mi, ni);
      wt.frag(0, j, nT, nw, mi0, ni0);
      if (mi != mi0 + (w / wt.wGridN) * wt.miCount ||
          ni != ni0 + (w % wt.wGridN) * wt.niCount ||
          (mi * nT + ni >= nFrag) != (mi0 * nT + ni0 >= nFrag))
        return false;
    }
  return true;
}

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
  // A loop induction variable is a multiple of the block when it starts on one
  // and steps by one: every trip's tile origin then lands on a block boundary.
  if (auto arg = dyn_cast<BlockArgument>(origin))
    if (auto forOp =
            dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp()))
      if (arg == forOp.getInductionVar()) {
        APInt lo, st;
        if (matchPattern(forOp.getLowerBound(), m_ConstantInt(&lo)) &&
            matchPattern(forOp.getStep(), m_ConstantInt(&st)))
          return lo.getSExtValue() % blk == 0 && st.getSExtValue() % blk == 0;
        return false;
      }
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

static bool axisNeverRagged(const UniformInt &bound, Value origin,
                            int64_t blk) {
  if (!bound || !bound.lit || blk <= 0)
    return false;
  if (*bound.lit % blk != 0)
    return false;
  return isMultipleOfBlock(origin, blk);
}

bool MSLEmitter::rowBoundNeverRagged(const DirectStage &ds, int64_t M) {
  if (!ds.rowMask)
    return true;
  auto cmp = definingOp<arith::CmpIOp>(peelBroadcastExpand(ds.rowMask));
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return false;
  Value origin = matchTileIndex(cmp.getLhs());
  if (!origin)
    return false;
  return axisNeverRagged(matchUniformInt(cmp.getRhs()), origin, M);
}

bool MSLEmitter::directStoreRowNeverRagged(const DirectStore &ds, int64_t M) {
  return axisNeverRagged(ds.boundM, ds.rowBase, M);
}

bool MSLEmitter::directStoreColNeverRagged(const DirectStore &ds, int64_t N) {
  return axisNeverRagged(ds.boundN, ds.colBase, N);
}

bool MSLEmitter::directStoreNeverRagged(const DirectStore &ds, int64_t M,
                                        int64_t N) {
  return directStoreRowNeverRagged(ds, M) && directStoreColNeverRagged(ds, N);
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
  // Emission may decline the direct path outright, so C still needs its pool
  // reservation whenever the batch offset is in play.
  if (ds->baseOff)
    return true;
  auto cTy = cast<RankedTensorType>(d.getResult().getType());
  int rk = cTy.getRank();
  return !directStoreNeverRagged(*ds, cTy.getShape()[rk - 2],
                                 cTy.getShape()[rk - 1]);
}

// A convert on the direct-store C chain never materialises its tile: the fast
// arm sends the fragments straight to device and the ragged arm drains them one
// fragment at a time. That covers both the accumulator's own relayout and the
// row-broadcast bias the fold consumes two scalars of per fragment column.
bool MSLEmitter::convertLayoutIsDirectStoreC(ttg::ConvertLayoutOp c) {
  auto fromDot = [&](Value v) -> std::optional<DirectStore> {
    if (auto tf = v.getDefiningOp<arith::TruncFOp>())
      v = tf.getOperand();
    scf::ForOp forOp;
    unsigned idx = 0;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!forOp)
        return std::nullopt;
      idx = arg.getArgNumber() - forOp.getNumInductionVars();
    } else if (auto res = dyn_cast<OpResult>(v)) {
      forOp = dyn_cast<scf::ForOp>(res.getOwner());
      if (!forOp)
        return std::nullopt;
      idx = res.getResultNumber();
    } else {
      return std::nullopt;
    }
    auto m = matchGemmDotLoop(forOp);
    if (!m || m->second != idx)
      return std::nullopt;
    return matchDirectStore(forOp.getResult(idx));
  };
  // acc (or truncf(acc)) -> convert -> store
  if (fromDot(c.getSrc()))
    return true;
  // bias -> convert -> addf(acc, .) : dead when that add is the folded bias
  for (Operation *u : c.getResult().getUsers()) {
    auto add = dyn_cast<arith::AddFOp>(u);
    if (!add)
      continue;
    Value other = add.getLhs() == c.getResult() ? add.getRhs() : add.getLhs();
    if (auto ds = fromDot(other))
      if (ds->biasAdd == add.getOperation())
        return true;
  }
  return false;
}

int64_t MSLEmitter::fusedGemmCCompactScratch(tt::DotOp d) {
  if (!fusedGemmCHasFallback(d))
    return 0;
  auto forOp = dyn_cast<scf::ForOp>(d->getParentOp());
  auto m = matchGemmDotLoop(forOp);
  auto ds = matchDirectStore(forOp.getResult(m->second));
  if (!ds || !ds->boundM || !ds->boundN)
    return 0;
  auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  return nw ? nw.getInt() * 64 : 0;
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
  SmallVector<tt::DotOp> dots;
  for (Operation &o : body->without_terminator())
    if (auto d = dyn_cast<tt::DotOp>(&o))
      dots.push_back(d);
  if (dots.empty())
    return std::nullopt;

  tt::DotOp found = dots.front();
  auto cArg = dyn_cast<BlockArgument>(found.getC());
  if (!cArg || cArg.getOwner() != body) {
    mslReject(op, "matchGemmDotLoop", "acc-not-iter-arg");
    return std::nullopt;
  }
  unsigned idx = cArg.getArgNumber();
  if (idx == 0)
    return std::nullopt; // arg 0 is the induction var
  unsigned iterIdx = idx - 1;

  tt::DotOp last = dots.back();
  for (size_t i = 1; i < dots.size(); ++i)
    if (dots[i].getC() != dots[i - 1].getResult()) {
      mslReject(op, "matchGemmDotLoop", "dots-not-one-acc-chain");
      return std::nullopt;
    }
  if (yield.getOperand(iterIdx) != last.getResult()) {
    mslReject(op, "matchGemmDotLoop", "yield-not-dot-result");
    return std::nullopt;
  }
  for (Operation *u : cArg.getUsers())
    if (u != found.getOperation()) {
      mslReject(op, "matchGemmDotLoop", "acc-extra-user");
      return std::nullopt;
    }
  for (size_t i = 0; i + 1 < dots.size(); ++i)
    for (Operation *u : dots[i].getResult().getUsers())
      if (u != dots[i + 1].getOperation()) {
        mslReject(op, "matchGemmDotLoop", "dot-result-extra-user");
        return std::nullopt;
      }
  for (Operation *u : last.getResult().getUsers())
    if (u != yield.getOperation()) {
      mslReject(op, "matchGemmDotLoop", "dot-result-extra-user");
      return std::nullopt;
    }
  for (tt::DotOp d : dots)
    if (d.getResult().getType() != last.getResult().getType()) {
      mslReject(op, "matchGemmDotLoop", "dot-acc-type-mismatch");
      return std::nullopt;
    }

  found = last;
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
  int64_t K = 0;
  for (tt::DotOp d : dots) {
    auto aTy = cast<RankedTensorType>(d.getA().getType());
    if (aTy.getElementType() != aElem) {
      mslReject(op, "matchGemmDotLoop", "elem-type-unsupported");
      return std::nullopt;
    }
    K = std::max(K, aTy.getShape()[1]);
  }
  if (M % 8 || N % 8 || K % 8) {
    mslReject(op, "matchGemmDotLoop", "MNK-not-multiple-of-8");
    return std::nullopt;
  }
  for (tt::DotOp d : dots)
    if (cast<RankedTensorType>(d.getA().getType()).getShape()[1] % 8) {
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
  if (stagedA && dotADirect(found, /*knownFusedAcc=*/true))
    stagedA = 0;
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
  int64_t liveOperandBytes = (stagedA == 0 && stagedB == 0)
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

// Whether the operand comes out of a local_alloc at all, ignoring which slot:
// the rotating slot has no memdescMap entry until the body walk reaches it.
bool MSLEmitter::dotOperandInLocalAlloc(Value operand, int64_t rows,
                                        int64_t cols) {
  ttg::LocalLoadOp ll = dotOperandLocalLoad(operand, rows, cols);
  if (!ll)
    return false;
  Value src = ll.getSrc();
  while (auto mi = src.getDefiningOp<ttg::MemDescIndexOp>())
    src = mi.getSrc();
  auto alloc = src.getDefiningOp<ttg::LocalAllocOp>();
  if (!alloc)
    return false;
  auto mt = dyn_cast<ttg::MemDescType>(ll.getSrc().getType());
  return mt && memdescStrides(mt).empty();
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

// A transpose changes neither the operand's tensor type nor its register linear
// layout, so a path deciding eligibility from the type alone cannot see it.
// Every A/B fast path must ask here rather than testing it privately.
bool MSLEmitter::dotOperandTransposed(tt::DotOp d, Value operand) {
  if (Value s = dotOperandConvertSource(d, operand))
    operand = s;
  auto tr = definingOp<tt::TransOp>(operand);
  if (!tr)
    return false;
  auto ord = tr.getOrder();
  return ord.size() == 2 && ord[0] == 1 && ord[1] == 0;
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

MSLEmitter::DotPanel MSLEmitter::dotPanelPlan(int64_t M, int64_t N, int64_t K,
                                              int64_t elemBytes,
                                              int64_t accBytes) {
  auto cost = [&](int64_t m, int64_t n, int64_t k) {
    DotPanel p;
    p.mp = m;
    p.np = n;
    p.kp = k;
    p.aBytes = m * k * elemBytes;
    p.bBytes = k * n * elemBytes;
    p.cBytes = m * n * accBytes;
    return p;
  };
  int64_t mp = M, np = N;
  while (cost(mp, np, K).bytes() > kTGResidentBudgetBytes) {
    if (mp >= np && mp > 8)
      mp -= 8;
    else if (np > 8)
      np -= 8;
    else if (mp > 8)
      mp -= 8;
    else
      break;
  }
  // Both operand extents are at their floor and the panel still overflows, so
  // the K extent is what does not fit. Halving it keeps the 8-multiple
  // alignment the MMA fragments need.
  int64_t kp = K;
  while (kp > 8 && cost(mp, np, kp).bytes() > kTGResidentBudgetBytes)
    kp /= 2;
  return cost(mp, np, kp);
}

bool MSLEmitter::dotNeedsPanel(int64_t M, int64_t N, int64_t K,
                               int64_t elemBytes, int64_t accBytes) {
  return M * K * elemBytes + K * N * elemBytes > kTGResidentBudgetBytes;
}

int64_t MSLEmitter::dotColChunk(int64_t nT, int64_t kT) {
  if (nT <= 1 || kT <= 0)
    return nT < 1 ? 1 : nT;
  int64_t chunk = kTargetLiveFrags / kT;
  // At chunk 1 the cache is cleared every column, so no A reuse survives.
  if (chunk < 2)
    chunk = 2;
  if (chunk > nT)
    chunk = nT;
  while (chunk > 1 && nT % chunk)
    --chunk;
  return chunk;
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
