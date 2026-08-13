#include "MSLLayoutExpr.h"

using namespace mlir;
using namespace mlir::triton::applegpu;

SmallVector<int32_t> LayoutExprBuilder::registerCoords(RankedTensorType rt,
                                                       int reg) {
  MLIRContext *ctx = rt.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  SmallVector<std::pair<StringAttr, int32_t>> ins;
  ins.push_back({kReg, reg});
  if (ll.hasInDim(kLane))
    ins.push_back({kLane, 0});
  if (ll.hasInDim(kWarp))
    ins.push_back({kWarp, 0});
  if (ll.hasInDim(kBlock))
    ins.push_back({kBlock, 0});
  auto outs = ll.apply(ins);
  SmallVector<int32_t> coords;
  for (auto &p : outs)
    coords.push_back(p.second);
  return coords;
}

msl::Expr *LayoutExprBuilder::layoutCoordExpr(RankedTensorType rt, int reg,
                                              StringAttr outDim) {
  MLIRContext *mctx = rt.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  if (!hoistId)
    return buildCoordExpr(ll, mctx, reg, outDim);

  // The emitted tree is a function of nothing but the bases this out-dim reads
  // out of each runtime in-dim, plus the register's constant fold. Keying on
  // those exact numbers makes two entries share a name only when they would
  // have printed the same text.
  SmallVector<int32_t> sig;
  auto kRegD = StringAttr::get(mctx, "register");
  int32_t constPart = 0;
  for (int b = 0, n = ll.getInDimSizeLog2(kRegD); b < n; ++b)
    if (reg & (1 << b))
      constPart ^= ll.getBasis(kRegD, b, outDim);
  sig.push_back(constPart);
  for (StringRef in : {"lane", "warp", "block"}) {
    auto dim = StringAttr::get(mctx, in);
    sig.push_back(-1);
    if (!ll.hasInDim(dim))
      continue;
    for (int b = 0, n = ll.getInDimSizeLog2(dim); b < n; ++b)
      sig.push_back(ll.getBasis(dim, b, outDim));
  }
  std::string key;
  for (int32_t v : sig)
    key += std::to_string(v) + ",";
  auto it = hoisted.find(key);
  if (it != hoisted.end())
    return ctx.var(it->second);

  msl::Expr *tree = buildCoordExpr(ll, mctx, reg, outDim);
  // A bare id or literal is already as short as a name for it would be.
  if (isa<msl::Literal>(tree) || isa<msl::VarRef>(tree))
    return tree;
  std::string name = "c" + std::to_string((*hoistId)++);
  msl::Stmt *d = ctx.declStmt(ctx.scalar(msl::Scalar::I32), name, tree);
  decls.push_back(d);
  llvm::StringRef saved = llvm::cast<msl::DeclStmt>(d)->name;
  hoisted[key] = saved;
  return ctx.var(saved);
}

msl::Expr *LayoutExprBuilder::buildCoordExpr(const tt::LinearLayout &ll,
                                             MLIRContext *mctx, int reg,
                                             StringAttr outDim) {
  auto kReg = StringAttr::get(mctx, "register");
  auto kLane = StringAttr::get(mctx, "lane");
  auto kWarp = StringAttr::get(mctx, "warp");
  auto kBlock = StringAttr::get(mctx, "block");

  SmallVector<msl::Expr *> terms;

  int32_t constPart = 0;
  for (int b = 0, n = ll.getInDimSizeLog2(kReg); b < n; ++b)
    if (reg & (1 << b))
      constPart ^= ll.getBasis(kReg, b, outDim);
  if (constPart != 0)
    terms.push_back(ctx.lit(std::to_string(constPart)));

  auto runtimeDim = [&](StringAttr in, StringRef idExpr) {
    if (!ll.hasInDim(in))
      return;
    int n = ll.getInDimSizeLog2(in);
    for (int b = 0; b < n;) {
      int32_t basis = ll.getBasis(in, b, outDim);
      if (basis == 0) {
        ++b;
        continue;
      }
      // A maximal run of consecutive bits whose basis is the identity
      // (basis(k) == 1<<k) collapses to one masked shift instead of a per-bit
      // (((id >> k) & 1) * basis) xor chain. The bases are powers of two, so
      // the xor over a disjoint run is an or, i.e. a contiguous bitfield of
      // idExpr.
      if (basis == (1 << b)) {
        int e = b;
        while (e < n && ll.getBasis(in, e, outDim) == (1 << e))
          ++e;
        int len = e - b;
        int32_t mask = ((1 << len) - 1) << b;
        // (idExpr & mask)  -- b is already the field's low bit, so no reshift.
        terms.push_back(ctx.paren(ctx.binary(msl::BinOp::And, ctx.var(idExpr),
                                             ctx.lit(std::to_string(mask)))));
        b = e;
        continue;
      }
      // (((idExpr >> b) & 1) * basis)
      msl::Expr *shifted = ctx.paren(
          ctx.binary(msl::BinOp::Shr, ctx.var(idExpr), ctx.i32lit(b)));
      msl::Expr *bit =
          ctx.paren(ctx.binary(msl::BinOp::And, shifted, ctx.lit("1")));
      terms.push_back(ctx.paren(
          ctx.binary(msl::BinOp::Mul, bit, ctx.lit(std::to_string(basis)))));
      ++b;
    }
  };
  runtimeDim(kLane, laneId);
  runtimeDim(kWarp, warpId);
  runtimeDim(kBlock, tgposId + ".x");

  if (terms.empty())
    return ctx.lit("0");
  msl::Expr *expr = terms[0];
  for (size_t i = 1; i < terms.size(); ++i)
    expr = ctx.paren(ctx.binary(msl::BinOp::Xor, expr, terms[i]));
  return expr;
}

void LayoutExprBuilder::coordRange(RankedTensorType rt, int reg,
                                   StringAttr outDim, int32_t &lo,
                                   int32_t &hi) {
  MLIRContext *mctx = rt.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto kReg = StringAttr::get(mctx, "register");

  int32_t constPart = 0;
  for (int b = 0, n = ll.getInDimSizeLog2(kReg); b < n; ++b)
    if (reg & (1 << b))
      constPart ^= ll.getBasis(kReg, b, outDim);

  int32_t varMask = 0;
  bool disjoint = true;
  for (StringRef in : {"lane", "warp", "block"}) {
    auto dim = StringAttr::get(mctx, in);
    if (!ll.hasInDim(dim))
      continue;
    for (int b = 0, n = ll.getInDimSizeLog2(dim); b < n; ++b) {
      int32_t basis = ll.getBasis(dim, b, outDim);
      if (basis & varMask)
        disjoint = false;
      varMask |= basis;
    }
  }
  // Overlapping bases make the reachable set a xor lattice rather than a plain
  // bitfield, so fall back to the whole dim.
  if (!disjoint || (constPart & varMask)) {
    lo = 0;
    hi = ll.getOutDimSize(outDim) - 1;
    return;
  }
  lo = constPart;
  hi = constPart | varMask;
}

msl::Expr *LayoutExprBuilder::layoutOffsetExpr(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outDim = *ll.getOutDimNames().begin();
  return layoutCoordExpr(rt, reg, outDim);
}

// Shared row-major fold over out-dims [lo, hi], using `coord(d)` for the
// per-dim coordinate expr and `stride[d]` accumulated from `shape`.
static msl::Expr *foldRowMajor(msl::MSLContext &ctx, int hi, int lo,
                               llvm::function_ref<msl::Expr *(int)> coord,
                               llvm::function_ref<int64_t(int)> shapeAt,
                               int64_t &stride) {
  stride = 1;
  msl::Expr *expr = nullptr;
  for (int d = hi; d >= lo; --d) {
    msl::Expr *c = coord(d);
    msl::Expr *term =
        stride == 1 ? c
                    : ctx.paren(ctx.binary(msl::BinOp::Mul, c,
                                           ctx.lit(std::to_string(stride))));
    expr = expr ? ctx.paren(ctx.binary(msl::BinOp::Add, expr, term)) : term;
    stride *= shapeAt(d);
  }
  return expr ? expr : ctx.lit("0");
}

msl::Expr *LayoutExprBuilder::flatTileOffset(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto shape = rt.getShape();
  int64_t stride;
  return foldRowMajor(
      ctx, (int)outNames.size() - 1, 0,
      [&](int d) { return layoutCoordExpr(rt, reg, outNames[d]); },
      [&](int d) { return shape[d]; }, stride);
}

msl::Expr *LayoutExprBuilder::sliceFlatOffset(RankedTensorType rt, int reg,
                                              int64_t rowPad) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto shape = rt.getShape();
  int last = (int)outNames.size() - 1;
  int lo = std::max<int>(0, last - 1);
  int64_t stride;
  return foldRowMajor(
      ctx, last, lo,
      [&](int d) { return layoutCoordExpr(rt, reg, outNames[d]); },
      [&](int d) { return d == last ? shape[d] + rowPad : shape[d]; }, stride);
}

msl::Expr *LayoutExprBuilder::batchCoordExpr(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  return layoutCoordExpr(rt, reg, outNames[0]);
}

msl::Expr *LayoutExprBuilder::transFlatOffset(RankedTensorType srcTy,
                                              ArrayRef<int32_t> perm,
                                              ArrayRef<int64_t> resShape,
                                              int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  int rank = outNames.size();
  int64_t stride;
  return foldRowMajor(
      ctx, rank - 1, 0,
      [&](int d) { return layoutCoordExpr(srcTy, reg, outNames[perm[d]]); },
      [&](int d) { return resShape[d]; }, stride);
}

uint64_t LayoutExprBuilder::coordKey(ArrayRef<int32_t> c) {
  uint64_t k = 0;
  for (int32_t v : c)
    k = k * 100003u + (uint32_t)v + 1;
  return k;
}
