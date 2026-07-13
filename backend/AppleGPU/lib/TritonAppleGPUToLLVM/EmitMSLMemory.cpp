// EmitMSLMemory.cpp - memory / offset-expression lowering (string + AST forms).
//
// Out-lined AST siblings (astXxx) of the offset/address string builders in
// MSLEmitter.h. They rebuild the same runtime address expression as typed
// msl::Expr nodes; emission still runs on the string path this layer, so text
// is unchanged.
//
// INVARIANT: the printer inserts no grouping parens. Wherever the string path
// wrapped a subexpression for precedence, the AST sibling inserts an explicit
// ctx.paren(...) so both forms print identically.

#include "MSLConstants.h"
#include "MSLEmitter.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

using namespace mlir;

namespace mlir::triton::applegpu {

//===----------------------------------------------------------------------===//
// Layout coordinate / offset
//===----------------------------------------------------------------------===//

msl::Expr *MSLEmitter::astLayoutCoordExpr(RankedTensorType rt, int reg,
                                          StringAttr outDim) {
  MLIRContext *mctx = rt.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
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
    for (int b = 0, n = ll.getInDimSizeLog2(in); b < n; ++b) {
      int32_t basis = ll.getBasis(in, b, outDim);
      if (basis == 0)
        continue;
      // (((idExpr >> b) & 1) * basis)
      msl::Expr *shifted = ctx.paren(
          ctx.binary(msl::BinOp::Shr, ctx.var(idExpr), ctx.i32lit(b)));
      msl::Expr *bit =
          ctx.paren(ctx.binary(msl::BinOp::And, shifted, ctx.lit("1")));
      msl::Expr *term = ctx.paren(
          ctx.binary(msl::BinOp::Mul, bit, ctx.lit(std::to_string(basis))));
      terms.push_back(term);
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

msl::Expr *MSLEmitter::astLayoutOffsetExpr(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outDim = *ll.getOutDimNames().begin();
  return astLayoutCoordExpr(rt, reg, outDim);
}

//===----------------------------------------------------------------------===//
// Threadgroup pool region
//===----------------------------------------------------------------------===//

// ((threadgroup sc*)base), base = poolBuf or (poolBuf + off). `sc` may be a
// scalar or an atomic type name, so the pointee is a NamedType.
msl::Expr *MSLEmitter::astPoolRegion(int64_t byteOffset, StringRef sc) {
  msl::Expr *base =
      byteOffset == 0
          ? static_cast<msl::Expr *>(ctx.var(poolBuf))
          : ctx.paren(ctx.binary(msl::BinOp::Add, ctx.var(poolBuf),
                                  ctx.lit(std::to_string(byteOffset))));
  msl::Type *ptr = ctx.ptr(ctx.named(sc), msl::AddrSpace::Threadgroup);
  return ctx.paren(ctx.cast(msl::Cast::Style::CStyle, ptr, base));
}

//===----------------------------------------------------------------------===//
// Row-major flat offsets
//===----------------------------------------------------------------------===//

// Shared row-major fold over out-dims [lo, hi], using `coord(d)` for the
// per-dim coordinate expr and `stride[d]` accumulated from `shape`.
static msl::Expr *
foldRowMajor(msl::MSLContext &ctx, int hi, int lo,
             llvm::function_ref<msl::Expr *(int)> coord,
             llvm::function_ref<int64_t(int)> shapeAt, int64_t &stride) {
  stride = 1;
  msl::Expr *expr = nullptr;
  for (int d = hi; d >= lo; --d) {
    msl::Expr *c = coord(d);
    msl::Expr *term =
        stride == 1
            ? c
            : ctx.paren(ctx.binary(msl::BinOp::Mul, c,
                                   ctx.lit(std::to_string(stride))));
    expr = expr ? ctx.paren(ctx.binary(msl::BinOp::Add, expr, term)) : term;
    stride *= shapeAt(d);
  }
  return expr ? expr : ctx.lit("0");
}

msl::Expr *MSLEmitter::astFlatTileOffset(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto shape = rt.getShape();
  int64_t stride;
  return foldRowMajor(
      ctx, (int)outNames.size() - 1, 0,
      [&](int d) { return astLayoutCoordExpr(rt, reg, outNames[d]); },
      [&](int d) { return shape[d]; }, stride);
}

msl::Expr *MSLEmitter::astSliceFlatOffset(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto shape = rt.getShape();
  int lo = std::max<int>(0, (int)outNames.size() - 2);
  int64_t stride;
  return foldRowMajor(
      ctx, (int)outNames.size() - 1, lo,
      [&](int d) { return astLayoutCoordExpr(rt, reg, outNames[d]); },
      [&](int d) { return shape[d]; }, stride);
}

msl::Expr *MSLEmitter::astBatchCoordExpr(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  return astLayoutCoordExpr(rt, reg, outNames[0]);
}

msl::Expr *MSLEmitter::astTransFlatOffset(RankedTensorType srcTy,
                                          ArrayRef<int32_t> perm,
                                          ArrayRef<int64_t> resShape, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  int rank = outNames.size();
  int64_t stride;
  return foldRowMajor(
      ctx, rank - 1, 0,
      [&](int d) { return astLayoutCoordExpr(srcTy, reg, outNames[perm[d]]); },
      [&](int d) { return resShape[d]; }, stride);
}

//===----------------------------------------------------------------------===//
// Memdesc element address
//===----------------------------------------------------------------------===//

msl::Expr *MSLEmitter::astMemdescElemAddr(const MemDescInfo &info,
                                          RankedTensorType tileTy, int reg) {
  msl::Expr *off;
  if (info.bufStrides.empty()) {
    off = astFlatTileOffset(tileTy, reg);
  } else {
    tt::LinearLayout ll = ttg::toLinearLayout(tileTy);
    auto outNames = llvm::to_vector(ll.getOutDimNames());
    off = nullptr;
    for (int d = 0; d < (int)outNames.size(); ++d) {
      msl::Expr *c = astLayoutCoordExpr(tileTy, reg, outNames[d]);
      int64_t s = info.bufStrides[d];
      msl::Expr *term =
          s == 1 ? c
                 : ctx.paren(ctx.binary(msl::BinOp::Mul, c,
                                        ctx.lit(std::to_string(s))));
      off = off ? ctx.paren(ctx.binary(msl::BinOp::Add, off, term)) : term;
    }
    if (!off)
      off = ctx.lit("0");
  }
  if (info.baseOffset == "0")
    return off;
  // baseOffset is still std::string on MemDescInfo this layer; bridge via raw.
  // The flip layer should convert baseOffset to Expr* and drop this ctx.raw.
  return ctx.paren(ctx.binary(msl::BinOp::Add, ctx.raw(info.baseOffset), off));
}

} // namespace mlir::triton::applegpu
