// EmitMSLMemory.cpp - memory / offset-expression lowering.
//
// AST builders (astXxx) that build the runtime offset/address expressions as
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
  if (!info.baseOffset)
    return off;
  return ctx.paren(ctx.binary(msl::BinOp::Add, info.baseOffset, off));
}

std::string MSLEmitter::layoutOffsetExpr(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outDim = *ll.getOutDimNames().begin();
  return layoutCoordExpr(rt, reg, outDim);
}

std::string MSLEmitter::layoutCoordExpr(RankedTensorType rt, int reg,
                                        StringAttr outDim) {
  MLIRContext *ctx = rt.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");

  SmallVector<std::string> terms;

  int32_t constPart = 0;
  for (int b = 0, n = ll.getInDimSizeLog2(kReg); b < n; ++b)
    if (reg & (1 << b))
      constPart ^= ll.getBasis(kReg, b, outDim);
  if (constPart != 0)
    terms.push_back(std::to_string(constPart));

  auto runtimeDim = [&](StringAttr in, StringRef idExpr) {
    if (!ll.hasInDim(in))
      return;
    for (int b = 0, n = ll.getInDimSizeLog2(in); b < n; ++b) {
      int32_t basis = ll.getBasis(in, b, outDim);
      if (basis == 0)
        continue;
      std::string bitExpr =
          "(((" + idExpr.str() + " >> " + std::to_string(b) + ") & 1) * " +
          std::to_string(basis) + ")";
      terms.push_back(bitExpr);
    }
  };
  runtimeDim(kLane, laneId);
  runtimeDim(kWarp, warpId);
  runtimeDim(kBlock, tgposId + ".x");

  if (terms.empty())
    return "0";
  std::string expr = terms[0];
  for (size_t i = 1; i < terms.size(); ++i)
    expr = "(" + expr + " ^ " + terms[i] + ")";
  return expr;
}

int64_t MSLEmitter::poolBudget() const {
  int64_t b = 32768 - liveTgBytes;
  return b < 0 ? 0 : b;
}

std::string MSLEmitter::poolRegion(int64_t byteOffset, StringRef sc) {
  std::string base = byteOffset == 0
                         ? poolBuf
                         : "(" + poolBuf + " + " + std::to_string(byteOffset) +
                               ")";
  return "((threadgroup " + sc.str() + "*)" + base + ")";
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
  int64_t elemBytes = bitsOf(e) / 8;
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
    auto st = cast<RankedTensorType>(c.getSrc().getType());
    poolBytes = std::max(poolBytes, tgScratchBytes(st, /*band2D=*/true));
  } else if (auto t = dyn_cast<tt::TransOp>(op)) {
    auto rt = cast<RankedTensorType>(t.getResult().getType());
    poolBytes = std::max(poolBytes, tgScratchBytes(rt, /*band2D=*/false));
  } else if (auto c = dyn_cast<tt::CatOp>(op)) {
    auto rt = cast<RankedTensorType>(c.getResult().getType());
    Type e = rt.getElementType();
    poolBytes = std::max(poolBytes, tileSize(rt) * (bitsOf(e) / 8));
  } else if (auto rs = dyn_cast<tt::ReshapeOp>(op)) {
    auto rt = cast<RankedTensorType>(rs.getResult().getType());
    poolBytes = std::max(poolBytes, tgScratchBytes(rt, /*band2D=*/false));
  } else if (auto g = dyn_cast<tt::GatherOp>(op)) {
    auto st = cast<RankedTensorType>(g.getSrc().getType());
    Type e = st.getElementType();
    poolBytes = std::max(poolBytes, tileSize(st) * (bitsOf(e) / 8));
  } else if (auto d = dyn_cast<tt::DotOp>(op)) {
    auto aTy = cast<RankedTensorType>(d.getA().getType());
    auto bTy = cast<RankedTensorType>(d.getB().getType());
    auto cTy = cast<RankedTensorType>(d.getResult().getType());
    int rk = cTy.getRank();
    int64_t M = cTy.getShape()[rk - 2];
    int64_t N = cTy.getShape()[rk - 1];
    int64_t Kd = aTy.getShape()[rk - 1];
    int64_t aBy = M * Kd * (bitsOf(aTy.getElementType()) / 8);
    int64_t bBy = Kd * N * (bitsOf(bTy.getElementType()) / 8);
    Type cE = cTy.getElementType();
    int64_t need;
    if (isa<IntegerType>(cE)) {
      need = aBy + bBy;
    } else {
      int64_t accBytes = 4;
      int64_t elemBytes = bitsOf(aTy.getElementType()) / 8;
      int64_t stagedA = aBy, stagedB = bBy;
      if (rk == 2 && aBy + bBy <= 32768) {
        if (dotOperandLocalLoad(d.getA(), M, Kd))
          stagedA = 0;
        if (dotOperandLocalLoad(d.getB(), Kd, N))
          stagedB = 0;
      }
      int64_t stagedAB = stagedA + stagedB;
      int64_t cFull = M * N * accBytes;
      if (stagedAB == aBy + bBy && dotNeedsPanel(M, N, Kd, elemBytes, accBytes)) {
        int64_t mp, np;
        dotPanelDims(M, N, Kd, elemBytes, accBytes, mp, np);
        need = mp * Kd * elemBytes + Kd * np * elemBytes + mp * np * accBytes;
      } else if (stagedAB + cFull <= poolBudget()) {
        need = stagedAB + cFull;
      } else {
        int64_t band = dotCBandRows(M, N, poolBudget(), accBytes);
        need = std::max(stagedAB, band * N * accBytes);
      }
    }
    poolBytes = std::max(poolBytes, need);
  } else if (auto r = dyn_cast<tt::ReduceOp>(op)) {
    auto st = cast<RankedTensorType>(r.getOperand(0).getType());
    tt::LinearLayout ll = ttg::toLinearLayout(st);
    auto kWarp = StringAttr::get(op->getContext(), "warp");
    if (ll.hasInDim(kWarp)) {
      int64_t nw = ll.getInDimSize(kWarp);
      int64_t bytes = 0;
      for (Value res : r.getResult())
        bytes += nw * 32 *
                 std::max<int64_t>(
                     1, bitsOf(elementScalarType(res.getType())) / 8);
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
            1, bitsOf(elementScalarType(ar.getResult().getType())) / 8);
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
        bytes += nw * 32 * (bitsOf(elementScalarType(res.getType())) / 8);
      poolBytes = std::max(poolBytes, bytes);
    }
  }
  for (Region &reg : op->getRegions())
    for (Block &blk : reg)
      for (Operation &o : blk)
        scanPool(&o);
}

std::string MSLEmitter::flatTileOffset(RankedTensorType rt, int reg) {
  MLIRContext *ctx = rt.getContext();
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto shape = rt.getShape();
  std::string expr;
  int64_t stride = 1;
  for (int d = (int)outNames.size() - 1; d >= 0; --d) {
    std::string c = layoutCoordExpr(rt, reg, outNames[d]);
    std::string term = stride == 1 ? c : ("(" + c + " * " +
                                          std::to_string(stride) + ")");
    expr = expr.empty() ? term : ("(" + expr + " + " + term + ")");
    stride *= shape[d];
  }
  (void)ctx;
  return expr.empty() ? "0" : expr;
}

std::string MSLEmitter::sliceFlatOffset(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto shape = rt.getShape();
  int lo = std::max<int>(0, (int)outNames.size() - 2);
  std::string expr;
  int64_t stride = 1;
  for (int d = (int)outNames.size() - 1; d >= lo; --d) {
    std::string c = layoutCoordExpr(rt, reg, outNames[d]);
    std::string term = stride == 1 ? c : ("(" + c + " * " +
                                          std::to_string(stride) + ")");
    expr = expr.empty() ? term : ("(" + expr + " + " + term + ")");
    stride *= shape[d];
  }
  return expr.empty() ? "0" : expr;
}

std::string MSLEmitter::batchCoordExpr(RankedTensorType rt, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(rt);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  return layoutCoordExpr(rt, reg, outNames[0]);
}

std::string MSLEmitter::transFlatOffset(RankedTensorType srcTy,
                                        ArrayRef<int32_t> perm,
                                        ArrayRef<int64_t> resShape, int reg) {
  tt::LinearLayout ll = ttg::toLinearLayout(srcTy);
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  int rank = outNames.size();
  std::string expr;
  int64_t stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    std::string c = layoutCoordExpr(srcTy, reg, outNames[perm[d]]);
    std::string term = stride == 1 ? c : ("(" + c + " * " +
                                          std::to_string(stride) + ")");
    expr = expr.empty() ? term : ("(" + expr + " + " + term + ")");
    stride *= resShape[d];
  }
  return expr.empty() ? "0" : expr;
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
  memdescMap[op.getResult()] = {parent.buf, base};
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
