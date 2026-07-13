// EmitMSLExpr.cpp - value-producing expression lowering (string + AST forms).
//
// Out-lined from MSLEmitter.h: the string helpers that emit an SSA-register
// decl for each value-producing op, plus their AST siblings (astXxxExpr) that
// build the same RHS as typed msl::Expr nodes. The AST builders do not drive
// output yet (Layer 2); later layers assign their result into a DeclStmt init.
//
// INVARIANT: the printer is dumb - it inserts no grouping parens. Wherever the
// string path wrapped a subexpression in parens for precedence, the matching
// AST builder inserts an explicit ctx.paren(...), so both forms print alike.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

//===----------------------------------------------------------------------===//
// Float literals
//===----------------------------------------------------------------------===//

std::string MSLEmitter::floatLit(const APFloat &v) {
  if (v.isInfinity())
    return v.isNegative() ? "(-INFINITY)" : "INFINITY";
  if (v.isNaN())
    return "NAN";
  char buf[32];
  snprintf(buf, sizeof(buf), "%.17g", v.convertToDouble());
  return buf;
}

std::string MSLEmitter::floatLit(const APFloat &v, StringRef sc) {
  std::string lit = floatLit(v);
  if (sc == "bfloat" || sc == "half")
    return sc.str() + "(" + lit + ")";
  return lit;
}

msl::Expr *MSLEmitter::astFloatLit(const APFloat &v, StringRef sc) {
  msl::Expr *lit = ctx.lit(floatLit(v));
  if (sc == "bfloat" || sc == "half")
    return ctx.call(sc, {lit});
  return lit;
}

//===----------------------------------------------------------------------===//
// Integer / float binary, shift, elementwise
//===----------------------------------------------------------------------===//

LogicalResult MSLEmitter::emitIntBinary(Operation *op) {
  const char *o = isa<arith::AddIOp>(op)   ? "+"
                  : isa<arith::SubIOp>(op) ? "-"
                  : isa<arith::MulIOp>(op) ? "*"
                  : isa<arith::DivSIOp, arith::DivUIOp>(op) ? "/"
                                                            : "%";
  Type resElem = elementScalarType(op->getResult(0).getType());
  if (auto it = dyn_cast<IntegerType>(resElem); it && it.getWidth() == 1) {
    Value res = op->getResult(0);
    auto &lhs = names(op->getOperand(0));
    auto &rhs = names(op->getOperand(1));
    int rc = regCount(res);
    SmallVector<std::string> ids;
    for (int r = 0; r < rc; ++r) {
      std::string id = fresh();
      const std::string &a = lhs[lhs.size() == 1 ? 0 : r];
      const std::string &b = rhs[rhs.size() == 1 ? 0 : r];
      os << ind() << "bool " << id << " = (bool)((((int)" << a << ") " << o
         << " ((int)" << b << ")) & 1);\n";
      ids.push_back(id);
    }
    valMap[res] = ids;
    return success();
  }
  std::string sc =
      mslScalarType(elementScalarType(op->getResult(0).getType()));
  std::string opCast;
  if (isa<arith::DivUIOp, arith::RemUIOp>(op)) {
    opCast = mslUnsignedType(elementScalarType(op->getResult(0).getType()));
    sc = opCast;
  }
  return emitElementwise(op, o, sc, opCast);
}

LogicalResult MSLEmitter::emitFloatBinary(Operation *op) {
  const char *o = isa<arith::AddFOp>(op)   ? "+"
                  : isa<arith::SubFOp>(op) ? "-"
                  : isa<arith::MulFOp>(op) ? "*"
                                           : "/";
  std::string sc =
      mslScalarType(elementScalarType(op->getResult(0).getType()));
  return emitElementwise(op, o, sc);
}

LogicalResult MSLEmitter::emitShift(Operation *op) {
  std::string sc =
      mslScalarType(elementScalarType(op->getResult(0).getType()));
  std::string usc = sc.front() == 'u' ? sc : "u" + sc;
  const char *o = isa<arith::ShLIOp>(op) ? "<<" : ">>";
  bool logical = isa<arith::ShLIOp, arith::ShRUIOp>(op);
  Value res = op->getResult(0);
  auto &lhs = names(op->getOperand(0));
  auto &rhs = names(op->getOperand(1));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &a = lhs[lhs.size() == 1 ? 0 : r];
    const std::string &b = rhs[rhs.size() == 1 ? 0 : r];
    std::string lcast = logical ? usc + "(" + a + ")" : a;
    os << ind() << sc << " " << id << " = (" << lcast << " " << o << " " << b
       << ");\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

LogicalResult MSLEmitter::emitElementwise(Operation *op, StringRef binop,
                                          StringRef sc, StringRef opCast) {
  Value res = op->getResult(0);
  auto &lhs = names(op->getOperand(0));
  auto &rhs = names(op->getOperand(1));
  int rc = regCount(res);
  std::string pre = opCast.empty() ? "" : "(" + opCast.str() + ")";
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &a = lhs[lhs.size() == 1 ? 0 : r];
    const std::string &b = rhs[rhs.size() == 1 ? 0 : r];
    os << ind() << "" << sc.str() << " " << id << " = (" << pre << a << " "
       << binop.str() << " " << pre << b << ");\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

LogicalResult MSLEmitter::emitMinMax(Operation *op, StringRef fn,
                                     StringRef opCast, bool propagateNan) {
  Value res = op->getResult(0);
  std::string sc = mslScalarType(elementScalarType(res.getType()));
  std::string pre = opCast.empty() ? "" : "(" + opCast.str() + ")";
  auto &lhs = names(op->getOperand(0));
  auto &rhs = names(op->getOperand(1));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &a = lhs[lhs.size() == 1 ? 0 : r];
    const std::string &b = rhs[rhs.size() == 1 ? 0 : r];
    std::string expr = fn.str() + "(" + pre + a + ", " + pre + b + ")";
    if (propagateNan)
      expr = "((metal::isnan(" + a + ") || metal::isnan(" + b + ")) ? " + a +
             " + " + b + " : " + expr + ")";
    os << ind() << sc << " " << id << " = " << expr << ";\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

LogicalResult MSLEmitter::emitUnary(Operation *op, StringRef fn, StringRef sc) {
  Value res = op->getResult(0);
  auto &a = names(op->getOperand(0));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &v = a[a.size() == 1 ? 0 : r];
    os << ind() << sc.str() << " " << id << " = (" << sc.str() << ")"
       << fn.str() << "(" << v << ");\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

LogicalResult MSLEmitter::emitTernary(Operation *op, StringRef fn,
                                      StringRef sc) {
  Value res = op->getResult(0);
  auto &a = names(op->getOperand(0));
  auto &b = names(op->getOperand(1));
  auto &c = names(op->getOperand(2));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    os << ind() << sc.str() << " " << id << " = " << fn.str() << "("
       << a[a.size() == 1 ? 0 : r] << ", " << b[b.size() == 1 ? 0 : r] << ", "
       << c[c.size() == 1 ? 0 : r] << ");\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

//===----------------------------------------------------------------------===//
// AST sub-expression builders (binary / minmax / unary / ternary)
//===----------------------------------------------------------------------===//

static msl::BinOp intBinOp(Operation *op) {
  if (isa<arith::AddIOp>(op))
    return msl::BinOp::Add;
  if (isa<arith::SubIOp>(op))
    return msl::BinOp::Sub;
  if (isa<arith::MulIOp>(op))
    return msl::BinOp::Mul;
  if (isa<arith::DivSIOp, arith::DivUIOp>(op))
    return msl::BinOp::Div;
  return msl::BinOp::Rem;
}

msl::Expr *MSLEmitter::astElementwiseExpr(msl::BinOp op, msl::Type *opCast,
                                          StringRef a, StringRef b) {
  auto recast = [&](StringRef n) -> msl::Expr * {
    msl::Expr *v = ctx.var(n);
    return opCast ? ctx.cast(msl::Cast::Style::CStyle, opCast, v) : v;
  };
  return ctx.paren(ctx.binary(op, recast(a), recast(b)));
}

msl::Expr *MSLEmitter::astIntBinaryExpr(Operation *op, StringRef a,
                                        StringRef b) {
  msl::BinOp o = intBinOp(op);
  Type resElem = elementScalarType(op->getResult(0).getType());
  if (auto it = dyn_cast<IntegerType>(resElem); it && it.getWidth() == 1) {
    // (bool)((((int)a) o ((int)b)) & 1)
    msl::Type *i32 = ctx.scalar(msl::Scalar::I32);
    msl::Expr *ca = ctx.cast(msl::Cast::Style::CStyle, i32, ctx.var(a));
    msl::Expr *cb = ctx.cast(msl::Cast::Style::CStyle, i32, ctx.var(b));
    msl::Expr *inner = ctx.paren(ctx.binary(o, ca, cb));
    msl::Expr *masked =
        ctx.paren(ctx.binary(msl::BinOp::And, inner, ctx.lit("1")));
    return ctx.cast(msl::Cast::Style::CStyle, ctx.scalar(msl::Scalar::I1),
                    masked);
  }
  msl::Type *opCast = nullptr;
  if (isa<arith::DivUIOp, arith::RemUIOp>(op))
    opCast = astUnsignedType(resElem);
  return astElementwiseExpr(o, opCast, a, b);
}

msl::Expr *MSLEmitter::astShiftExpr(Operation *op, StringRef a, StringRef b) {
  Type resElem = elementScalarType(op->getResult(0).getType());
  msl::BinOp o = isa<arith::ShLIOp>(op) ? msl::BinOp::Shl : msl::BinOp::Shr;
  bool logical = isa<arith::ShLIOp, arith::ShRUIOp>(op);
  msl::Expr *lhs = ctx.var(a);
  if (logical)
    lhs = ctx.cast(msl::Cast::Style::CStyle, astUnsignedType(resElem), lhs);
  return ctx.paren(ctx.binary(o, lhs, ctx.var(b)));
}

msl::Expr *MSLEmitter::astMinMaxExpr(StringRef fn, msl::Type *opCast,
                                     bool propagateNan, StringRef a,
                                     StringRef b) {
  auto recast = [&](StringRef n) -> msl::Expr * {
    msl::Expr *v = ctx.var(n);
    return opCast ? ctx.cast(msl::Cast::Style::CStyle, opCast, v) : v;
  };
  msl::Expr *call = ctx.call(fn, {recast(a), recast(b)});
  if (!propagateNan)
    return call;
  // ((isnan(a) || isnan(b)) ? a + b : call)
  msl::Expr *na = ctx.call(msl::builtin::math::Isnan, {ctx.var(a)});
  msl::Expr *nb = ctx.call(msl::builtin::math::Isnan, {ctx.var(b)});
  msl::Expr *cond = ctx.paren(ctx.binary(msl::BinOp::LOr, na, nb));
  msl::Expr *sum = ctx.binary(msl::BinOp::Add, ctx.var(a), ctx.var(b));
  return ctx.paren(ctx.ternary(cond, sum, call));
}

msl::Expr *MSLEmitter::astUnaryExpr(StringRef fn, msl::Type *sc, StringRef v) {
  return ctx.cast(msl::Cast::Style::CStyle, sc, ctx.call(fn, {ctx.var(v)}));
}

msl::Expr *MSLEmitter::astTernaryCallExpr(StringRef fn, StringRef a,
                                          StringRef b, StringRef c) {
  return ctx.call(fn, {ctx.var(a), ctx.var(b), ctx.var(c)});
}

//===----------------------------------------------------------------------===//
// Casts / bitcast
//===----------------------------------------------------------------------===//

LogicalResult MSLEmitter::emitCast(Operation *op) {
  Value res = op->getResult(0);
  std::string dst = mslScalarType(elementScalarType(res.getType()));
  if (dst.empty()) {
    op->emitError("EmitMSL: unhandled cast target type");
    return failure();
  }
  {
    Type srcElem = elementScalarType(op->getOperand(0).getType());
    bool toHalf = dst == "half" || dst == "bfloat";
    if (srcElem.isF32() && toHalf) {
      bool rtz = false, handle = false;
      if (auto f = dyn_cast<tt::FpToFpOp>(op)) {
        if (auto rnd = f.getRounding()) {
          handle = true;
          rtz = *rnd == tt::RoundingMode::RTZ;
        }
      } else if (isa<arith::TruncFOp>(op)) {
        handle = true;
      }
      if (handle) {
        auto &a = names(op->getOperand(0));
        int rc = regCount(res);
        SmallVector<std::string> ids;
        for (int r = 0; r < rc; ++r) {
          const std::string &v = a[a.size() == 1 ? 0 : r];
          ids.push_back(rtz ? emitTruncatedFloatValue(dst, v)
                            : emitRoundedHalfValueFull(dst, v));
        }
        valMap[res] = ids;
        return success();
      }
    }
  }
  std::string srcCast;
  if (isa<arith::ExtUIOp, arith::UIToFPOp>(op))
    srcCast = mslUnsignedType(elementScalarType(op->getOperand(0).getType()));
  auto &a = names(op->getOperand(0));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &v = a[a.size() == 1 ? 0 : r];
    std::string src = srcCast.empty() ? v : "(" + srcCast + ")" + v;
    os << ind() << dst << " " << id << " = static_cast<" << dst << ">(" << src
       << ");\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

LogicalResult MSLEmitter::emitPtrIntCast(Operation *op) {
  Value res = op->getResult(0);
  bool toPtr = isa<tt::IntToPtrOp>(op);
  std::string dst = toPtr ? mslStorageType(res.getType())
                          : mslScalarType(elementScalarType(res.getType()));
  if (dst.empty()) {
    op->emitError("EmitMSL: unhandled ptr/int cast type");
    return failure();
  }
  auto &a = names(op->getOperand(0));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &v = a[a.size() == 1 ? 0 : r];
    os << ind() << dst << " " << id << " = (" << dst << ")" << v << ";\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

LogicalResult MSLEmitter::emitBitcast(Operation *op) {
  Value res = op->getResult(0);
  Type resElem = res.getType();
  if (auto rt = dyn_cast<RankedTensorType>(resElem))
    resElem = rt.getElementType();
  bool isPtr = isa<tt::PointerType>(resElem);
  std::string dst = isPtr ? mslStorageType(res.getType())
                          : mslScalarType(elementScalarType(res.getType()));
  if (dst.empty()) {
    op->emitError("EmitMSL: unhandled bitcast target type");
    return failure();
  }
  auto &a = names(op->getOperand(0));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &v = a[a.size() == 1 ? 0 : r];
    if (isPtr)
      os << ind() << dst << " " << id << " = (" << dst << ")" << v << ";\n";
    else
      os << ind() << dst << " " << id << " = as_type<" << dst << ">(" << v
         << ");\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

// General arith cast RHS: static_cast<dst>(src). The fp32->half/bfloat
// narrowing path (rtz/round-to-nearest) stays on the string route (bespoke
// bit-twiddling, a Raw escape hatch); this builds the common static_cast form.
msl::Expr *MSLEmitter::astCastExpr(Operation *op, StringRef v) {
  Type resElem = elementScalarType(op->getResult(0).getType());
  msl::Expr *src = ctx.var(v);
  if (isa<arith::ExtUIOp, arith::UIToFPOp>(op))
    src = ctx.cast(msl::Cast::Style::CStyle,
                   astUnsignedType(elementScalarType(op->getOperand(0).getType())),
                   src);
  return ctx.cast(msl::Cast::Style::Static, astScalarType(resElem), src);
}

msl::Expr *MSLEmitter::astPtrIntCastExpr(Operation *op, StringRef v) {
  Value res = op->getResult(0);
  bool toPtr = isa<tt::IntToPtrOp>(op);
  msl::Type *dst = toPtr ? astStorageType(res.getType())
                         : astScalarType(elementScalarType(res.getType()));
  return ctx.cast(msl::Cast::Style::CStyle, dst, ctx.var(v));
}

msl::Expr *MSLEmitter::astBitcastExpr(Operation *op, StringRef v) {
  Value res = op->getResult(0);
  Type resElem = res.getType();
  if (auto rt = dyn_cast<RankedTensorType>(resElem))
    resElem = rt.getElementType();
  bool isPtr = isa<tt::PointerType>(resElem);
  if (isPtr)
    return ctx.cast(msl::Cast::Style::CStyle, astStorageType(res.getType()),
                    ctx.var(v));
  return ctx.cast(msl::Cast::Style::AsType,
                  astScalarType(elementScalarType(res.getType())), ctx.var(v));
}

//===----------------------------------------------------------------------===//
// Clamp / compare / select
//===----------------------------------------------------------------------===//

LogicalResult MSLEmitter::emitClamp(tt::ClampFOp op) {
  Value res = op.getResult();
  std::string sc = mslScalarType(elementScalarType(res.getType()));
  auto &x = names(op.getX());
  auto &lo = names(op.getMin());
  auto &hi = names(op.getMax());
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &xv = x[x.size() == 1 ? 0 : r];
    const std::string &lv = lo[lo.size() == 1 ? 0 : r];
    const std::string &hv = hi[hi.size() == 1 ? 0 : r];
    std::string clamped = "metal::clamp(" + xv + ", " + lv + ", " + hv + ")";
    if (op.getPropagateNan() == tt::PropagateNan::ALL)
      clamped = "(metal::isnan(" + xv + ") ? " + xv + " : " + clamped + ")";
    os << ind() << sc << " " << id << " = " << clamped << ";\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

LogicalResult MSLEmitter::emitCmpI(arith::CmpIOp op) {
  const char *o;
  bool uns = false;
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::ult:
    uns = true;
    [[fallthrough]];
  case arith::CmpIPredicate::slt:
    o = "<";
    break;
  case arith::CmpIPredicate::ule:
    uns = true;
    [[fallthrough]];
  case arith::CmpIPredicate::sle:
    o = "<=";
    break;
  case arith::CmpIPredicate::ugt:
    uns = true;
    [[fallthrough]];
  case arith::CmpIPredicate::sgt:
    o = ">";
    break;
  case arith::CmpIPredicate::uge:
    uns = true;
    [[fallthrough]];
  case arith::CmpIPredicate::sge:
    o = ">=";
    break;
  case arith::CmpIPredicate::eq:
    o = "==";
    break;
  case arith::CmpIPredicate::ne:
    o = "!=";
    break;
  }
  std::string opCast =
      uns ? mslUnsignedType(elementScalarType(op.getLhs().getType())) : "";
  return emitElementwise(op, o, "bool", opCast);
}

LogicalResult MSLEmitter::emitCmpF(arith::CmpFOp op) {
  const char *o;
  switch (op.getPredicate()) {
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    o = "<";
    break;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    o = "<=";
    break;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    o = ">";
    break;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    o = ">=";
    break;
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    o = "==";
    break;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    o = "!=";
    break;
  default:
    op.emitError("EmitMSL: unsupported cmpf predicate");
    return failure();
  }
  return emitElementwise(op, o, "bool");
}

LogicalResult MSLEmitter::emitSelect(arith::SelectOp op) {
  Value res = op.getResult();
  auto &cond = names(op.getCondition());
  auto &tval = names(op.getTrueValue());
  auto &fval = names(op.getFalseValue());
  Type resElem = res.getType();
  if (auto rt = dyn_cast<RankedTensorType>(resElem))
    resElem = rt.getElementType();
  std::string sc = isa<tt::PointerType>(resElem)
                       ? mslStorageType(res.getType())
                       : mslScalarType(elementScalarType(res.getType()));
  int rc = regCount(res);
  SmallVector<std::string> ids;
  for (int r = 0; r < rc; ++r) {
    std::string id = fresh();
    const std::string &c = cond[cond.size() == 1 ? 0 : r];
    const std::string &t = tval[tval.size() == 1 ? 0 : r];
    const std::string &f = fval[fval.size() == 1 ? 0 : r];
    os << ind() << sc << " " << id << " = " << c << " ? " << t << " : " << f
       << ";\n";
    ids.push_back(id);
  }
  valMap[res] = ids;
  return success();
}

msl::Expr *MSLEmitter::astClampExpr(tt::ClampFOp op, StringRef x, StringRef lo,
                                    StringRef hi) {
  msl::Expr *clamped = ctx.call(msl::builtin::math::Clamp,
                                {ctx.var(x), ctx.var(lo), ctx.var(hi)});
  if (op.getPropagateNan() == tt::PropagateNan::ALL) {
    msl::Expr *isnan = ctx.call(msl::builtin::math::Isnan, {ctx.var(x)});
    clamped = ctx.paren(ctx.ternary(isnan, ctx.var(x), clamped));
  }
  return clamped;
}

msl::Expr *MSLEmitter::astSelectExpr(StringRef c, StringRef t, StringRef f) {
  return ctx.ternary(ctx.var(c), ctx.var(t), ctx.var(f));
}

//===----------------------------------------------------------------------===//
// Cross-lane shuffle
//===----------------------------------------------------------------------===//

std::string MSLEmitter::emitShuffle(StringRef op, StringRef sc, StringRef val,
                                    StringRef arg) {
  std::string out = fresh();
  if (sc == "long" || sc == "ulong") {
    std::string lo = fresh(), hi = fresh();
    os << ind() << "uint2 " << lo << " = as_type<uint2>(" << val << ");\n";
    os << ind() << "uint2 " << hi << ";\n";
    os << ind() << hi << ".x = " << op << "(" << lo << ".x, " << arg << ");\n";
    os << ind() << hi << ".y = " << op << "(" << lo << ".y, " << arg << ");\n";
    os << ind() << sc << " " << out << " = as_type<" << sc << ">(" << hi
       << ");\n";
    return out;
  }
  if (sc == "bool") {
    std::string b = fresh(), s = fresh();
    os << ind() << "uchar " << b << " = (uchar)" << val << ";\n";
    os << ind() << "uchar " << s << " = " << op << "(" << b << ", " << arg
       << ");\n";
    os << ind() << sc << " " << out << " = (bool)" << s << ";\n";
    return out;
  }
  if (sc == "bfloat" || sc == "half" || sc == "short" || sc == "char") {
    std::string bits = sc == "char" ? "uchar" : "ushort";
    std::string b = fresh(), s = fresh();
    os << ind() << bits << " " << b << " = as_type<" << bits << ">(" << val
       << ");\n";
    os << ind() << bits << " " << s << " = " << op << "(" << b << ", " << arg
       << ");\n";
    os << ind() << sc << " " << out << " = as_type<" << sc << ">(" << s
       << ");\n";
    return out;
  }
  os << ind() << sc << " " << out << " = " << op << "(" << val << ", " << arg
     << ");\n";
  return out;
}

msl::Expr *MSLEmitter::astShuffleExpr(StringRef op, StringRef sc, StringRef val,
                                      StringRef arg) {
  auto scalarOf = [&](StringRef s) -> msl::Scalar {
    if (s == "bfloat")
      return msl::Scalar::BF16;
    if (s == "half")
      return msl::Scalar::F16;
    if (s == "short")
      return msl::Scalar::I16;
    if (s == "char")
      return msl::Scalar::I8;
    if (s == "bool")
      return msl::Scalar::I1;
    if (s == "long")
      return msl::Scalar::I64;
    return msl::Scalar::U64; // ulong
  };
  auto shuffle = [&](msl::Expr *v) {
    return ctx.call(op, {v, ctx.var(arg)});
  };
  if (sc == "long" || sc == "ulong") {
    msl::Type *u2 = ctx.vector(msl::Scalar::U32, 2);
    msl::Expr *bits =
        ctx.cast(msl::Cast::Style::AsType, u2, ctx.var(val));
    // Both lanes shuffled by the same arg; a single as_type round-trips the pair.
    return ctx.cast(msl::Cast::Style::AsType, ctx.scalar(scalarOf(sc)),
                    shuffle(bits));
  }
  if (sc == "bool") {
    msl::Type *uc = ctx.scalar(msl::Scalar::U8);
    msl::Expr *b = ctx.cast(msl::Cast::Style::CStyle, uc, ctx.var(val));
    return ctx.cast(msl::Cast::Style::CStyle, ctx.scalar(msl::Scalar::I1),
                    shuffle(b));
  }
  if (sc == "bfloat" || sc == "half" || sc == "short" || sc == "char") {
    msl::Type *bits =
        ctx.scalar(sc == "char" ? msl::Scalar::U8 : msl::Scalar::U16);
    msl::Expr *b = ctx.cast(msl::Cast::Style::AsType, bits, ctx.var(val));
    return ctx.cast(msl::Cast::Style::AsType, ctx.scalar(scalarOf(sc)),
                    shuffle(b));
  }
  return shuffle(ctx.var(val));
}

//===----------------------------------------------------------------------===//
// Per-element builders for reshape / aliasing ops (emission stays in-header)
//===----------------------------------------------------------------------===//

msl::Expr *MSLEmitter::astMakeRangeElem(int start, StringRef off) {
  return ctx.binary(msl::BinOp::Add, ctx.lit(std::to_string(start)),
                    ctx.raw(off));
}

msl::Expr *MSLEmitter::astAliasElem(StringRef name) { return ctx.var(name); }

} // namespace mlir::triton::applegpu
