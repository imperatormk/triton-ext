// EmitMSLExpr.cpp - value-producing expression lowering.
//
// AST builders (astXxxExpr) that build the RHS for each value-producing op as
// typed msl::Expr nodes, which astEmitOp assigns into a DeclStmt init.
//
// INVARIANT: the printer is dumb - it inserts no grouping parens. A builder
// inserts an explicit ctx.paren(...) wherever a subexpression needs precedence
// grouping.

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

std::string MSLEmitter::floatLit(const APFloat &v, Type ty) {
  std::string lit = floatLit(v);
  if (ty.isBF16())
    return "bfloat(" + lit + ")";
  if (ty.isF16())
    return "half(" + lit + ")";
  return lit;
}

msl::Expr *MSLEmitter::astFloatLit(const APFloat &v, Type ty) {
  msl::Expr *lit = ctx.lit(floatLit(v));
  if (ty.isBF16())
    return ctx.call("bfloat", {lit});
  if (ty.isF16())
    return ctx.call("half", {lit});
  return lit;
}

//===----------------------------------------------------------------------===//
// Integer / float binary, shift, elementwise
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AST sub-expression builders (binary / minmax / unary / ternary)
//===----------------------------------------------------------------------===//

msl::BinOp arithBinOp(Operation *op) {
  if (isa<arith::AddFOp, arith::AddIOp>(op))
    return msl::BinOp::Add;
  if (isa<arith::SubFOp, arith::SubIOp>(op))
    return msl::BinOp::Sub;
  if (isa<arith::MulFOp, arith::MulIOp>(op))
    return msl::BinOp::Mul;
  if (isa<arith::DivFOp, tt::PreciseDivFOp, arith::DivSIOp, arith::DivUIOp>(op))
    return msl::BinOp::Div;
  if (isa<arith::RemSIOp, arith::RemUIOp>(op))
    return msl::BinOp::Rem;
  if (isa<arith::AndIOp>(op))
    return msl::BinOp::And;
  if (isa<arith::OrIOp>(op))
    return msl::BinOp::Or;
  return msl::BinOp::Xor;
}

msl::Expr *MSLEmitter::recast(msl::Type *opCast, StringRef n) {
  msl::Expr *v = ctx.var(n);
  return opCast ? ctx.cast(msl::Cast::Style::CStyle, opCast, v) : v;
}

msl::Expr *MSLEmitter::astElementwiseExpr(msl::BinOp op, msl::Type *opCast,
                                          StringRef a, StringRef b) {
  return ctx.paren(ctx.binary(op, recast(opCast, a), recast(opCast, b)));
}

msl::Expr *MSLEmitter::astIntBinaryExpr(Operation *op, StringRef a,
                                        StringRef b) {
  msl::BinOp o = arithBinOp(op);
  Type resElem = elementScalarType(op->getResult(0).getType());
  if (auto it = dyn_cast<IntegerType>(resElem); it && it.getWidth() == 1) {
    // (bool)((((int)a) o ((int)b)) & 1)
    msl::Type *i32 = ctx.scalar(msl::Scalar::I32);
    msl::Expr *ca = ctx.paren(ctx.cast(msl::Cast::Style::CStyle, i32, ctx.var(a)));
    msl::Expr *cb = ctx.paren(ctx.cast(msl::Cast::Style::CStyle, i32, ctx.var(b)));
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
  if (logical) {
    // Functional-style cast `usc(a)`.
    std::string usc = mslUnsignedType(resElem);
    if (usc.empty())
      usc = "u" + mslScalarType(resElem);
    lhs = ctx.call(usc, {ctx.var(a)});
  }
  return ctx.paren(ctx.binary(o, lhs, ctx.var(b)));
}

msl::Expr *MSLEmitter::astMinMaxExpr(StringRef fn, msl::Type *opCast,
                                     bool propagateNan, StringRef a,
                                     StringRef b) {
  msl::Expr *call = ctx.call(fn, {recast(opCast, a), recast(opCast, b)});
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

// General arith cast RHS: static_cast<dst>(src). The fp32->half/bfloat
// narrowing path (rtz/round-to-nearest) stays on the string route (bespoke
// bit-twiddling, a Raw escape hatch); this builds the common static_cast form.
msl::Expr *MSLEmitter::astCastExpr(Operation *op, StringRef v) {
  Type resElem = elementScalarType(op->getResult(0).getType());
  msl::Expr *src = ctx.var(v);
  if (isa<arith::ExtUIOp, arith::UIToFPOp>(op)) {
    // i1 has no unsigned MSL type; a bool source needs no reinterpret cast
    // (bool->wider is already zero-extending), so skip it.
    if (msl::Type *u =
            astUnsignedType(elementScalarType(op->getOperand(0).getType())))
      src = ctx.cast(msl::Cast::Style::CStyle, u, src);
  }
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

static msl::Scalar shuffleScalarOf(StringRef s) {
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
}

// Statement form of the shuffle: push the temp decls into `body` and return the
// fresh result name (used by reduce/scan where the shuffled value must be a name
// the combiner region binds to).
std::string MSLEmitter::astShuffle(StringRef op, StringRef sc, StringRef val,
                                   StringRef arg, msl::Block &body) {
  using CS = msl::Cast::Style;
  std::string out = fresh();
  auto call = [&](msl::Expr *v) { return ctx.call(op, {v, ctx.var(arg)}); };
  if (sc == "long" || sc == "ulong") {
    msl::Scalar wide = shuffleScalarOf(sc);
    std::string lo = fresh(), hi = fresh();
    msl::Type *u2 = ctx.vector(msl::Scalar::U32, 2);
    body.push_back(ctx.declStmt(u2, lo,
                                ctx.cast(CS::AsType, u2, ctx.var(val))));
    body.push_back(ctx.declStmt(u2, hi, nullptr));
    body.push_back(ctx.assignStmt(ctx.member(ctx.var(hi), "x"),
                                  call(ctx.member(ctx.var(lo), "x"))));
    body.push_back(ctx.assignStmt(ctx.member(ctx.var(hi), "y"),
                                  call(ctx.member(ctx.var(lo), "y"))));
    body.push_back(ctx.declStmt(ctx.scalar(wide), out,
                                ctx.cast(CS::AsType, ctx.scalar(wide),
                                         ctx.var(hi))));
    return out;
  }
  if (sc == "bool") {
    msl::Type *uc = ctx.scalar(msl::Scalar::U8);
    std::string b = fresh(), s = fresh();
    body.push_back(ctx.declStmt(uc, b,
                                ctx.cast(CS::CStyle, uc, ctx.var(val))));
    body.push_back(ctx.declStmt(uc, s, call(ctx.var(b))));
    body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), out,
                                ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::I1),
                                         ctx.var(s))));
    return out;
  }
  if (sc == "bfloat" || sc == "half" || sc == "short" || sc == "char") {
    msl::Scalar bitsS = sc == "char" ? msl::Scalar::U8 : msl::Scalar::U16;
    msl::Type *bits = ctx.scalar(bitsS);
    msl::Scalar scS = shuffleScalarOf(sc);
    std::string b = fresh(), s = fresh();
    body.push_back(ctx.declStmt(bits, b,
                                ctx.cast(CS::AsType, bits, ctx.var(val))));
    body.push_back(ctx.declStmt(bits, s, call(ctx.var(b))));
    body.push_back(ctx.declStmt(ctx.scalar(scS), out,
                                ctx.cast(CS::AsType, ctx.scalar(scS),
                                         ctx.var(s))));
    return out;
  }
  body.push_back(
      ctx.declStmt(ctx.named(sc), out, call(ctx.var(val))));
  return out;
}

msl::Expr *MSLEmitter::astShuffleExpr(StringRef op, StringRef sc, StringRef val,
                                      StringRef arg) {
  auto shuffle = [&](msl::Expr *v) {
    return ctx.call(op, {v, ctx.var(arg)});
  };
  if (sc == "long" || sc == "ulong") {
    msl::Type *u2 = ctx.vector(msl::Scalar::U32, 2);
    msl::Expr *bits =
        ctx.cast(msl::Cast::Style::AsType, u2, ctx.var(val));
    // Both lanes shuffled by the same arg; a single as_type round-trips the pair.
    return ctx.cast(msl::Cast::Style::AsType, ctx.scalar(shuffleScalarOf(sc)),
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
    return ctx.cast(msl::Cast::Style::AsType, ctx.scalar(shuffleScalarOf(sc)),
                    shuffle(b));
  }
  return shuffle(ctx.var(val));
}

//===----------------------------------------------------------------------===//
// Per-element builders for reshape / aliasing ops (emission stays in-header)
//===----------------------------------------------------------------------===//

msl::Expr *MSLEmitter::astMakeRangeElem(int start, msl::Expr *off) {
  return ctx.binary(msl::BinOp::Add, ctx.i32lit(start), off);
}

msl::Expr *MSLEmitter::astAliasElem(StringRef name) { return ctx.var(name); }

} // namespace mlir::triton::applegpu
