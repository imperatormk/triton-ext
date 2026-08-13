// EmitMSLDotFragments.cpp - simdgroup-matrix fragment MMA sub-builders.
//
// The leaf vocabulary the dot emitters build their MMA blocks out of: fragment
// types, decls, simdgroup_load/store, multiply_accumulate, and the C readback
// value. Pure AST construction over `ctx` - no emitter state, no IR matching,
// no policy. Split out of EmitMSLDot.cpp unchanged.
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

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

// `simdgroup_load(frag, base + off, ld[, origin, transpose]);`
//
// stageOperand writes operands into threadgroup memory row-major, so the
// fragment layout is canonical and the transpose flag is normally left off. An
// operand the IR transposes is staged in its untransposed order instead and
// read back through the flag, which the hardware honours for free.
msl::Stmt *MSLEmitter::sgLoad(StringRef frag, StringRef base, int64_t off,
                              int64_t ld, bool transpose) {
  SmallVector<msl::Expr *, 5> args{ctx.var(frag), fragAddr(base, off),
                                   ctx.lit(std::to_string(ld))};
  if (transpose) {
    args.push_back(ctx.raw("ulong2(0, 0)"));
    args.push_back(ctx.lit("true"));
  }
  return ctx.exprStmt(ctx.call(msl::builtin::sg::Load, args));
}

// `simdgroup_store(acc, base + off, ld);`
msl::Stmt *MSLEmitter::sgStore(StringRef acc, StringRef base, int64_t off,
                               int64_t ld) {
  return ctx.exprStmt(
      ctx.call(msl::builtin::sg::Store, {ctx.var(acc), fragAddr(base, off),
                                         ctx.lit(std::to_string(ld))}));
}

msl::Stmt *MSLEmitter::sgLoadExpr(StringRef frag, StringRef base,
                                  msl::Expr *off, int64_t ld) {
  return ctx.exprStmt(ctx.call(
      msl::builtin::sg::Load,
      {ctx.var(frag), ctx.binary(msl::BinOp::Add, ctx.var(base), off),
       ctx.lit(std::to_string(ld))}));
}

msl::Stmt *MSLEmitter::sgStoreExpr(StringRef acc, StringRef base,
                                   msl::Expr *off, int64_t ld) {
  return ctx.exprStmt(ctx.call(
      msl::builtin::sg::Store,
      {ctx.var(acc), ctx.binary(msl::BinOp::Add, ctx.var(base), off),
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

} // namespace mlir::triton::applegpu
