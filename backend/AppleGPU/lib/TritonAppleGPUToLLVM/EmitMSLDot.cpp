// EmitMSLDot.cpp - dot/GEMM simdgroup-matrix + fp-narrowing lowering
// (string + AST forms).
//
// Out-lined AST siblings of the simdgroup-matrix fragment MMA sub-builders in
// emitDot/emitDotScalar/emitFusedGemm, plus the fp_to_fp narrowing helpers.
// Emission still runs on the string path this layer; these nodes only need to
// exist and print byte-identically once the flip layer routes emission here.
//
// INVARIANT: the printer inserts no grouping parens; wherever the string path
// wrapped a subexpression the AST sibling inserts an explicit ctx.paren(...).
// The fp-narrowing bit-twiddling body is the design-sanctioned Raw escape
// hatch (MSL_AST_DESIGN.md): the sibling wraps the exact string block in one
// RawStmt and keeps only the outer `sc h = as_type<sc>(bits);` as real nodes.
// These siblings must never touch nextId/indent - emission owns those - so the
// fp helpers reconstruct their block through a private id/indent local seeded
// from the emitter's current state without mutating it.

#include "MSLConstants.h"
#include "MSLEmitter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::triton::applegpu {

//===----------------------------------------------------------------------===//
// simdgroup-matrix fragment MMA sub-builders
//===----------------------------------------------------------------------===//

msl::MatrixType *MSLEmitter::astSgFragType(Type t) {
  if (t.isF16())
    return ctx.matrix(msl::MatrixType::Elem::Half);
  if (t.isBF16())
    return ctx.matrix(msl::MatrixType::Elem::Bfloat);
  return ctx.matrix(msl::MatrixType::Elem::Float);
}

// `base + off` (no outer paren - the string path emits it bare inside the call).
msl::Expr *MSLEmitter::astFragAddr(StringRef base, int64_t off) {
  return ctx.binary(msl::BinOp::Add, ctx.var(base), ctx.lit(std::to_string(off)));
}

// `frag name;` - uninitialized operand fragment decl.
msl::Stmt *MSLEmitter::astFragDecl(msl::MatrixType *frag, StringRef name) {
  return ctx.declStmt(frag, name);
}

// `frag name = frag(0.0f);` - zeroed accumulator fragment. The type name doubles
// as the ctor callee (simdgroup_float8x8(0.0f)); read the printed name back off
// the MatrixType so the call callee matches the decl type exactly.
msl::Stmt *MSLEmitter::astAccFragDecl(msl::MatrixType *frag, StringRef name) {
  StringRef ctorName = frag->elem == msl::MatrixType::Elem::Half
                           ? msl::builtin::sg::Half8x8
                       : frag->elem == msl::MatrixType::Elem::Bfloat
                           ? msl::builtin::sg::Bfloat8x8
                           : msl::builtin::sg::Float8x8;
  return ctx.declStmt(frag, name, ctx.call(ctorName, {ctx.lit("0.0f")}));
}

// `simdgroup_load(frag, base + off, ld);`
msl::Stmt *MSLEmitter::astSgLoad(StringRef frag, StringRef base, int64_t off,
                                 int64_t ld) {
  return ctx.exprStmt(ctx.call(
      msl::builtin::sg::Load,
      {ctx.var(frag), astFragAddr(base, off), ctx.lit(std::to_string(ld))}));
}

// `simdgroup_store(acc, base + off, ld);`
msl::Stmt *MSLEmitter::astSgStore(StringRef acc, StringRef base, int64_t off,
                                  int64_t ld) {
  return ctx.exprStmt(ctx.call(
      msl::builtin::sg::Store,
      {ctx.var(acc), astFragAddr(base, off), ctx.lit(std::to_string(ld))}));
}

// `simdgroup_multiply_accumulate(acc, a, b, acc);`
msl::Stmt *MSLEmitter::astSgMultiplyAccumulate(StringRef acc, StringRef a,
                                               StringRef b) {
  return ctx.exprStmt(
      ctx.call(msl::builtin::sg::MultiplyAccumulate,
               {ctx.var(acc), ctx.var(a), ctx.var(b), ctx.var(acc)}));
}

// C readback value: `buf[off] + base` (off already carries its outer parens).
msl::Expr *MSLEmitter::astReadbackValue(StringRef buf, msl::Expr *off,
                                        StringRef base) {
  return ctx.binary(msl::BinOp::Add, ctx.subscript(ctx.var(buf), off),
                    ctx.var(base));
}

//===----------------------------------------------------------------------===//
// fp_to_fp narrowing (Raw bit-twiddling body + real outer as_type decl)
//===----------------------------------------------------------------------===//

namespace {
// Mirror of the emitter's fresh()/ind() over a private state so a sibling can
// rebuild a helper's block byte-identically without perturbing nextId/indent.
struct LocalGen {
  int id;
  int indent;
  std::string fresh() { return "v" + std::to_string(id++); }
  std::string ind() const { return std::string(indent * 4, ' '); }
};

// A helper block's two result names: `bits` (the assembled ushort) and `h` (the
// result var the outer `sc h = as_type<sc>(bits);` decl binds).
struct NarrowNames {
  std::string bits;
  std::string h;
};

// Writes emitRoundedHalfValueFull's block (everything up to but excluding the
// final `sc h = as_type<sc>(bits);`) to `os`, returning the bits/h var names.
NarrowNames buildRoundedHalfValueFull(llvm::raw_ostream &os, LocalGen &g,
                                      const std::string &sc,
                                      const std::string &v) {
  std::string f = g.fresh(), u = g.fresh(), h = g.fresh(), bits = g.fresh();
  os << g.ind() << "float " << f << " = (float)(" << v << ");\n";
  os << g.ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
  os << g.ind() << "ushort " << bits << ";\n";
  std::string sgn = g.fresh(), e32 = g.fresh(), mant = g.fresh();
  os << g.ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
  os << g.ind() << "int " << e32 << " = (int)((" << u << " >> 23) & 0xffu);\n";
  os << g.ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
  if (sc == "bfloat") {
    std::string r = g.fresh();
    os << g.ind() << "if (" << e32 << " == 0xff) {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)(((" << u << " >> 16) & 0xffffu) | ("
       << mant << " ? 0x40u : 0u));\n";
    --g.indent;
    os << g.ind() << "} else {\n";
    ++g.indent;
    os << g.ind() << "uint " << r << " = (" << u << " >> 16) & 1u;\n";
    os << g.ind() << "uint __t = (" << u << " + 0x7fffu + " << r << ");\n";
    os << g.ind() << bits << " = (ushort)((__t >> 16) & 0xffffu);\n";
    --g.indent;
    os << g.ind() << "}\n";
  } else {
    std::string ex = g.fresh();
    os << g.ind() << "int " << ex << " = " << e32 << " - 112;\n";
    os << g.ind() << "if (" << e32 << " == 0xff) {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u | (" << mant
       << " ? 0x200u : 0u));\n";
    --g.indent;
    os << g.ind() << "} else if (" << ex << " >= 31) {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u);\n";
    --g.indent;
    os << g.ind() << "} else if (" << ex << " <= 0) {\n";
    ++g.indent;
    os << g.ind() << "if (" << ex << " < -10) { " << bits << " = (ushort)"
       << sgn << "; }\n";
    os << g.ind() << "else {\n";
    ++g.indent;
    os << g.ind() << "uint __fm = " << mant << " | 0x800000u;\n";
    os << g.ind() << "int __sh = 14 - " << ex << ";\n";
    os << g.ind() << "uint __m = __fm >> __sh;\n";
    os << g.ind() << "uint __rem = __fm & ((1u << __sh) - 1u);\n";
    os << g.ind() << "uint __half = 1u << (__sh - 1);\n";
    os << g.ind() << "if (__rem > __half || (__rem == __half && (__m & 1u))) "
       << "__m += 1;\n";
    os << g.ind() << bits << " = (ushort)(" << sgn << " | __m);\n";
    --g.indent;
    os << g.ind() << "}\n";
    --g.indent;
    os << g.ind() << "} else {\n";
    ++g.indent;
    std::string m = g.fresh(), rem = g.fresh();
    os << g.ind() << "uint " << m << " = " << mant << " >> 13;\n";
    os << g.ind() << "uint " << rem << " = " << mant << " & 0x1fffu;\n";
    os << g.ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
       << " << 10) | " << m << ");\n";
    os << g.ind() << "if (" << rem << " > 0x1000u || (" << rem
       << " == 0x1000u && (" << m << " & 1u))) " << bits << " += 1;\n";
    --g.indent;
    os << g.ind() << "}\n";
  }
  return {bits, h};
}

NarrowNames buildTruncatedFloatValue(llvm::raw_ostream &os, LocalGen &g,
                                     const std::string &sc,
                                     const std::string &v) {
  std::string f = g.fresh(), u = g.fresh(), h = g.fresh(), bits = g.fresh();
  os << g.ind() << "float " << f << " = (float)(" << v << ");\n";
  os << g.ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
  os << g.ind() << "ushort " << bits << ";\n";
  if (sc == "bfloat") {
    os << g.ind() << bits << " = (ushort)((" << u << " >> 16) & 0xffffu);\n";
  } else {
    std::string sgn = g.fresh(), ex = g.fresh(), mant = g.fresh();
    os << g.ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
    os << g.ind() << "int " << ex << " = (int)((" << u
       << " >> 23) & 0xffu) - 112;\n";
    os << g.ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
    os << g.ind() << "if (((" << u << " >> 23) & 0xffu) == 0xffu) {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u | (" << mant
       << " ? 0x200u : 0u));\n";
    --g.indent;
    os << g.ind() << "} else if (" << ex << " >= 31) {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)(" << sgn << " | 0x7bffu);\n";
    --g.indent;
    os << g.ind() << "} else if (" << ex << " <= 0) {\n";
    ++g.indent;
    os << g.ind() << "if (" << ex << " < -10) { " << bits << " = (ushort)"
       << sgn << "; }\n";
    os << g.ind() << "else { uint __m = (" << mant << " | 0x800000u) >> (14 - "
       << ex << "); " << bits << " = (ushort)(" << sgn << " | __m); }\n";
    --g.indent;
    os << g.ind() << "} else {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
       << " << 10) | (" << mant << " >> 13));\n";
    --g.indent;
    os << g.ind() << "}\n";
  }
  return {bits, h};
}

NarrowNames buildRoundedHalfValue(llvm::raw_ostream &os, LocalGen &g,
                                  const std::string &sc, const std::string &v) {
  std::string f = g.fresh(), u = g.fresh(), h = g.fresh();
  os << g.ind() << "float " << f << " = (float)(" << v << ");\n";
  os << g.ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
  std::string bits = g.fresh();
  os << g.ind() << "ushort " << bits << ";\n";
  if (sc == "bfloat") {
    std::string r = g.fresh();
    os << g.ind() << "uint " << r << " = (" << u << " >> 16) & 1u;\n";
    os << g.ind() << bits << " = (ushort)(((" << u << " + 0x7fffu + " << r
       << ") >> 16) & 0xffffu);\n";
  } else {
    std::string sgn = g.fresh(), ex = g.fresh(), mant = g.fresh(),
                m = g.fresh(), rem = g.fresh();
    os << g.ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
    os << g.ind() << "int " << ex << " = (int)((" << u
       << " >> 23) & 0xffu) - 112;\n";
    os << g.ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
    os << g.ind() << "if (" << ex << " <= 0) {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)" << sgn << ";\n";
    --g.indent;
    os << g.ind() << "} else if (" << ex << " >= 31) {\n";
    ++g.indent;
    os << g.ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u);\n";
    --g.indent;
    os << g.ind() << "} else {\n";
    ++g.indent;
    os << g.ind() << "uint " << m << " = " << mant << " >> 13;\n";
    os << g.ind() << "uint " << rem << " = " << mant << " & 0x1fffu;\n";
    os << g.ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
       << " << 10) | " << m << ");\n";
    os << g.ind() << "if (" << rem << " > 0x1000u || (" << rem
       << " == 0x1000u && (" << m << " & 1u))) " << bits << " += 1;\n";
    --g.indent;
    os << g.ind() << "}\n";
  }
  return {bits, h};
}
} // namespace

// A DeclStmt owns a single init Expr, not a preceding block, and RawStmt owns no
// init slot - so the block and its terminating `sc h = as_type<sc>(bits);` decl
// cannot split into (RawStmt, DeclStmt) without a wrapping scope that would add
// stray braces. Per MSL_AST_DESIGN.md the whole narrowing lowers as one Raw
// block; the sibling assembles it byte-for-byte with the string helper.
static msl::Stmt *makeNarrowSibling(
    msl::MSLContext &ctx, int nextId, int indent, const std::string &sc,
    const std::string &v, std::string &outName,
    llvm::function_ref<NarrowNames(llvm::raw_ostream &, LocalGen &)> body) {
  std::string block;
  llvm::raw_string_ostream bos(block);
  LocalGen g{nextId, indent};
  NarrowNames nm = body(bos, g);
  bos.flush();
  block += g.ind() + sc + " " + nm.h + " = as_type<" + sc + ">(" + nm.bits +
           ");\n";
  outName = nm.h;
  return ctx.rawStmt(block);
}

msl::Stmt *MSLEmitter::astRoundedHalfValueFull(const std::string &sc,
                                               const std::string &v,
                                               std::string &outName) {
  return makeNarrowSibling(
      ctx, nextId, indent, sc, v, outName,
      [&](llvm::raw_ostream &os, LocalGen &g) {
        return buildRoundedHalfValueFull(os, g, sc, v);
      });
}

msl::Stmt *MSLEmitter::astTruncatedFloatValue(const std::string &sc,
                                              const std::string &v,
                                              std::string &outName) {
  return makeNarrowSibling(
      ctx, nextId, indent, sc, v, outName,
      [&](llvm::raw_ostream &os, LocalGen &g) {
        return buildTruncatedFloatValue(os, g, sc, v);
      });
}

msl::Stmt *MSLEmitter::astRoundedHalfValue(const std::string &sc,
                                           const std::string &v,
                                           std::string &outName) {
  return makeNarrowSibling(
      ctx, nextId, indent, sc, v, outName,
      [&](llvm::raw_ostream &os, LocalGen &g) {
        return buildRoundedHalfValue(os, g, sc, v);
      });
}

} // namespace mlir::triton::applegpu
