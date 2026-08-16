// EmitMSLAtomic.cpp - atomic RMW / CAS / poll + histogram lowering.
//
// AST builders for the atomic families, dispatched from emitOp. The CAS /
// spin / zero-init control flow is modelled as real WhileScope / IfScope /
// ForScope node trees (not Raw blocks) - atomics control flow is exactly what
// the node set exists to express.
//
// INVARIANT: the printer inserts no grouping parens; a builder inserts an
// explicit ctx.paren(...) wherever a subexpression needs precedence grouping.

#include "MSLAtomic.h"
#include "MSLConstants.h"

using namespace mlir;
using llvm::StringRef;

namespace mlir::triton::applegpu {

namespace ba = msl::builtin::atomic;
namespace border = msl::builtin::order;
namespace bmem = msl::builtin::memflags;
using B = msl::BinOp;
using CS = msl::Cast::Style;

//===----------------------------------------------------------------------===//
// Shared leaves
//===----------------------------------------------------------------------===//

// init0(sc): "0.0" for float/half, else "0".
msl::Expr *AtomicEmitter::init0(StringRef sc) {
  return ctx.lit(sc == "float" || sc == "half" ? "0.0" : "0");
}

// (device atomic_<elem>*)p - the atomic-typed device pointer the RMW/poll
// casts to.
msl::Expr *AtomicEmitter::deviceAtomicPtr(msl::Scalar elem, StringRef p) {
  msl::Type *ptr = ctx.ptr(ctx.atomic(elem), msl::AddrSpace::Device);
  return ctx.cast(CS::CStyle, ptr, ctx.var(p));
}

// atomic_compare_exchange_weak_explicit(ptr, &exp, newVal, relaxed, relaxed).
msl::Expr *AtomicEmitter::casWeak(msl::Expr *ptr, StringRef expVar,
                                  msl::Expr *newVal) {
  return ctx.call(ba::CompareExchangeWeak,
                  {ptr, ctx.addrOf(ctx.var(expVar)), newVal,
                   ctx.lit(border::Relaxed), ctx.lit(border::Relaxed)});
}

// (isHigh) ? (word >> 16) : (word & 0xffffu) - the packed-16 lane selector.
msl::Expr *AtomicEmitter::packed16Extract(StringRef word, StringRef isHigh) {
  msl::Expr *hi = ctx.paren(ctx.binary(B::Shr, ctx.var(word), ctx.lit("16")));
  msl::Expr *lo =
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0xffffu")));
  return ctx.paren(ctx.ternary(ctx.paren(ctx.var(isHigh)), hi, lo));
}

// (isHigh) ? ((word & 0x0000ffffu) | (newBitsU32 << 16))
//          : ((word & 0xffff0000u) | newBitsU32)
msl::Expr *AtomicEmitter::packed16Merge(StringRef word, StringRef isHigh,
                                        msl::Expr *newBitsU32) {
  msl::Expr *hiShift =
      ctx.paren(ctx.binary(B::Shl, ctx.paren(newBitsU32), ctx.lit("16")));
  msl::Expr *highWord = ctx.paren(ctx.binary(
      B::Or,
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0x0000ffffu"))),
      hiShift));
  msl::Expr *lowWord = ctx.paren(ctx.binary(
      B::Or,
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0xffff0000u"))),
      newBitsU32));
  return ctx.ternary(ctx.paren(ctx.var(isHigh)), highWord, lowWord);
}

// Binds wordPtr / isHigh names into the out-params:
//   bool isHigh = ((size_t)(p) & 2u) != 0u;
//   device atomic_uint *wordPtr = (device atomic_uint *)((size_t)(p) & ~3);
// Every packed-16 site (RMW, CAS, poll) routes word addressing through here.
msl::Block AtomicEmitter::packed16Base(StringRef p, std::string &wordPtrOut,
                                       std::string &isHighOut) {
  std::string isHigh = fresh(), wordPtr = fresh();
  msl::Type *auptr =
      ctx.ptr(ctx.atomic(msl::Scalar::U32), msl::AddrSpace::Device);
  auto szOf = [&] {
    return ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::SizeT),
                    ctx.paren(ctx.var(p)));
  };
  // ((size_t)(p) & 2u) != 0u
  msl::Expr *hi =
      ctx.binary(B::Ne, ctx.paren(ctx.binary(B::And, szOf(), ctx.lit("2u"))),
                 ctx.lit("0u"));
  // (size_t)(p) & ~(size_t)3
  msl::Expr *addr =
      ctx.binary(B::And, szOf(),
                 ctx.unary(msl::UnOp::Not,
                           ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::SizeT),
                                    ctx.lit("3"))));

  msl::Block block;
  block.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), isHigh, hi));
  block.push_back(ctx.declStmt(auptr, wordPtr,
                               ctx.cast(CS::CStyle, auptr, ctx.paren(addr))));
  wordPtrOut = wordPtr;
  isHighOut = isHigh;
  return block;
}

//===----------------------------------------------------------------------===//
// rmwCall - native atomic_fetch_*_explicit call
//===----------------------------------------------------------------------===//

// fn((device atomic_<elem>*)p, v, order[, mem_flags::mem_device]).
// `order` / `memflags` come from MSLConstants so the ordering surface stays
// greppable; the memflags arg is present iff `tail` is set.
msl::Expr *AtomicEmitter::rmwCall(StringRef fn, msl::Scalar atomicElem,
                                  StringRef p, StringRef v, StringRef order,
                                  bool memFlags) {
  llvm::SmallVector<msl::Expr *> args{deviceAtomicPtr(atomicElem, p),
                                      ctx.var(v), ctx.lit(order)};
  if (memFlags)
    args.push_back(ctx.lit(bmem::Device));
  return ctx.call(fn, args);
}

//===----------------------------------------------------------------------===//
// int32CAS / float32CAS / packed16CAS - int32 / float32 / packed16
// compare-exchange
//===----------------------------------------------------------------------===//

// sc exp = c;
// while (exp == c && !(atomic_compare_exchange_weak_explicit(
//          (device atomic_int *)p, &exp, v, memory_order_relaxed,
//          memory_order_relaxed))) {}
// [sc ]id = exp;
// Metal device atomics are relaxed-only; acquire/release orders are invalid
// MSL.
msl::Block AtomicEmitter::int32CAS(StringRef p, StringRef c, StringRef v,
                                   StringRef sc, StringRef id, bool declare) {
  std::string exp = fresh();
  msl::Type *scTy = ctx.named(sc);

  msl::Type *aiptr = ctx.deviceAtomicPtr(msl::Scalar::I32);
  msl::Expr *cas =
      casWeak(ctx.cast(CS::CStyle, aiptr, ctx.var(p)), exp, ctx.var(v));
  msl::Expr *cond =
      ctx.binary(B::LAnd, ctx.binary(B::Eq, ctx.var(exp), ctx.var(c)),
                 ctx.unary(msl::UnOp::LNot, ctx.paren(cas)));

  msl::Block block;
  block.push_back(ctx.declStmt(scTy, exp, ctx.var(c)));
  block.push_back(ctx.whileScope(cond, msl::Block{}));
  if (declare)
    block.push_back(ctx.declStmt(scTy, id, ctx.var(exp)));
  else
    block.push_back(ctx.assignStmt(ctx.var(id), ctx.var(exp)));
  return block;
}

// uint exp = as_type<uint>(c); uint cbits = as_type<uint>(c);
// while (exp == cbits && !(atomic_compare_exchange_weak_explicit(
//          (device atomic_uint *)p, &exp, as_type<uint>(v),
//          memory_order_relaxed, memory_order_relaxed))) {}
// [float ]id = as_type<float>(exp);
msl::Block AtomicEmitter::float32CAS(StringRef p, StringRef c, StringRef v,
                                     StringRef id, bool declare) {
  std::string exp = fresh(), cbits = fresh();
  auto asU32 = [&](msl::Expr *x) {
    return asType(ctx.scalar(msl::Scalar::U32), x);
  };

  msl::Type *auptr = ctx.deviceAtomicPtr(msl::Scalar::U32);
  msl::Expr *cas =
      casWeak(ctx.cast(CS::CStyle, auptr, ctx.var(p)), exp, asU32(ctx.var(v)));
  msl::Expr *cond =
      ctx.binary(B::LAnd, ctx.binary(B::Eq, ctx.var(exp), ctx.var(cbits)),
                 ctx.unary(msl::UnOp::LNot, ctx.paren(cas)));

  msl::Block block;
  block.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U32), exp, asU32(ctx.var(c))));
  block.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U32), cbits, asU32(ctx.var(c))));
  block.push_back(ctx.whileScope(cond, msl::Block{}));
  msl::Expr *result = asType(ctx.scalar(msl::Scalar::F32), ctx.var(exp));
  if (declare)
    block.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::F32), id, result));
  else
    block.push_back(ctx.assignStmt(ctx.var(id), result));
  return block;
}

// The packed-fp16 CAS: word load + while(true){ read lane; if(got!=cur)break;
// repack; if(cas)break; }
msl::Block AtomicEmitter::packed16CAS(StringRef wordPtr, StringRef isHigh,
                                      StringRef c, StringRef v, StringRef sc,
                                      StringRef id, bool declare) {
  std::string cur = fresh(), lane = fresh(), word = fresh(), newWord = fresh(),
              got = fresh();
  msl::Type *scTy = ctx.named(sc);
  auto asU16 = [&](msl::Expr *x) {
    return asType(ctx.scalar(msl::Scalar::U16), x);
  };
  // as_type<ushort>((sc)(x))
  auto asU16OfSc = [&](StringRef x) {
    return asU16(ctx.paren(ctx.cast(CS::CStyle, scTy, ctx.var(x))));
  };

  msl::Block outer;
  outer.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U16), cur, asU16OfSc(c)));
  outer.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U16), lane, asU16OfSc(v)));
  outer.push_back(ctx.declStmt(
      ctx.scalar(msl::Scalar::U32), word,
      ctx.call(ba::Load, {ctx.var(wordPtr), ctx.lit(border::Relaxed)})));
  if (declare)
    outer.push_back(ctx.declStmt(scTy, id, init0(sc)));
  else
    outer.push_back(ctx.assignStmt(ctx.var(id), init0(sc)));

  // ushort got = (ushort)((isHigh) ? (word >> 16) : (word & 0xffffu));
  msl::Block body;
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::U16), got,
                              ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U16),
                                       packed16Extract(word, isHigh))));
  body.push_back(ctx.assignStmt(ctx.var(id), asType(scTy, ctx.var(got))));
  msl::Block brk;
  brk.push_back(ctx.breakStmt());
  body.push_back(ctx.ifScope(ctx.binary(B::Ne, ctx.var(got), ctx.var(cur)),
                             std::move(brk)));

  // uint newWord = (isHigh) ? ((word & 0x0000ffffu) | ((uint)lane << 16))
  //                         : ((word & 0xffff0000u) | (uint)lane);
  msl::Expr *newBitsU32 =
      ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U32), ctx.var(lane));
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::U32), newWord,
                              packed16Merge(word, isHigh, newBitsU32)));

  msl::Expr *cas = casWeak(ctx.var(wordPtr), word, ctx.var(newWord));
  msl::Block casBrk;
  casBrk.push_back(ctx.breakStmt());
  body.push_back(ctx.ifScope(cas, std::move(casBrk)));

  outer.push_back(ctx.whileScope(nullptr, std::move(body)));
  return outer;
}

// atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
// Metal fences accept only relaxed or seq_cst; seq_cst is the valid
// stronger-than-acquire/release order.
msl::Stmt *AtomicEmitter::deviceFence() {
  return ctx.exprStmt(ctx.call(
      ba::ThreadFence, {ctx.lit(bmem::Device), ctx.lit(border::SeqCst)}));
}

//===----------------------------------------------------------------------===//
// histBinsDecl / histZeroInit / histFetchAdd - threadgroup atomic bins
//===----------------------------------------------------------------------===//

// threadgroup atomic_uint* bins = ((threadgroup atomic_uint*)poolBuf);
msl::Stmt *AtomicEmitter::histBinsDecl(StringRef bins) {
  msl::Type *tgptr =
      ctx.ptr(ctx.atomic(msl::Scalar::U32), msl::AddrSpace::Threadgroup);
  msl::Expr *region = ctx.paren(ctx.cast(CS::CStyle, tgptr, ctx.var(poolBuf)));
  return ctx.declStmt(tgptr, bins, region);
}

// for (uint zi = tid.x; zi < nBinsu; zi += threadsu)
//   atomic_store_explicit(&bins[zi], 0u, memory_order_relaxed);
msl::Stmt *AtomicEmitter::histZeroInit(StringRef bins, StringRef zi,
                                       int64_t nBins, int64_t threads) {
  msl::Stmt *init = ctx.declStmt(ctx.scalar(msl::Scalar::U32), zi,
                                 ctx.member(ctx.var(tidId), "x"));
  msl::Expr *cond =
      ctx.binary(B::Lt, ctx.var(zi), ctx.lit(std::to_string(nBins) + "u"));
  msl::Stmt *step =
      ctx.addAssignStmt(ctx.var(zi), ctx.lit(std::to_string(threads) + "u"));
  msl::Expr *store = ctx.call(
      ba::Store, {ctx.addrOf(ctx.subscript(ctx.var(bins), ctx.var(zi))),
                  ctx.lit("0u"), ctx.lit(border::Relaxed)});
  return ctx.compactForScope(init, cond, step, ctx.exprStmt(store));
}

// if (guard) atomic_fetch_add_explicit(&bins[v], 1u, memory_order_relaxed);
msl::Stmt *AtomicEmitter::histFetchAdd(msl::Expr *guard, StringRef bins,
                                       StringRef v) {
  msl::Expr *add = ctx.call(
      ba::FetchAdd, {ctx.addrOf(ctx.subscript(ctx.var(bins), ctx.var(v))),
                     ctx.lit("1u"), ctx.lit(border::Relaxed)});
  return ctx.compactIf(guard, ctx.exprStmt(add));
}

} // namespace mlir::triton::applegpu
