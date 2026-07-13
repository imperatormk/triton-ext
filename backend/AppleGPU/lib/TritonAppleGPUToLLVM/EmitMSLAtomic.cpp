// EmitMSLAtomic.cpp - atomic RMW / CAS / poll + histogram lowering
// (string + AST forms).
//
// Out-lined AST siblings of the atomic sub-builders in emitAtomicRMW,
// emitAtomicCAS, emitAtomicPoll and emitHistogram (MSLEmitter.h). The CAS /
// spin / zero-init control flow is modelled as real WhileScope / IfScope /
// ForScope node trees (not Raw blocks) - atomics control flow is exactly what
// the node set exists to express. Emission still runs on the string path this
// layer; these nodes only need to exist and print byte-identically once the
// flip layer routes emission here.
//
// INVARIANT: the printer inserts no grouping parens; wherever the string path
// wrapped a subexpression the AST sibling inserts an explicit ctx.paren(...).

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

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
msl::Expr *MSLEmitter::astInit0(StringRef sc) {
  return ctx.lit(sc == "float" || sc == "half" ? "0.0" : "0");
}

// (device <atomicTy>*)p - the atomic-typed device pointer the RMW/poll casts to.
// atomicTy is spelled by name (atomic_int/atomic_uint/atomic_float/...) so the
// pointee is a NamedType, keeping the sibling agnostic to the string path's
// scalar->atomic choice.
msl::Expr *MSLEmitter::astDeviceAtomicPtr(StringRef atomicTy, StringRef p) {
  msl::Type *ptr = ctx.ptr(ctx.named(atomicTy), msl::AddrSpace::Device);
  return ctx.cast(CS::CStyle, ptr, ctx.var(p));
}

// device uchar *bytePtr = (device uchar *)(p);
// size_t wordAddr = (size_t)bytePtr & ~(size_t)3;
// bool isHigh = ((size_t)bytePtr & 2u) != 0u;
// device atomic_uint *wordPtr = (device atomic_uint *)wordAddr;
// Binds wordPtr / isHigh names into the out-params.
msl::Block MSLEmitter::astPacked16Base(StringRef p, std::string &wordPtrOut,
                                       std::string &isHighOut) {
  LocalGen g{nextId};
  std::string bytePtr = g.fresh(), wordAddr = g.fresh(), isHigh = g.fresh(),
              wordPtr = g.fresh();
  msl::Type *ucptr = ctx.ptr(ctx.scalar(msl::Scalar::U8), msl::AddrSpace::Device);
  msl::Type *auptr = ctx.ptr(ctx.named(ba::Uint), msl::AddrSpace::Device);
  auto szOf = [&](StringRef x) {
    return ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::SizeT), ctx.var(x));
  };
  // (size_t)bytePtr & ~(size_t)3
  msl::Expr *addr = ctx.binary(
      B::And, szOf(bytePtr),
      ctx.unary(msl::UnOp::Not,
                ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::SizeT),
                         ctx.lit("3"))));
  // ((size_t)bytePtr & 2u) != 0u
  msl::Expr *hi = ctx.binary(
      B::Ne, ctx.paren(ctx.binary(B::And, szOf(bytePtr), ctx.lit("2u"))),
      ctx.lit("0u"));

  msl::Block block;
  block.push_back(ctx.declStmt(
      ucptr, bytePtr, ctx.cast(CS::CStyle, ucptr, ctx.paren(ctx.var(p)))));
  block.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::SizeT), wordAddr, addr));
  block.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::I1), isHigh, hi));
  block.push_back(ctx.declStmt(
      auptr, wordPtr, ctx.cast(CS::CStyle, auptr, ctx.var(wordAddr))));
  wordPtrOut = wordPtr;
  isHighOut = isHigh;
  return block;
}

//===----------------------------------------------------------------------===//
// emitAtomicRMW - native atomic_fetch_*_explicit call
//===----------------------------------------------------------------------===//

// fn((device atomicTy*)p, v, order[, mem_flags::mem_device]).
// `order` / `memflags` come from MSLConstants so the ordering surface stays
// greppable; the memflags arg is present iff the string path appended `tail`.
msl::Expr *MSLEmitter::astAtomicRmwCall(StringRef fn, StringRef atomicTy,
                                        StringRef p, StringRef v,
                                        StringRef order, bool memFlags) {
  SmallVector<msl::Expr *> args{astDeviceAtomicPtr(atomicTy, p), ctx.var(v),
                                ctx.lit(order)};
  if (memFlags)
    args.push_back(ctx.lit(bmem::Device));
  return ctx.call(fn, args);
}

//===----------------------------------------------------------------------===//
// emitAtomicRMW - fp32 / packed-fp16 emulated CAS loops
//===----------------------------------------------------------------------===//

// device atomic_uint *wordPtr = (device atomic_uint *)(p);
// uint word = atomic_load_explicit(wordPtr, memory_order_relaxed);
// while (true) {
//   id  = as_type<float>(word);
//   float cur = as_type<float>(word);
//   uint newWord = as_type<uint>((float)(newFloatExpr));
//   if (atomic_compare_exchange_weak_explicit(wordPtr, &word, newWord,
//         memory_order_relaxed, memory_order_relaxed)) break;
// }
msl::Block MSLEmitter::astFloat32CASLoop(StringRef p, StringRef curId,
                                         msl::Expr *newFloatExpr, StringRef id) {
  LocalGen g{nextId};
  std::string wordPtr = g.fresh(), word = g.fresh(), newWord = g.fresh();

  msl::Type *auptr = ctx.ptr(ctx.named(ba::Uint), msl::AddrSpace::Device);
  msl::Stmt *wpDecl = ctx.declStmt(
      auptr, wordPtr, ctx.cast(CS::CStyle, auptr, ctx.paren(ctx.var(p))));
  msl::Stmt *wDecl = ctx.declStmt(
      ctx.scalar(msl::Scalar::U32), word,
      ctx.call(ba::Load, {ctx.var(wordPtr), ctx.lit(border::Relaxed)}));

  auto asF32 = [&](msl::Expr *x) {
    return ctx.cast(CS::AsType, ctx.scalar(msl::Scalar::F32), x);
  };
  msl::Block body;
  body.push_back(ctx.assignStmt(ctx.var(id), asF32(ctx.var(word))));
  body.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::F32), curId, asF32(ctx.var(word))));
  msl::Expr *repacked = ctx.cast(
      CS::AsType, ctx.scalar(msl::Scalar::U32),
      ctx.paren(ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::F32),
                         ctx.paren(newFloatExpr))));
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::U32), newWord, repacked));
  msl::Expr *cas =
      ctx.call(ba::CompareExchangeWeak,
               {ctx.var(wordPtr), ctx.addrOf(ctx.var(word)), ctx.var(newWord),
                ctx.lit(border::Relaxed), ctx.lit(border::Relaxed)});
  msl::Block casThen;
  casThen.push_back(ctx.breakStmt());
  body.push_back(ctx.ifScope(cas, std::move(casThen)));

  msl::Block outer;
  outer.push_back(wpDecl);
  outer.push_back(wDecl);
  outer.push_back(ctx.whileScope(nullptr, std::move(body)));
  return outer;
}

// The packed-fp16 RMW CAS loop. `base` supplies the aligned atomic_uint* word
// pointer and the isHigh selector; newHalfExpr references curId.
msl::Block MSLEmitter::astPacked16CASLoop(StringRef wordPtr, StringRef isHigh,
                                          StringRef sc, StringRef curId,
                                          msl::Expr *newHalfExpr, StringRef id) {
  LocalGen g{nextId};
  std::string word = g.fresh(), lane = g.fresh(), newLane = g.fresh(),
              newWord = g.fresh();
  msl::Type *scTy = ctx.named(sc);

  msl::Stmt *wDecl = ctx.declStmt(
      ctx.scalar(msl::Scalar::U32), word,
      ctx.call(ba::Load, {ctx.var(wordPtr), ctx.lit(border::Relaxed)}));

  // (ushort)((isHigh) ? (word >> 16) : (word & 0xffffu))
  msl::Expr *hi =
      ctx.paren(ctx.binary(B::Shr, ctx.var(word), ctx.lit("16")));
  msl::Expr *lo =
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0xffffu")));
  msl::Expr *sel =
      ctx.paren(ctx.ternary(ctx.paren(ctx.var(isHigh)), hi, lo));
  msl::Expr *laneInit = ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U16), sel);

  auto asType = [&](msl::Type *t, msl::Expr *x) {
    return ctx.cast(CS::AsType, t, x);
  };
  msl::Block body;
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::U16), lane, laneInit));
  body.push_back(ctx.assignStmt(ctx.var(id), asType(scTy, ctx.var(lane))));
  body.push_back(ctx.declStmt(scTy, curId, asType(scTy, ctx.var(lane))));
  body.push_back(ctx.declStmt(scTy, newLane, newHalfExpr));

  // (isHigh) ? ((word & 0x0000ffffu) | ((uint)as_type<ushort>(newLane) << 16))
  //          : ((word & 0xffff0000u) | (uint)as_type<ushort>(newLane))
  msl::Expr *newBitsU16 =
      asType(ctx.scalar(msl::Scalar::U16), ctx.var(newLane));
  msl::Expr *newBitsHi = ctx.paren(ctx.binary(
      B::Shl, ctx.paren(ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U32),
                                 newBitsU16)),
      ctx.lit("16")));
  msl::Expr *highWord = ctx.paren(ctx.binary(
      B::Or,
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0x0000ffffu"))),
      newBitsHi));
  msl::Expr *newBitsLo = ctx.paren(ctx.binary(
      B::Or,
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0xffff0000u"))),
      ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U32), newBitsU16)));
  msl::Expr *newWordInit =
      ctx.ternary(ctx.paren(ctx.var(isHigh)), highWord, newBitsLo);
  body.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U32), newWord, newWordInit));

  msl::Expr *cas =
      ctx.call(ba::CompareExchangeWeak,
               {ctx.var(wordPtr), ctx.addrOf(ctx.var(word)), ctx.var(newWord),
                ctx.lit(border::Relaxed), ctx.lit(border::Relaxed)});
  msl::Block casThen;
  casThen.push_back(ctx.breakStmt());
  body.push_back(ctx.ifScope(cas, std::move(casThen)));

  msl::Block outer;
  outer.push_back(wDecl);
  outer.push_back(ctx.whileScope(nullptr, std::move(body)));
  return outer;
}

//===----------------------------------------------------------------------===//
// emitAtomicCAS - int32 / float32 / packed16 compare-exchange
//===----------------------------------------------------------------------===//

// sc exp = c;
// while (exp == c && !(atomic_compare_exchange_weak_explicit(
//          (device atomic_int *)p, &exp, v, memory_order_relaxed,
//          memory_order_relaxed))) {}
// [sc ]id = exp;
// Metal device atomics are relaxed-only; acquire/release orders are invalid MSL.
msl::Block MSLEmitter::astInt32CAS(StringRef p, StringRef c, StringRef v,
                                   StringRef sc, StringRef id, bool declare) {
  LocalGen g{nextId};
  std::string exp = g.fresh();
  msl::Type *scTy = ctx.named(sc);

  msl::Type *aiptr = ctx.named(("device " + std::string(ba::Int) + " *"));
  msl::Expr *cas = ctx.call(
      ba::CompareExchangeWeak,
      {ctx.cast(CS::CStyle, aiptr, ctx.var(p)), ctx.addrOf(ctx.var(exp)),
       ctx.var(v), ctx.lit(border::Relaxed), ctx.lit(border::Relaxed)});
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
msl::Block MSLEmitter::astFloat32CAS(StringRef p, StringRef c, StringRef v,
                                     StringRef id, bool declare) {
  LocalGen g{nextId};
  std::string exp = g.fresh(), cbits = g.fresh();
  auto asU32 = [&](msl::Expr *x) {
    return ctx.cast(CS::AsType, ctx.scalar(msl::Scalar::U32), x);
  };

  msl::Type *auptr = ctx.named(("device " + std::string(ba::Uint) + " *"));
  msl::Expr *cas = ctx.call(
      ba::CompareExchangeWeak,
      {ctx.cast(CS::CStyle, auptr, ctx.var(p)), ctx.addrOf(ctx.var(exp)),
       asU32(ctx.var(v)), ctx.lit(border::Relaxed), ctx.lit(border::Relaxed)});
  msl::Expr *cond =
      ctx.binary(B::LAnd, ctx.binary(B::Eq, ctx.var(exp), ctx.var(cbits)),
                 ctx.unary(msl::UnOp::LNot, ctx.paren(cas)));

  msl::Block block;
  block.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U32), exp, asU32(ctx.var(c))));
  block.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U32), cbits, asU32(ctx.var(c))));
  block.push_back(ctx.whileScope(cond, msl::Block{}));
  msl::Expr *result =
      ctx.cast(CS::AsType, ctx.scalar(msl::Scalar::F32), ctx.var(exp));
  if (declare)
    block.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::F32), id, result));
  else
    block.push_back(ctx.assignStmt(ctx.var(id), result));
  return block;
}

// The packed-fp16 CAS: word load + while(true){ read lane; if(got!=cur)break;
// repack; if(cas)break; } (mirrors emitPacked16CAS).
msl::Block MSLEmitter::astPacked16CAS(StringRef wordPtr, StringRef isHigh,
                                      StringRef c, StringRef v, StringRef sc,
                                      StringRef id, bool declare) {
  LocalGen g{nextId};
  std::string cur = g.fresh(), lane = g.fresh(), word = g.fresh(),
              newWord = g.fresh(), got = g.fresh();
  msl::Type *scTy = ctx.named(sc);
  auto asU16 = [&](msl::Expr *x) {
    return ctx.cast(CS::AsType, ctx.scalar(msl::Scalar::U16), x);
  };
  // as_type<ushort>((sc)(x))
  auto asU16OfSc = [&](StringRef x) {
    return asU16(ctx.paren(ctx.cast(CS::CStyle, scTy, ctx.var(x))));
  };

  msl::Block outer;
  outer.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::U16), cur, asU16OfSc(c)));
  outer.push_back(
      ctx.declStmt(ctx.scalar(msl::Scalar::U16), lane, asU16OfSc(v)));
  outer.push_back(ctx.declStmt(
      ctx.scalar(msl::Scalar::U32), word,
      ctx.call(ba::Load, {ctx.var(wordPtr), ctx.lit(border::Relaxed)})));
  if (declare)
    outer.push_back(ctx.declStmt(scTy, id, astInit0(sc)));
  else
    outer.push_back(ctx.assignStmt(ctx.var(id), astInit0(sc)));

  // ushort got = (ushort)((isHigh) ? (word >> 16) : (word & 0xffffu));
  msl::Expr *hi = ctx.paren(ctx.binary(B::Shr, ctx.var(word), ctx.lit("16")));
  msl::Expr *lo =
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0xffffu")));
  msl::Expr *sel = ctx.paren(ctx.ternary(ctx.paren(ctx.var(isHigh)), hi, lo));
  msl::Block body;
  body.push_back(ctx.declStmt(ctx.scalar(msl::Scalar::U16), got,
                              ctx.cast(CS::CStyle,
                                       ctx.scalar(msl::Scalar::U16), sel)));
  body.push_back(ctx.assignStmt(
      ctx.var(id), ctx.cast(CS::AsType, scTy, ctx.var(got))));
  msl::Block brk;
  brk.push_back(ctx.breakStmt());
  body.push_back(ctx.ifScope(ctx.binary(B::Ne, ctx.var(got), ctx.var(cur)),
                             std::move(brk)));

  // uint newWord = (isHigh) ? ((word & 0x0000ffffu) | ((uint)lane << 16))
  //                         : ((word & 0xffff0000u) | (uint)lane);
  msl::Expr *laneHi = ctx.paren(ctx.binary(
      B::Shl, ctx.paren(ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U32),
                                 ctx.var(lane))),
      ctx.lit("16")));
  msl::Expr *highWord = ctx.paren(ctx.binary(
      B::Or,
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0x0000ffffu"))),
      laneHi));
  msl::Expr *lowWord = ctx.paren(ctx.binary(
      B::Or,
      ctx.paren(ctx.binary(B::And, ctx.var(word), ctx.lit("0xffff0000u"))),
      ctx.cast(CS::CStyle, ctx.scalar(msl::Scalar::U32), ctx.var(lane))));
  body.push_back(ctx.declStmt(
      ctx.scalar(msl::Scalar::U32), newWord,
      ctx.ternary(ctx.paren(ctx.var(isHigh)), highWord, lowWord)));

  msl::Expr *cas =
      ctx.call(ba::CompareExchangeWeak,
               {ctx.var(wordPtr), ctx.addrOf(ctx.var(word)), ctx.var(newWord),
                ctx.lit(border::Relaxed), ctx.lit(border::Relaxed)});
  msl::Block casBrk;
  casBrk.push_back(ctx.breakStmt());
  body.push_back(ctx.ifScope(cas, std::move(casBrk)));

  outer.push_back(ctx.whileScope(nullptr, std::move(body)));
  return outer;
}

// atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
// Metal fences accept only relaxed or seq_cst; seq_cst is the valid
// stronger-than-acquire order.
msl::Stmt *MSLEmitter::astAcquireFence() {
  return ctx.exprStmt(ctx.call(
      ba::ThreadFence, {ctx.lit(bmem::Device), ctx.lit(border::SeqCst)}));
}

//===----------------------------------------------------------------------===//
// emitAtomicPoll - single-probe spin loop
//===----------------------------------------------------------------------===//

// while (loadExpr != want) {}
msl::Stmt *MSLEmitter::astPollSpin(msl::Expr *loadExpr, StringRef want) {
  return ctx.whileScope(ctx.binary(B::Ne, loadExpr, ctx.var(want)),
                        msl::Block{});
}

// The 64-bit poll's load: `volatile device ulong *wordPtr = (volatile device
// ulong *)(p);` + `(*wordPtr)`. Returns the deref load expr; binds the ptr decl
// into `out`.
msl::Expr *MSLEmitter::astPoll64Load(StringRef p, StringRef wordPtr,
                                     msl::Stmt *&out) {
  msl::Type *vptr = ctx.ptr(ctx.scalar(msl::Scalar::U64), msl::AddrSpace::Device,
                            /*coherent=*/false, /*vol=*/true);
  out = ctx.declStmt(vptr, wordPtr, ctx.cast(CS::CStyle, vptr, ctx.paren(ctx.var(p))));
  return ctx.paren(ctx.deref(ctx.var(wordPtr)));
}

//===----------------------------------------------------------------------===//
// emitHistogram - threadgroup atomic bins
//===----------------------------------------------------------------------===//

// threadgroup atomic_uint* bins = ((threadgroup atomic_uint*)poolBuf);
msl::Stmt *MSLEmitter::astHistBinsDecl(StringRef bins) {
  msl::Type *tgptr =
      ctx.ptr(ctx.named(ba::Uint), msl::AddrSpace::Threadgroup);
  return ctx.declStmt(tgptr, bins, astPoolRegion(0, ba::Uint));
}

// for (uint zi = tid.x; zi < nBinsu; zi += threadsu)
//   atomic_store_explicit(&bins[zi], 0u, memory_order_relaxed);
msl::Stmt *MSLEmitter::astHistZeroInit(StringRef bins, StringRef zi,
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
msl::Stmt *MSLEmitter::astHistFetchAdd(msl::Expr *guard, StringRef bins,
                                       StringRef v) {
  msl::Expr *add = ctx.call(
      ba::FetchAdd, {ctx.addrOf(ctx.subscript(ctx.var(bins), ctx.var(v))),
                     ctx.lit("1u"), ctx.lit(border::Relaxed)});
  return ctx.compactIf(guard, ctx.exprStmt(add));
}

} // namespace mlir::triton::applegpu
