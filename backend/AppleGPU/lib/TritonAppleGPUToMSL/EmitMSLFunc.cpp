// EmitMSLFunc.cpp - function / scope / control-flow AST builders.
//
// AST builders for the function scopes (driven from emitFunc / emitDeviceFunc /
// emitDeviceFuncProto / declRetStruct in MSLEmitter.h) plus the per-op dispatch
// spine (emitOp): each op's body is walked into a Block and wrapped with a
// scope builder here, and emitOp is the sole minter of fresh() names.
//
// Control flow is modelled with real scope nodes (KernelFn/DeviceFn/ForScope/
// TripCountForScope/IfScope/WhileScope/StateMachineScope) - never Raw blocks.
//
// INVARIANTS:
//  - wide-IV i64 loops carry the induction-var decl as the FIRST body stmt
//    (never floated into the for-header) so the AGX i65 Gauss-sum fold can't
//    fire; the break is an IfScope + BreakStmt.
//  - a barrier is a BarrierStmt node (the printer's peephole collapses adjacent
//    ones); builders here never spell threadgroup_barrier text.

#include "MSLConstants.h"
#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

using B = msl::BinOp;
using CS = msl::Cast::Style;

//===----------------------------------------------------------------------===//
// Dead local elimination
//===----------------------------------------------------------------------===//

namespace {
// Collects every name that appears in a *read* position, then drops DeclStmts
// whose name is never read. The emitter materialises one local per SSA value,
// so any op whose result a later op folded into its own expression leaves a
// declaration behind with no use. The in-process compile path never surfaced
// these, but the toolchain path (used when a kernel links the async-copy shim)
// warns on each one.
bool initHasCall(msl::Expr *e);

struct DeadLocalMarker {
  llvm::DenseSet<llvm::StringRef> used;
  // How many times each name appears in a read position, so a self-sustaining
  // induction cycle can be told apart from a genuinely consumed value.
  llvm::DenseMap<llvm::StringRef, unsigned> readCount;
  // Raw text is opaque to the AST, so any name it mentions must be assumed
  // live. Rather than parse it, a Raw node anywhere disables the pass.
  bool sawRaw = false;

  void expr(msl::Expr *e) {
    if (!e)
      return;
    switch (e->kind) {
    case msl::Expr::Kind::VarRef: {
      llvm::StringRef n = llvm::cast<msl::VarRef>(e)->name;
      used.insert(n);
      ++readCount[n];
      return;
    }
    case msl::Expr::Kind::Raw:
      sawRaw = true;
      return;
    case msl::Expr::Kind::Literal:
      return;
    case msl::Expr::Kind::Binary: {
      auto *b = llvm::cast<msl::Binary>(e);
      expr(b->lhs);
      expr(b->rhs);
      return;
    }
    case msl::Expr::Kind::Unary:
      expr(llvm::cast<msl::Unary>(e)->x);
      return;
    case msl::Expr::Kind::Cast:
      expr(llvm::cast<msl::Cast>(e)->x);
      return;
    case msl::Expr::Kind::Call:
      for (msl::Expr *a : llvm::cast<msl::Call>(e)->args)
        expr(a);
      return;
    case msl::Expr::Kind::Ternary: {
      auto *t = llvm::cast<msl::Ternary>(e);
      expr(t->c);
      expr(t->a);
      expr(t->b);
      return;
    }
    case msl::Expr::Kind::Subscript: {
      auto *s = llvm::cast<msl::Subscript>(e);
      expr(s->base);
      expr(s->idx);
      return;
    }
    case msl::Expr::Kind::Member:
      expr(llvm::cast<msl::Member>(e)->base);
      return;
    case msl::Expr::Kind::Deref:
      expr(llvm::cast<msl::Deref>(e)->x);
      return;
    case msl::Expr::Kind::AddrOf:
      expr(llvm::cast<msl::AddrOf>(e)->x);
      return;
    case msl::Expr::Kind::Paren:
      expr(llvm::cast<msl::Paren>(e)->x);
      return;
    }
  }

  void block(const msl::Block &b) {
    for (msl::Stmt *s : b)
      stmt(s);
  }

  void stmt(msl::Stmt *s) {
    if (!s)
      return;
    switch (s->kind) {
    case msl::Stmt::Kind::Decl:
      // The initialiser is a read; the declared name itself is not.
      expr(llvm::cast<msl::DeclStmt>(s)->init);
      return;
    case msl::Stmt::Kind::ArrayDecl:
      return;
    case msl::Stmt::Kind::Assign: {
      auto *a = llvm::cast<msl::AssignStmt>(s);
      // A name assigned a side-effecting call is live even if nothing has read
      // it yet: dropping its declaration leaves the assignment referring to an
      // identifier that no longer exists.
      if (initHasCall(a->rhs))
        if (auto *lv = llvm::dyn_cast<msl::VarRef>(a->lhs))
          used.insert(lv->name);
      // A plain `v = rhs` does not read v, but `v[i] = rhs`, `*v = rhs` and
      // `v += rhs` all do.
      if (a->compound != msl::AssignStmt::Compound::None ||
          !llvm::isa<msl::VarRef>(a->lhs))
        expr(a->lhs);
      expr(a->rhs);
      return;
    }
    case msl::Stmt::Kind::Expr:
      expr(llvm::cast<msl::ExprStmt>(s)->e);
      return;
    case msl::Stmt::Kind::Return: {
      auto *r = llvm::cast<msl::ReturnStmt>(s);
      expr(r->val);
      for (msl::Expr *f : r->structFields)
        expr(f);
      return;
    }
    case msl::Stmt::Kind::Raw:
      sawRaw = true;
      return;
    case msl::Stmt::Kind::Break:
    case msl::Stmt::Kind::Continue:
    case msl::Stmt::Kind::Barrier:
      return;
    case msl::Stmt::Kind::CompactIf: {
      auto *c = llvm::cast<msl::CompactIfStmt>(s);
      expr(c->cond);
      stmt(c->then);
      return;
    }
    case msl::Stmt::Kind::KernelFn:
      block(llvm::cast<msl::KernelFn>(s)->body);
      return;
    case msl::Stmt::Kind::DeviceFn:
      block(llvm::cast<msl::DeviceFn>(s)->body);
      return;
    case msl::Stmt::Kind::ForScope: {
      auto *f = llvm::cast<msl::ForScope>(s);
      stmt(f->initDecl);
      expr(f->cond);
      stmt(f->step);
      block(f->body);
      return;
    }
    case msl::Stmt::Kind::TripCountForScope: {
      auto *f = llvm::cast<msl::TripCountForScope>(s);
      stmt(f->ivDecl);
      expr(f->guard);
      block(f->body);
      return;
    }
    case msl::Stmt::Kind::IfScope: {
      auto *i = llvm::cast<msl::IfScope>(s);
      expr(i->cond);
      block(i->thenB);
      block(i->elseB);
      return;
    }
    case msl::Stmt::Kind::WhileScope: {
      auto *w = llvm::cast<msl::WhileScope>(s);
      expr(w->cond);
      block(w->body);
      return;
    }
    case msl::Stmt::Kind::StateMachineScope: {
      auto *m = llvm::cast<msl::StateMachineScope>(s);
      used.insert(m->stateVar);
      for (auto &c : m->cases)
        block(c.body);
      return;
    }
    case msl::Stmt::Kind::PlainScope:
      block(llvm::cast<msl::PlainScope>(s)->body);
      return;
    }
  }
};

// An initialiser that calls a function may have side effects, so its decl is
// only removable if the call itself is known pure. Nothing else in the emitted
// AST can have a side effect in a read position.
bool initHasCall(msl::Expr *e) {
  if (!e)
    return false;
  switch (e->kind) {
  case msl::Expr::Kind::Call:
  case msl::Expr::Kind::Raw:
    return true;
  case msl::Expr::Kind::VarRef:
  case msl::Expr::Kind::Literal:
    return false;
  case msl::Expr::Kind::Binary: {
    auto *b = llvm::cast<msl::Binary>(e);
    return initHasCall(b->lhs) || initHasCall(b->rhs);
  }
  case msl::Expr::Kind::Unary:
    return initHasCall(llvm::cast<msl::Unary>(e)->x);
  case msl::Expr::Kind::Cast:
    return initHasCall(llvm::cast<msl::Cast>(e)->x);
  case msl::Expr::Kind::Ternary: {
    auto *t = llvm::cast<msl::Ternary>(e);
    return initHasCall(t->c) || initHasCall(t->a) || initHasCall(t->b);
  }
  case msl::Expr::Kind::Subscript: {
    auto *s = llvm::cast<msl::Subscript>(e);
    return initHasCall(s->base) || initHasCall(s->idx);
  }
  case msl::Expr::Kind::Member:
    return initHasCall(llvm::cast<msl::Member>(e)->base);
  case msl::Expr::Kind::Deref:
    return initHasCall(llvm::cast<msl::Deref>(e)->x);
  case msl::Expr::Kind::AddrOf:
    return initHasCall(llvm::cast<msl::AddrOf>(e)->x);
  case msl::Expr::Kind::Paren:
    return initHasCall(llvm::cast<msl::Paren>(e)->x);
  }
  return true;
}

unsigned dropDeadDecls(msl::Block &b,
                       const llvm::DenseSet<llvm::StringRef> &used);

unsigned dropDeadDeclsIn(msl::Stmt *s,
                         const llvm::DenseSet<llvm::StringRef> &used) {
  if (!s)
    return 0;
  switch (s->kind) {
  case msl::Stmt::Kind::KernelFn:
    return dropDeadDecls(llvm::cast<msl::KernelFn>(s)->body, used);
  case msl::Stmt::Kind::DeviceFn:
    return dropDeadDecls(llvm::cast<msl::DeviceFn>(s)->body, used);
  case msl::Stmt::Kind::ForScope:
    return dropDeadDecls(llvm::cast<msl::ForScope>(s)->body, used);
  case msl::Stmt::Kind::TripCountForScope:
    return dropDeadDecls(llvm::cast<msl::TripCountForScope>(s)->body, used);
  case msl::Stmt::Kind::IfScope: {
    auto *i = llvm::cast<msl::IfScope>(s);
    return dropDeadDecls(i->thenB, used) + dropDeadDecls(i->elseB, used);
  }
  case msl::Stmt::Kind::WhileScope:
    return dropDeadDecls(llvm::cast<msl::WhileScope>(s)->body, used);
  case msl::Stmt::Kind::StateMachineScope: {
    unsigned n = 0;
    for (auto &c : llvm::cast<msl::StateMachineScope>(s)->cases)
      n += dropDeadDecls(c.body, used);
    return n;
  }
  case msl::Stmt::Kind::PlainScope:
    return dropDeadDecls(llvm::cast<msl::PlainScope>(s)->body, used);
  default:
    return 0;
  }
}

// A loop-carried induction pair keeps itself alive: `adv = base + step;` and
// `base = adv;` make each name a reader of the other, so plain liveness never
// removes them even when nothing else reads either. Collect the pairs whose
// names are read exactly twice -- once in the advance, once in the carry --
// which means every read is internal to the cycle.
void findDeadInductionCycles(
    const msl::Block &b, const llvm::DenseMap<llvm::StringRef, unsigned> &reads,
    llvm::DenseSet<llvm::StringRef> &dead) {
  // adv name -> base name, for `adv = base + step;` decls in this block.
  llvm::DenseMap<llvm::StringRef, llvm::StringRef> advOf;
  for (const msl::Stmt *s : b) {
    if (auto *d = llvm::dyn_cast<msl::DeclStmt>(s)) {
      auto *bin = llvm::dyn_cast_or_null<msl::Binary>(d->init);
      if (bin && bin->op == msl::BinOp::Add)
        if (auto *lhs = llvm::dyn_cast<msl::VarRef>(bin->lhs))
          if (llvm::isa<msl::VarRef>(bin->rhs))
            advOf[d->name] = lhs->name;
      continue;
    }
    if (auto *a = llvm::dyn_cast<msl::AssignStmt>(s)) {
      if (a->compound != msl::AssignStmt::Compound::None)
        continue;
      auto *lv = llvm::dyn_cast<msl::VarRef>(a->lhs);
      auto *rv = llvm::dyn_cast<msl::VarRef>(a->rhs);
      if (!lv || !rv)
        continue;
      auto it = advOf.find(rv->name);
      if (it == advOf.end() || it->second != lv->name)
        continue;
      // `base` is read once by the advance, `adv` once by the carry. Any other
      // read means the pointer is genuinely used.
      auto rb = reads.find(lv->name), ra = reads.find(rv->name);
      if (rb != reads.end() && rb->second == 1 && ra != reads.end() &&
          ra->second == 1) {
        dead.insert(lv->name);
        dead.insert(rv->name);
      }
      continue;
    }
    // Recurse into scopes.
    if (auto *f = llvm::dyn_cast<msl::ForScope>(s))
      findDeadInductionCycles(f->body, reads, dead);
    else if (auto *f = llvm::dyn_cast<msl::TripCountForScope>(s))
      findDeadInductionCycles(f->body, reads, dead);
    else if (auto *i = llvm::dyn_cast<msl::IfScope>(s)) {
      findDeadInductionCycles(i->thenB, reads, dead);
      findDeadInductionCycles(i->elseB, reads, dead);
    } else if (auto *w = llvm::dyn_cast<msl::WhileScope>(s))
      findDeadInductionCycles(w->body, reads, dead);
    else if (auto *p = llvm::dyn_cast<msl::PlainScope>(s))
      findDeadInductionCycles(p->body, reads, dead);
  }
}

// Drops the decls and carries of a dead induction cycle.
unsigned dropDeadCycle(msl::Block &b,
                       const llvm::DenseSet<llvm::StringRef> &dead) {
  unsigned removed = 0;
  msl::Block keep;
  keep.reserve(b.size());
  for (msl::Stmt *s : b) {
    if (auto *d = llvm::dyn_cast<msl::DeclStmt>(s))
      if (dead.count(d->name)) {
        ++removed;
        continue;
      }
    if (auto *a = llvm::dyn_cast<msl::AssignStmt>(s))
      if (auto *lv = llvm::dyn_cast<msl::VarRef>(a->lhs))
        if (dead.count(lv->name) && !initHasCall(a->rhs)) {
          ++removed;
          continue;
        }
    if (auto *f = llvm::dyn_cast<msl::ForScope>(s))
      removed += dropDeadCycle(f->body, dead);
    else if (auto *f = llvm::dyn_cast<msl::TripCountForScope>(s))
      removed += dropDeadCycle(f->body, dead);
    else if (auto *i = llvm::dyn_cast<msl::IfScope>(s)) {
      removed += dropDeadCycle(i->thenB, dead);
      removed += dropDeadCycle(i->elseB, dead);
    } else if (auto *w = llvm::dyn_cast<msl::WhileScope>(s))
      removed += dropDeadCycle(w->body, dead);
    else if (auto *p = llvm::dyn_cast<msl::PlainScope>(s))
      removed += dropDeadCycle(p->body, dead);
    keep.push_back(s);
  }
  b = std::move(keep);
  return removed;
}

// Returns the number of decls removed, so the driver can iterate to fixpoint.
unsigned dropDeadDecls(msl::Block &b,
                       const llvm::DenseSet<llvm::StringRef> &used) {
  unsigned removed = 0;
  msl::Block keep;
  keep.reserve(b.size());
  for (msl::Stmt *s : b) {
    if (auto *d = llvm::dyn_cast<msl::DeclStmt>(s))
      if (!used.count(d->name) && !initHasCall(d->init)) {
        ++removed;
        continue;
      }
    // A plain `v = rhs` to a name nothing reads dies with its declaration;
    // leaving it behind would reference an identifier that no longer exists.
    msl::Stmt *assign = s;
    if (auto *c = llvm::dyn_cast<msl::CompactIfStmt>(s))
      assign = c->then;
    if (auto *a = llvm::dyn_cast_or_null<msl::AssignStmt>(assign))
      if (a->compound == msl::AssignStmt::Compound::None)
        if (auto *lhs = llvm::dyn_cast<msl::VarRef>(a->lhs))
          if (!used.count(lhs->name) && !initHasCall(a->rhs)) {
            ++removed;
            continue;
          }
    removed += dropDeadDeclsIn(s, used);
    keep.push_back(s);
  }
  b = std::move(keep);
  return removed;
}
} // namespace

// Removes locals the emitter declared but nothing reads. Runs to fixpoint:
// dropping one decl can orphan the one feeding its initialiser.
static void eliminateDeadLocals(msl::KernelFn *fn) {
  for (int pass = 0; pass < 8; ++pass) {
    DeadLocalMarker m;
    m.stmt(fn);
    if (m.sawRaw)
      return;
    llvm::DenseSet<llvm::StringRef> deadCycle;
    findDeadInductionCycles(fn->body, m.readCount, deadCycle);
    unsigned n = dropDeadDecls(fn->body, m.used);
    if (!deadCycle.empty())
      n += dropDeadCycle(fn->body, deadCycle);
    if (n == 0)
      return;
  }
}

//===----------------------------------------------------------------------===//
// Return-struct type + device return type
//===----------------------------------------------------------------------===//

// The `struct fn_<name>_ret { sc f0; ... };` declaration. The field body has no
// dedicated node kind (a struct decl is not a Stmt in the set), so it is a
// RawStmt - the one design-sanctioned escape for a leaf with no node.
msl::Stmt *MSLEmitter::retStructDecl(tt::FuncOp func) {
  auto results = func.getFunctionType().getResults();
  std::string name = mslDeviceFuncName(func.getName()) + "_ret";
  std::string body = "struct " + name + " {\n";
  if (isTensorResult(results)) {
    std::string sc =
        mslScalarType(cast<RankedTensorType>(results[0]).getElementType());
    Value res = func.getBody().front().getTerminator()->getOperand(0);
    int rc = regCount(res);
    for (int i = 0; i < rc; ++i)
      body += "  " + sc + " f" + std::to_string(i) + ";\n";
  } else {
    for (auto [i, ty] : llvm::enumerate(results))
      body += "  " + mslScalarType(ty) + " f" + std::to_string(i) + ";\n";
  }
  body += "};";
  return ctx.rawStmt(body);
}

msl::NamedType *MSLEmitter::retStructType(tt::FuncOp func) {
  return ctx.named(mslDeviceFuncName(func.getName()) + "_ret");
}

msl::Type *MSLEmitter::deviceRetType(tt::FuncOp func) {
  auto results = func.getFunctionType().getResults();
  if (results.empty())
    return ctx.scalar(msl::Scalar::Void);
  if (isTensorResult(results))
    return retStructType(func);
  if (results.size() == 1)
    return storageType(results[0]);
  return retStructType(func);
}

//===----------------------------------------------------------------------===//
// Kernel / device-function signatures
//===----------------------------------------------------------------------===//

// `int lane = (int)(tid.x & 31u); int warp = (int)(tid.x >> 5);` using the
// already-minted laneId/warpId/tidId members (kernel + device-fn prologues).
llvm::SmallVector<msl::Stmt *, 2> MSLEmitter::laneWarpProlog() {
  msl::Type *i32 = ctx.scalar(msl::Scalar::I32);
  msl::Expr *lane =
      ctx.cast(CS::CStyle, i32,
               ctx.paren(ctx.binary(B::And, ctx.member(ctx.var(tidId), "x"),
                                    ctx.lit("31u"))));
  msl::Expr *warp =
      ctx.cast(CS::CStyle, i32,
               ctx.paren(ctx.binary(B::Shr, ctx.member(ctx.var(tidId), "x"),
                                    ctx.lit("5"))));
  return {ctx.declStmt(i32, laneId, lane), ctx.declStmt(i32, warpId, warp)};
}

// Build the device-fn param list. `bind`:
// true mints fresh() names (the definition), false uses aN/__tgpos/... (the
// prototype). Trailing thread-context uint3s + optional threadgroup pool ptr.
llvm::SmallVector<msl::Param, 8> MSLEmitter::deviceParams(tt::FuncOp func,
                                                          int &id, bool bind) {
  LocalGen g{id};
  auto fnTy = func.getFunctionType();
  llvm::SmallVector<msl::Param, 8> params;
  for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
    std::string id = bind ? g.fresh() : ("a" + std::to_string(i));
    if (auto pt = dyn_cast<tt::PointerType>(argTy))
      params.push_back(ctx.param(
          ctx.ptr(scalarType(pt.getPointeeType()), msl::AddrSpace::Device),
          id));
    else
      params.push_back(ctx.param(scalarType(argTy), id));
  }
  msl::Type *u3 = ctx.vector(msl::Scalar::U32, 3);
  params.push_back(ctx.param(u3, bind ? g.fresh() : "__tgpos"));
  params.push_back(ctx.param(u3, bind ? g.fresh() : "__tid"));
  params.push_back(ctx.param(u3, bind ? g.fresh() : "__numtg"));
  if (globalPoolBytes > 0)
    params.push_back(ctx.param(
        ctx.ptr(ctx.scalar(msl::Scalar::I8), msl::AddrSpace::Threadgroup),
        bind ? g.fresh() : "__poolptr"));
  return params;
}

msl::DeviceFn *MSLEmitter::deviceProto(tt::FuncOp func) {
  int id = nextId;
  return ctx.deviceFn(deviceRetType(func), mslDeviceFuncName(func.getName()),
                      deviceParams(func, id, /*bind=*/false), msl::Block{});
}

//===----------------------------------------------------------------------===//
// Return
//===----------------------------------------------------------------------===//

msl::Stmt *MSLEmitter::emitReturn(tt::ReturnOp op) {
  unsigned n = op.getNumOperands();
  if (n == 0)
    return ctx.returnStmt();
  if (n == 1 && !isa<RankedTensorType>(op.getOperand(0).getType()))
    return ctx.returnStmt(ctx.var(names(op.getOperand(0))[0]));
  llvm::SmallVector<msl::Expr *, 4> fields;
  if (n == 1) {
    for (const std::string &nm : names(op.getOperand(0)))
      fields.push_back(ctx.var(nm));
  } else {
    for (Value v : op.getOperands())
      fields.push_back(ctx.var(scalarName(v)));
  }
  return ctx.returnStmt(nullptr, fields);
}

// Mirrors the conditions under which emitMath / the fp_to_fp narrowing path
// / the emitAtomic* families emit a call to each preamble helper. Must stay in
// lockstep
// with those emit sites: a missed helper is a compile error in the MSL.
msl::HelperSet MSLEmitter::scanHelpers() {
  msl::HelperSet h;
  auto elemOf = [](Type t) {
    if (auto rt = dyn_cast<RankedTensorType>(t))
      t = rt.getElementType();
    return t;
  };
  mod.walk([&](Operation *op) {
    for (Type t : op->getOperandTypes())
      h.fp8 |= isFp8Type(elemOf(t));
    for (Type t : op->getResultTypes())
      h.fp8 |= isFp8Type(elemOf(t));

    if (op->getName().getStringRef() == "math.erf")
      h.erf = true;

    if (isa<arith::TruncFOp, tt::FpToFpOp>(op)) {
      Type dstE = elementScalarType(op->getResult(0).getType());
      Type srcE = elementScalarType(op->getOperand(0).getType());
      bool toHalf = dstE.isF16() || dstE.isBF16();
      if (srcE.isF32() && toHalf && !isFp8Type(dstE) && !isFp8Type(srcE)) {
        bool rtz = false, narrow = !isa<tt::FpToFpOp>(op);
        if (auto f = dyn_cast<tt::FpToFpOp>(op))
          if (auto rnd = f.getRounding()) {
            narrow = true;
            rtz = *rnd == tt::RoundingMode::RTZ;
          }
        if (narrow) {
          if (rtz)
            (dstE.isF16() ? h.rtzHalf : h.rtzBfloat) = true;
          else
            (dstE.isF16() ? h.rtneHalf : h.rtneBfloat) = true;
        }
      }
    }

    if (auto ar = dyn_cast<tt::AtomicRMWOp>(op)) {
      Type sTy = elementScalarType(ar.getResult().getType());
      if (isa<FloatType>(sTy)) {
        unsigned bw = sTy.getIntOrFloatBitWidth();
        tt::RMWOp kind = ar.getAtomicRmwOp();
        bool native =
            bw == 32 && (kind == tt::RMWOp::ADD || kind == tt::RMWOp::FADD);
        if (!native) {
          if (bw == 32)
            h.atomicF32 = true;
          else if (bw == 16) {
            // The packed16 CAS loop narrows through tt_rtne_int_*, passed as a
            // template argument, so the matching narrowing helper must be
            // emitted alongside the shared template.
            h.atomicPacked16 = true;
            if (sTy.isF16())
              h.rtneIntHalf = true;
            else
              h.rtneIntBfloat = true;
          }
        }
      }
    }
  });
  // The preamble is printed before any body is emitted, and whether a given dot
  // takes the DMA path depends on emission-time state (the fused phase), so it
  // cannot be predicted reliably here. Declaring the shim entry points whenever
  // the gate is on and any dot could match is safe: unused extern "C"
  // declarations cost nothing, whereas a missing one is a compile error.
  // The declarations are also what select the shim-linking compile path, so
  // this must not be narrower than what the emitter actually emits.
  if (dmaStagingEnabled()) {
    mod.walk([&](tt::DotOp d) { h.tgAsyncCopy = true; });
    mod.walk([&](ttg::AsyncCopyGlobalToLocalOp) { h.tgAsyncCopy = true; });
  }
  return h;
}

LogicalResult MSLEmitter::emit() {
  msl::HelperSet helpers = scanHelpers();

  msl::MSLPrinter preamblePrinter(os);
  preamblePrinter.printPreamble(helpers);

  llvm::DenseSet<StringRef> callTargets;
  mod.walk([&](tt::CallOp call) {
    callTargets.insert(call.getCalleeAttr().getValue());
  });

  SmallVector<tt::FuncOp> devFuncs, kernels;
  for (auto func : mod.getOps<tt::FuncOp>()) {
    if (callTargets.contains(func.getSymName()))
      devFuncs.push_back(func);
    else
      kernels.push_back(func);
  }

  // MSL forbids declaring threadgroup memory in a non-kernel function, so a
  // shared pool is declared once in the kernel and passed down to every
  // device function as a threadgroup pointer. Size it for the whole module.
  globalPoolBytes = 0;
  for (auto func : mod.getOps<tt::FuncOp>()) {
    poolBytes = 0;
    liveTgBytes = 0;
    func.walk([&](ttg::LocalAllocOp la) {
      auto mt = cast<ttg::MemDescType>(la.getResult().getType());
      liveTgBytes += memdescFlatSize(mt) * byteWidth(mt.getElementType());
    });
    func.walk([&](Operation *op) { scanPool(op); });
    globalPoolBytes = std::max(globalPoolBytes, poolBytes);
  }
  // This pre-scan caches DotFacts before emitFunc has registered any
  // local_alloc in memdescMap, so aInPlace/bInPlace come back empty for
  // operands that ARE resident in an alloc; planDot would then stage a value
  // that was never materialised into registers. The sizing above is unaffected
  // (it keys off the map-independent aNoStage/bNoStage), so recompute the
  // facts once the allocs exist.
  dotFactsCache.clear();
  moduleHasDevFuncs = !devFuncs.empty();

  for (auto func : devFuncs)
    if (failed(emitDeviceFuncProto(func, /*asDecl=*/true)))
      return failure();
  if (!devFuncs.empty())
    os << "\n";
  for (auto func : devFuncs)
    if (failed(emitDeviceFunc(func)))
      return failure();
  for (auto func : kernels)
    if (failed(emitFunc(func)))
      return failure();
  return success();
}

LogicalResult MSLEmitter::emitFunc(tt::FuncOp func) {
  fnFragDecls = 0;
  auto fnTy = func.getFunctionType();
  // Pin the threadgroup size the runtime always dispatches (num_warps*32) so
  // the Metal compiler budgets for exactly this size instead of an
  // occupancy-driven ceiling that could reject a valid launch.
  // Pinning the threadgroup size lets the compiler budget registers for exactly
  // this launch, which is what keeps a big kernel from being rejected as
  // OutOfResources -- but it also lets it spend the whole register file on one
  // threadgroup, so nothing stays resident beside it. That trade only pays when
  // the kernel is already down to one resident threadgroup for another reason:
  // a pool over half the per-core budget cannot fit a second one regardless.
  // Below that, leaving it off is what buys the second threadgroup.
  msl::Attr *maxThreads = nullptr;
  int64_t launchThreads = 0;
  if (auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps")) {
    int64_t tpw = 32;
    if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
      tpw = a.getInt();
    launchThreads = nw.getInt() * tpw;
  }

  // Params + prologue mint real fresh() names IN ORDER (before the body walk)
  // so body register names stay in lockstep with the string ABI numbering.
  unsigned buffer = 0;
  llvm::SmallVector<msl::Param, 8> params;
  SmallVector<BlockArgument> scalarArgs;
  for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
    BlockArgument arg = func.getArgument(i);
    if (auto pt = dyn_cast<tt::PointerType>(argTy)) {
      std::string id = fresh();
      params.push_back(ctx.param(
          ctx.ptr(scalarType(pt.getPointeeType()), msl::AddrSpace::Device), id,
          ctx.bufferAttr(buffer++)));
      bindScalar(arg, id);
    } else if (isa<IntegerType, FloatType>(argTy)) {
      scalarArgs.push_back(arg);
    } else {
      func.emitError("EmitMSL: unsupported kernel argument type");
      return failure();
    }
  }
  std::string argbufId;
  if (!scalarArgs.empty()) {
    argbufId = fresh();
    params.push_back(ctx.param(
        ctx.ptr(ctx.scalar(msl::Scalar::I8), msl::AddrSpace::Constant),
        argbufId, ctx.bufferAttr(buffer++)));
  }

  // Metal binds at most 31 buffers (indices 0-30). Past that the front end
  // rejects the source with "'buffer' attribute parameter is out of bounds",
  // one diagnostic per offending parameter and no hint that an argument count
  // is to blame -- so say it here instead.
  if (buffer > kMaxMSLBuffers) {
    func.emitError() << "EmitMSL: kernel needs " << buffer
                     << " buffer bindings but Metal allows at most "
                     << kMaxMSLBuffers
                     << "; the fused kernel has too many tensor arguments";
    return failure();
  }
  msl::Type *u3 = ctx.vector(msl::Scalar::U32, 3);
  tgposId = fresh();
  tidId = fresh();
  numTgId = fresh();
  params.push_back(ctx.param(u3, tgposId, ctx.tgPosAttr()));
  params.push_back(ctx.param(u3, tidId, ctx.threadPosAttr()));
  params.push_back(ctx.param(u3, numTgId, ctx.tgsPerGridAttr()));

  msl::Block prologue;
  int off = 0;
  for (BlockArgument arg : scalarArgs) {
    Type ty = arg.getType();
    unsigned bits = ty.getIntOrFloatBitWidth();
    int size = bits == 1 ? 1 : (int)(bits / 8);
    off = (off + size - 1) / size * size;
    msl::Type *sc = scalarType(ty);
    std::string id = fresh();
    // *(constant sc*)(argbuf + off)
    msl::Expr *addr = ctx.paren(ctx.binary(msl::BinOp::Add, ctx.var(argbufId),
                                           ctx.lit(std::to_string(off))));
    prologue.push_back(ctx.declStmt(
        sc, id,
        ctx.deref(ctx.cast(msl::Cast::Style::CStyle,
                           ctx.ptr(sc, msl::AddrSpace::Constant), addr))));
    bindScalar(arg, id);
    off += size;
  }
  laneId = fresh();
  warpId = fresh();
  for (msl::Stmt *s : laneWarpProlog())
    prologue.push_back(s);
  coordId = 0;
  layout.beginFunc(&coordId);

  scalarSpinlock = false;
  func.walk([&](tt::AtomicCASOp cas) {
    if (isa<RankedTensorType>(cas.getPtr().getType()))
      return;
    if (tracesToKernelArg(cas.getPtr()))
      scalarSpinlock = true;
  });

  coherentScalarPtrs.clear();
  func.walk([&](LoopLikeOpInterface loop) {
    DenseSet<Value> stored;
    loop->walk([&](tt::StoreOp st) {
      if (isa<RankedTensorType>(st.getPtr().getType()))
        return;
      if (Value base = traceToKernelArg(st.getPtr()))
        stored.insert(base);
    });
    if (stored.empty())
      return;
    loop->walk([&](tt::LoadOp ld) {
      if (isa<RankedTensorType>(ld.getPtr().getType()))
        return;
      Value base = traceToKernelArg(ld.getPtr());
      if (base && stored.contains(base))
        coherentScalarPtrs.insert(base);
    });
  });

  poolBytes = 0;
  liveTgBytes = 0;
  func.walk([&](ttg::LocalAllocOp la) {
    auto mt = cast<ttg::MemDescType>(la.getResult().getType());
    liveTgBytes += memdescFlatSize(mt) * byteWidth(mt.getElementType());
  });

  // Declare every local_alloc's buffer up front. planDot decides whether a dot
  // can read its operand straight out of the allocation, and it runs while the
  // loop's Decl phase is emitted -- before the body walk would have reached an
  // alloc sitting inside (or hoisted just before) that loop. Registering them
  // here keeps the pool sizing, which already assumes those operands need no
  // staging, and the emitter in agreement.
  func.walk([&](ttg::LocalAllocOp la) {
    if (memdescMap.count(la.getResult()))
      return;
    auto mt = cast<ttg::MemDescType>(la.getResult().getType());
    msl::Type *scTy =
        ctx.named("threadgroup " + mslScalarType(mt.getElementType()));
    std::string buf = "__tg_buf_" + std::to_string(tgScratchId++);
    prologue.push_back(ctx.arrayDeclStmt(scTy, buf, memdescFlatSize(mt)));
    memdescMap[la.getResult()] = {buf, nullptr, memdescStrides(mt)};
  });
  // A pipelined operand reaches its dot through memdesc_index (the buffer slot
  // for this trip), and that is the value the dot looks up, not the allocation.
  // Resolve those here too so planDot sees the operand as already resident.
  func.walk([&](ttg::MemDescIndexOp mi) {
    if (memdescMap.count(mi.getResult()) || !memdescMap.count(mi.getSrc()))
      return;
    // The index is an SSA value with no MSL name yet at prescan time, so only
    // a constant slot can be resolved here; a rotating slot is registered when
    // the body walk reaches it.
    APInt slot;
    if (!matchPattern(mi.getIndex(), m_ConstantInt(&slot)))
      return;
    auto resMt = cast<ttg::MemDescType>(mi.getResult().getType());
    MemDescInfo parent = memdescMap[mi.getSrc()];
    int64_t byteOff = slot.getSExtValue() * memdescFlatSize(resMt);
    msl::Expr *base =
        byteOff == 0
            ? nullptr
            : static_cast<msl::Expr *>(ctx.lit(std::to_string(byteOff)));
    memdescMap[mi.getResult()] = {parent.buf, base, parent.bufStrides};
  });

  func.walk([&](Operation *op) { scanPool(op); });
  int64_t kernelPool = moduleHasDevFuncs ? globalPoolBytes : poolBytes;
  if (kernelPool > 0) {
    poolBuf = "__pool";
    prologue.push_back(
        ctx.arrayDeclStmt(ctx.named("threadgroup char"), poolBuf, kernelPool));
  }
  // Left undeclared, the compiler budgets registers for a threadgroup it has to
  // assume the worst about, and caps the pipeline at whatever that budget
  // allows -- as low as 384 threads on a register-hungry kernel. A launch wider
  // than the cap is then rejected as OutOfResources, which is fatal, while
  // declaring the size costs at most one step of occupancy. So the attribute is
  // the default and only a launch too small to ever hit the cap goes without.
  if (launchThreads > kAlwaysAdmittedThreads)
    maxThreads = ctx.maxThreadsAttr(launchThreads);

  // One DMA event token per async-copy site, declared before the walk: the
  // issue and its wait can land in different scopes, and the loop body is
  // walked more than once, so the name has to be stable and already in scope.
  if (dmaStagingEnabled()) {
    // Copies feeding the same loop-carried token slot are one pipeline stage:
    // the peeled prologue copy fills what trip 0 waits on, the in-loop copy
    // fills what later trips wait on. They must share a token, or trip 0 waits
    // on a null one and reads an unfilled tile.
    //
    // The slot -- not the allocation -- is the stage. A depth-2 pipeline aims
    // two prologue copies and one in-loop copy at one allocation but at two
    // different iter-args; keying on the allocation collapses them onto a
    // single token, so the last issue overwrites the others and every trip's
    // wait blocks on the copy it was supposed to be overlapping.
    llvm::DenseMap<std::pair<void *, int>, std::string> stageTok;
    func.walk([&](ttg::AsyncCopyGlobalToLocalOp ac) {
      if (dmaHandleFor.count(ac.getOperation()))
        return;
      if (!dmaCopyEligible(ac))
        return;
      Value dst = ac.getResult();
      while (auto mi = dst.getDefiningOp<ttg::MemDescIndexOp>())
        dst = mi.getSrc();
      std::string &tok = stageTok[{dst.getAsOpaquePointer(), dmaStageSlot(ac)}];
      if (tok.empty()) {
        tok = fresh();
        prologue.push_back(ctx.declStmt(ctx.named("ulong"), tok, ctx.lit("0")));
      }
      dmaHandleFor[ac.getOperation()] = tok;
    });
  }

  Region &region = func.getBody();
  msl::Block body = region.hasOneBlock() ? walkBlock(region.front(), indent)
                                         : emitBlockCFG(region);
  if (emitFailed)
    return failure();
  dmaHandleDecls.clear();
  // Tokens are keyed per copy site; a later kernel must mint its own.
  dmaHandleFor.clear();
  for (msl::Stmt *s : layout.takeDecls())
    prologue.push_back(s);
  for (msl::Stmt *s : body)
    prologue.push_back(s);
  msl::KernelFn *fn = ctx.kernelFn(maxThreads, mslKernelName(func.getName()),
                                   params, std::move(prologue));
  eliminateDeadLocals(fn);
  msl::MSLPrinter printer(os);
  printer.print(fn);
  return success();
}

LogicalResult MSLEmitter::declRetStruct(tt::FuncOp func) {
  auto results = func.getFunctionType().getResults();
  if (!isTensorResult(results) && results.size() <= 1)
    return success();
  for (Type ty : results)
    if (!isTensorResult(results) && !isa<IntegerType, FloatType>(ty)) {
      func.emitError("EmitMSL: unsupported device function result type");
      return failure();
    }
  devRetStruct[func] = mslDeviceFuncName(func.getName()) + "_ret";
  msl::MSLPrinter printer(os);
  printer.print(retStructDecl(func));
  os << "\n";
  return success();
}

LogicalResult MSLEmitter::emitDeviceFuncProto(tt::FuncOp func, bool asDecl) {
  if (failed(declRetStruct(func)))
    return failure();
  msl::MSLPrinter printer(os);
  printer.printProto(deviceProto(func));
  return success();
}

LogicalResult MSLEmitter::emitDeviceFunc(tt::FuncOp func) {
  auto fnTy = func.getFunctionType();
  llvm::SmallVector<msl::Param, 8> params;
  for (auto [i, argTy] : llvm::enumerate(fnTy.getInputs())) {
    std::string id = fresh();
    if (auto pt = dyn_cast<tt::PointerType>(argTy))
      params.push_back(ctx.param(
          ctx.ptr(scalarType(pt.getPointeeType()), msl::AddrSpace::Device),
          id));
    else if (isa<IntegerType, FloatType>(argTy))
      params.push_back(ctx.param(scalarType(argTy), id));
    else {
      func.emitError("EmitMSL: unsupported device function argument type");
      return failure();
    }
    bindScalar(func.getArgument(i), id);
  }
  msl::Type *u3 = ctx.vector(msl::Scalar::U32, 3);
  tgposId = fresh();
  tidId = fresh();
  numTgId = fresh();
  params.push_back(ctx.param(u3, tgposId));
  params.push_back(ctx.param(u3, tidId));
  params.push_back(ctx.param(u3, numTgId));
  devPoolPtr.clear();
  if (globalPoolBytes > 0) {
    devPoolPtr = fresh();
    params.push_back(ctx.param(
        ctx.ptr(ctx.scalar(msl::Scalar::I8), msl::AddrSpace::Threadgroup),
        devPoolPtr));
  }

  laneId = fresh();
  warpId = fresh();
  msl::Block prologue = laneWarpProlog();
  coordId = 0;
  layout.beginFunc(&coordId);

  poolBuf = devPoolPtr;
  curDevFunc = func;
  Region &region = func.getBody();
  msl::Block body = region.hasOneBlock() ? walkBlock(region.front(), indent)
                                         : emitBlockCFG(region);
  if (emitFailed)
    return failure();
  curDevFunc = nullptr;
  for (msl::Stmt *s : layout.takeDecls())
    prologue.push_back(s);
  for (msl::Stmt *s : body)
    prologue.push_back(s);
  msl::DeviceFn *fn =
      ctx.deviceFn(deviceRetType(func), mslDeviceFuncName(func.getName()),
                   params, std::move(prologue));
  msl::MSLPrinter printer(os);
  printer.print(fn);
  return success();
}

} // namespace mlir::triton::applegpu
