// MSLFuncBudget.cpp - measurement, policy and post-emission shrink mitigations.

#include "MSLFuncBudget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace mlir::triton::applegpu {

namespace {

bool exprsEqual(const msl::Expr *a, const msl::Expr *b) {
  if (a == b)
    return true;
  if (!a || !b || a->kind != b->kind)
    return false;
  switch (a->kind) {
  case msl::Expr::Kind::VarRef:
    return llvm::cast<msl::VarRef>(a)->name ==
           llvm::cast<msl::VarRef>(b)->name;
  case msl::Expr::Kind::Literal:
    return llvm::cast<msl::Literal>(a)->text ==
           llvm::cast<msl::Literal>(b)->text;
  case msl::Expr::Kind::Raw:
    return llvm::cast<msl::Raw>(a)->text == llvm::cast<msl::Raw>(b)->text;
  case msl::Expr::Kind::Binary: {
    auto *x = llvm::cast<msl::Binary>(a);
    auto *y = llvm::cast<msl::Binary>(b);
    return x->op == y->op && exprsEqual(x->lhs, y->lhs) &&
           exprsEqual(x->rhs, y->rhs);
  }
  case msl::Expr::Kind::Unary: {
    auto *x = llvm::cast<msl::Unary>(a);
    auto *y = llvm::cast<msl::Unary>(b);
    return x->op == y->op && exprsEqual(x->x, y->x);
  }
  case msl::Expr::Kind::Paren:
    return exprsEqual(llvm::cast<msl::Paren>(a)->x,
                      llvm::cast<msl::Paren>(b)->x);
  case msl::Expr::Kind::Subscript: {
    auto *x = llvm::cast<msl::Subscript>(a);
    auto *y = llvm::cast<msl::Subscript>(b);
    return exprsEqual(x->base, y->base) && exprsEqual(x->idx, y->idx);
  }
  case msl::Expr::Kind::Member: {
    auto *x = llvm::cast<msl::Member>(a);
    auto *y = llvm::cast<msl::Member>(b);
    return x->field == y->field && exprsEqual(x->base, y->base);
  }
  default:
    return false;
  }
}

void collectReadNames(const msl::Expr *e,
                      llvm::DenseSet<llvm::StringRef> &names) {
  if (!e)
    return;
  switch (e->kind) {
  case msl::Expr::Kind::VarRef:
    names.insert(llvm::cast<msl::VarRef>(e)->name);
    return;
  case msl::Expr::Kind::Binary: {
    auto *b = llvm::cast<msl::Binary>(e);
    collectReadNames(b->lhs, names);
    collectReadNames(b->rhs, names);
    return;
  }
  case msl::Expr::Kind::Unary:
    collectReadNames(llvm::cast<msl::Unary>(e)->x, names);
    return;
  case msl::Expr::Kind::Paren:
    collectReadNames(llvm::cast<msl::Paren>(e)->x, names);
    return;
  case msl::Expr::Kind::Subscript: {
    auto *s = llvm::cast<msl::Subscript>(e);
    collectReadNames(s->base, names);
    collectReadNames(s->idx, names);
    return;
  }
  case msl::Expr::Kind::Member:
    collectReadNames(llvm::cast<msl::Member>(e)->base, names);
    return;
  case msl::Expr::Kind::Literal:
    return;
  default:
    // Anything not modelled above (calls, casts, ternaries, raw text) may read
    // names this walk cannot see, so no run containing it is ever fused.
    names.insert(llvm::StringRef());
    return;
  }
}

// A store `p[i] = v` cannot change a scalar condition, but a plain `v = rhs`
// or a decl of a name the condition reads can.
bool bodyMayClobber(const msl::Stmt *s,
                    const llvm::DenseSet<llvm::StringRef> &condNames) {
  if (auto *a = llvm::dyn_cast<msl::AssignStmt>(s)) {
    if (auto *lv = llvm::dyn_cast<msl::VarRef>(a->lhs))
      return condNames.count(lv->name);
    if (auto *sub = llvm::dyn_cast<msl::Subscript>(a->lhs)) {
      llvm::DenseSet<llvm::StringRef> touched;
      collectReadNames(sub->base, touched);
      for (llvm::StringRef n : touched)
        if (condNames.count(n))
          return true;
      return false;
    }
    return true;
  }
  if (auto *d = llvm::dyn_cast<msl::DeclStmt>(s))
    return condNames.count(d->name);
  return true;
}

// A coordinate temp's reachable value set: { base | s : s subset of freeMask }.
struct CoordSet {
  int32_t base = 0;
  int32_t freeMask = 0;
  bool valid = false;
};

bool coordSetOf(const msl::Expr *e, CoordSet &out);

bool coordTerm(const msl::Expr *e, CoordSet &out) {
  if (auto *p = llvm::dyn_cast<msl::Paren>(e))
    return coordTerm(p->x, out);
  if (auto *l = llvm::dyn_cast<msl::Literal>(e)) {
    int32_t v;
    if (l->text.getAsInteger(10, v))
      return false;
    if (v & out.freeMask)
      return false;
    out.base ^= v;
    return true;
  }
  auto *b = llvm::dyn_cast<msl::Binary>(e);
  if (!b)
    return false;
  // ((id >> k) & 1) * w
  if (b->op == msl::BinOp::Mul) {
    auto *w = llvm::dyn_cast<msl::Literal>(b->rhs);
    if (!w)
      return false;
    int32_t wv;
    if (w->text.getAsInteger(10, wv) || wv <= 0 || (wv & (wv - 1)))
      return false;
    if ((wv & out.freeMask) || (wv & out.base))
      return false;
    out.freeMask |= wv;
    return true;
  }
  // (id & mask): a contiguous bitfield, every bit free.
  if (b->op == msl::BinOp::And) {
    auto *m = llvm::dyn_cast<msl::Literal>(b->rhs);
    if (!m)
      return false;
    int32_t mv;
    if (m->text.getAsInteger(10, mv) || mv <= 0)
      return false;
    if ((mv & out.freeMask) || (mv & out.base))
      return false;
    out.freeMask |= mv;
    return true;
  }
  return false;
}

bool coordSetOf(const msl::Expr *e, CoordSet &out) {
  if (auto *p = llvm::dyn_cast<msl::Paren>(e))
    return coordSetOf(p->x, out);
  if (auto *b = llvm::dyn_cast<msl::Binary>(e))
    if (b->op == msl::BinOp::Xor)
      return coordSetOf(b->lhs, out) && coordSetOf(b->rhs, out);
  return coordTerm(e, out);
}

// Resolves a VarRef through the function's coordinate-temp declarations.
struct CoordEnv {
  llvm::DenseMap<llvm::StringRef, CoordSet> sets;

  void addDecl(const msl::DeclStmt *d) {
    if (!d->init)
      return;
    CoordSet cs;
    if (coordSetOf(d->init, cs)) {
      cs.valid = true;
      sets[d->name] = cs;
    }
  }

  CoordSet lookup(const msl::Expr *e) const {
    if (auto *p = llvm::dyn_cast<msl::Paren>(e))
      return lookup(p->x);
    if (auto *v = llvm::dyn_cast<msl::VarRef>(e)) {
      auto it = sets.find(v->name);
      if (it != sets.end())
        return it->second;
    }
    if (auto *l = llvm::dyn_cast<msl::Literal>(e)) {
      CoordSet cs;
      if (!l->text.getAsInteger(10, cs.base)) {
        cs.valid = true;
        return cs;
      }
    }
    if (auto *b = llvm::dyn_cast<msl::Binary>(e)) {
      if (b->op == msl::BinOp::Mul)
        return lookup(b->lhs);
      // `c - k` / `c + k` shifts the whole set by a constant.
      if (b->op == msl::BinOp::Sub || b->op == msl::BinOp::Add) {
        auto *k = llvm::dyn_cast<msl::Literal>(b->rhs);
        CoordSet cs = lookup(b->lhs);
        int32_t kv;
        if (k && cs.valid && !k->text.getAsInteger(10, kv)) {
          int32_t delta = b->op == msl::BinOp::Sub ? -kv : kv;
          // Only exact when the shift does not disturb the free bits.
          if ((cs.base + delta) >= 0 &&
              ((cs.base + delta) & cs.freeMask) == 0) {
            cs.base += delta;
            return cs;
          }
        }
      }
    }
    return CoordSet{};
  }
};

bool provablyDisjoint(const CoordSet &a, const CoordSet &b) {
  if (!a.valid || !b.valid)
    return false;
  int32_t free = a.freeMask | b.freeMask;
  return (a.base & ~free) != (b.base & ~free);
}

// The row selector of `buf[(row * stride + col)] = v`, or null.
const msl::Expr *storeRowSelector(const msl::Stmt *s, llvm::StringRef &buf) {
  auto *a = llvm::dyn_cast<msl::AssignStmt>(s);
  if (!a || a->compound != msl::AssignStmt::Compound::None)
    return nullptr;
  auto *sub = llvm::dyn_cast<msl::Subscript>(a->lhs);
  if (!sub)
    return nullptr;
  auto *base = llvm::dyn_cast<msl::VarRef>(sub->base);
  if (!base)
    return nullptr;
  buf = base->name;
  const msl::Expr *idx = sub->idx;
  while (auto *p = llvm::dyn_cast<msl::Paren>(idx))
    idx = p->x;
  auto *add = llvm::dyn_cast<msl::Binary>(idx);
  if (!add || add->op != msl::BinOp::Add)
    return nullptr;
  const msl::Expr *lhs = add->lhs;
  while (auto *p = llvm::dyn_cast<msl::Paren>(lhs))
    lhs = p->x;
  auto *mul = llvm::dyn_cast<msl::Binary>(lhs);
  if (!mul || mul->op != msl::BinOp::Mul)
    return nullptr;
  return mul->lhs;
}

// Sinks `if (C) buf[row*s + c] = v` past stores that provably cannot alias it
// and cannot change C, making same-predicate stores adjacent and fusable.
unsigned sinkGuardedStores(msl::Block &b, const CoordEnv &env) {
  unsigned moved = 0;
  for (size_t i = 0; i + 1 < b.size(); ++i) {
    auto *ci = llvm::dyn_cast<msl::CompactIfStmt>(b[i]);
    if (!ci)
      continue;
    llvm::StringRef bufI;
    const msl::Expr *rowI = storeRowSelector(ci->then, bufI);
    if (!rowI)
      continue;
    CoordSet setI = env.lookup(rowI);
    if (!setI.valid)
      continue;
    llvm::DenseSet<llvm::StringRef> condNames;
    collectReadNames(ci->cond, condNames);
    if (condNames.count(llvm::StringRef()))
      continue;

    size_t target = 0;
    for (size_t j = i + 1; j < b.size(); ++j) {
      auto *cj = llvm::dyn_cast<msl::CompactIfStmt>(b[j]);
      if (cj && cj->parenCond == ci->parenCond &&
          exprsEqual(cj->cond, ci->cond)) {
        target = j;
        break;
      }
      llvm::StringRef bufJ;
      const msl::Expr *rowJ = storeRowSelector(b[j], bufJ);
      if (!rowJ || bodyMayClobber(b[j], condNames))
        break;
      if (bufJ == bufI && !provablyDisjoint(setI, env.lookup(rowJ)))
        break;
      auto *aj = llvm::dyn_cast<msl::AssignStmt>(b[j]);
      llvm::DenseSet<llvm::StringRef> rhsNames;
      collectReadNames(llvm::cast<msl::AssignStmt>(ci->then)->rhs, rhsNames);
      if (auto *lv = llvm::dyn_cast<msl::VarRef>(aj->lhs))
        if (rhsNames.count(lv->name))
          break;
    }
    if (target <= i + 1)
      continue;
    msl::Stmt *s = b[i];
    b.erase(b.begin() + i);
    b.insert(b.begin() + (target - 1), s);
    ++moved;
    --i;
  }
  return moved;
}

void collectCoordDecls(const msl::Block &b, CoordEnv &env) {
  for (msl::Stmt *s : b)
    if (auto *d = llvm::dyn_cast<msl::DeclStmt>(s))
      env.addDecl(d);
}

unsigned fuseCompactIfRuns(msl::Block &b, msl::MSLContext &ctx, CoordEnv &env);

unsigned fuseCompactIfRunsIn(msl::Stmt *s, msl::MSLContext &ctx,
                             CoordEnv &env) {
  switch (s->kind) {
  case msl::Stmt::Kind::KernelFn:
    return fuseCompactIfRuns(llvm::cast<msl::KernelFn>(s)->body, ctx, env);
  case msl::Stmt::Kind::DeviceFn:
    return fuseCompactIfRuns(llvm::cast<msl::DeviceFn>(s)->body, ctx, env);
  case msl::Stmt::Kind::ForScope:
    return fuseCompactIfRuns(llvm::cast<msl::ForScope>(s)->body, ctx, env);
  case msl::Stmt::Kind::TripCountForScope:
    return fuseCompactIfRuns(llvm::cast<msl::TripCountForScope>(s)->body, ctx, env);
  case msl::Stmt::Kind::WhileScope:
    return fuseCompactIfRuns(llvm::cast<msl::WhileScope>(s)->body, ctx, env);
  case msl::Stmt::Kind::PlainScope:
    return fuseCompactIfRuns(llvm::cast<msl::PlainScope>(s)->body, ctx, env);
  case msl::Stmt::Kind::IfScope: {
    auto *i = llvm::cast<msl::IfScope>(s);
    return fuseCompactIfRuns(i->thenB, ctx, env) +
           fuseCompactIfRuns(i->elseB, ctx, env);
  }
  case msl::Stmt::Kind::StateMachineScope: {
    unsigned n = 0;
    for (auto &c : llvm::cast<msl::StateMachineScope>(s)->cases)
      n += fuseCompactIfRuns(c.body, ctx, env);
    return n;
  }
  default:
    return 0;
  }
}

// Fuses a maximal run of consecutive `if (C) S` sharing one condition into a
// single `if (C) { S... }`. Purely syntactic: the guarded statements and their
// order are preserved exactly, so this is a statement-count reduction only.
unsigned fuseCompactIfRuns(msl::Block &b, msl::MSLContext &ctx, CoordEnv &env) {
  collectCoordDecls(b, env);
  for (int i = 0; i < 16 && sinkGuardedStores(b, env); ++i)
    ;
  unsigned fused = 0;
  msl::Block keep;
  keep.reserve(b.size());
  for (size_t i = 0; i < b.size();) {
    auto *c = llvm::dyn_cast<msl::CompactIfStmt>(b[i]);
    if (!c) {
      fused += fuseCompactIfRunsIn(b[i], ctx, env);
      keep.push_back(b[i]);
      ++i;
      continue;
    }
    llvm::DenseSet<llvm::StringRef> condNames;
    collectReadNames(c->cond, condNames);
    bool opaqueCond = condNames.count(llvm::StringRef());
    size_t j = i;
    msl::Block run;
    while (j < b.size()) {
      auto *cj = llvm::dyn_cast<msl::CompactIfStmt>(b[j]);
      if (!cj || cj->parenCond != c->parenCond ||
          !exprsEqual(cj->cond, c->cond))
        break;
      if (opaqueCond || bodyMayClobber(cj->then, condNames))
        break;
      run.push_back(cj->then);
      ++j;
    }
    // A short run costs two brace lines to save a handful of `if (c) ` prefixes,
    // so fusing it grows the output.
    if (run.size() < 4) {
      keep.push_back(b[i]);
      ++i;
      continue;
    }
    fused += run.size() - 1;
    // IfScope always prints its own parens; a bare guard already carries them.
    msl::Expr *cond = c->cond;
    if (!c->parenCond)
      if (auto *p = llvm::dyn_cast<msl::Paren>(cond))
        cond = p->x;
    keep.push_back(ctx.ifScope(cond, std::move(run)));
    i = j;
  }
  b = std::move(keep);
  return fused;
}
} // namespace

//===----------------------------------------------------------------------===//
// Measurement
//===----------------------------------------------------------------------===//

namespace {
void measureBlock(const msl::Block &b, FuncSize &s);

void measureStmt(msl::Stmt *st, FuncSize &s) {
  using K = msl::Stmt::Kind;
  ++s.stmts;
  switch (st->kind) {
  case K::Decl: {
    auto *d = llvm::cast<msl::DeclStmt>(st);
    ++s.decls;
    if (llvm::isa<msl::MatrixType>(d->ty))
      ++s.fragDecls;
    return;
  }
  case K::ArrayDecl:
    ++s.decls;
    return;
  case K::Barrier:
    ++s.barriers;
    return;
  case K::CompactIf:
    ++s.branches;
    return;
  case K::ForScope:
    ++s.loops;
    measureBlock(llvm::cast<msl::ForScope>(st)->body, s);
    return;
  case K::TripCountForScope:
    ++s.loops;
    measureBlock(llvm::cast<msl::TripCountForScope>(st)->body, s);
    return;
  case K::WhileScope:
    ++s.loops;
    measureBlock(llvm::cast<msl::WhileScope>(st)->body, s);
    return;
  case K::PlainScope:
    measureBlock(llvm::cast<msl::PlainScope>(st)->body, s);
    return;
  case K::IfScope: {
    auto *i = llvm::cast<msl::IfScope>(st);
    ++s.branches;
    measureBlock(i->thenB, s);
    measureBlock(i->elseB, s);
    return;
  }
  case K::StateMachineScope: {
    auto *sm = llvm::cast<msl::StateMachineScope>(st);
    ++s.branches;
    for (auto &c : sm->cases)
      measureBlock(c.body, s);
    return;
  }
  case K::KernelFn:
    measureBlock(llvm::cast<msl::KernelFn>(st)->body, s);
    return;
  case K::DeviceFn:
    measureBlock(llvm::cast<msl::DeviceFn>(st)->body, s);
    return;
  case K::Expr: {
    auto *c = llvm::dyn_cast<msl::Call>(llvm::cast<msl::ExprStmt>(st)->e);
    if (c && c->callee == "simdgroup_multiply_accumulate")
      ++s.mma;
    return;
  }
  default:
    return;
  }
}

void measureBlock(const msl::Block &b, FuncSize &s) {
  for (msl::Stmt *st : b)
    measureStmt(st, s);
}
} // namespace

FuncSize measureFunc(const msl::Block &body) {
  FuncSize s;
  measureBlock(body, s);
  return s;
}

//===----------------------------------------------------------------------===//
// Policy
//===----------------------------------------------------------------------===//

// Measured over a full crashing run (321 kernels): every kernel above 30k
// emitted lines has >= 16077 declarations and every one below has <= 13657, so
// declarations separate the two populations where no other count does. The gate
// sits under that boundary with margin, since a kernel just below it still
// compiles today.
static constexpr int64_t kDeclBudget = 12000;

// Rolling replaces kT unrolled k-steps with one loop body, so it only pays back
// its loop when there are enough fragments for the division to matter.
static constexpr int64_t kRollFragFloor = 1024;

ShrinkPlan planShrink(const FuncSize &s) {
  ShrinkPlan p;
  // Fusing costs nothing semantically and only ever lowers the statement count
  // the optimiser must carry, so it is not rationed by the budget.
  p.fuseGuards = s.branches > 0;
  p.rollKSteps =
      s.optimiserLoad() > kDeclBudget && s.fragDecls >= kRollFragFloor;
  return p;
}

void shrink(msl::KernelFn *fn, msl::MSLContext &ctx, const ShrinkPlan &plan) {
  if (!plan.fuseGuards)
    return;
  CoordEnv env;
  fuseCompactIfRunsIn(fn, ctx, env);
}

void debugBudget(llvm::StringRef fn, llvm::StringRef stage, const FuncSize &s,
                 const ShrinkPlan &plan) {
  if (!getenv("MSL_FUNC_BUDGET_DEBUG"))
    return;
  llvm::errs() << "[budget] " << fn << " " << stage << " decls=" << s.decls
               << " stmts=" << s.stmts << " branches=" << s.branches
               << " frags=" << s.fragDecls << " mma=" << s.mma
               << " barriers=" << s.barriers << " loops=" << s.loops
               << " budget=" << kDeclBudget << " roll=" << plan.rollKSteps
               << " fuse=" << plan.fuseGuards << "\n";
}

} // namespace mlir::triton::applegpu
