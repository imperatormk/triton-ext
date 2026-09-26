// EmitSchedule.h - the order of a block's statements.
//
// Elementwise ops are emitted one op at a time over every register, so a
// thread holds each op's results for all of its registers at once. A wide tile
// in a large threadgroup then spills. Placing each pure declaration right
// before its first reader runs one register's chain to its end before the next
// begins.
#ifndef AGPU_EMIT_SCHEDULE_H
#define AGPU_EMIT_SCHEDULE_H

#include "agpu/msl/Analysis.h"
#include "agpu/msl/AstWalk.h"

#include <vector>

namespace agpu {

namespace detail {

// A value whose initialiser reads no memory, calls nothing another lane sees
// and writes no argument: `sincos`, `modf` and `frexp` return a second value
// through a reference.
inline bool sinkable(const msl::Stmt *s) {
  if (s->kind != msl::StmtKind::Decl)
    return false;
  const auto *d = static_cast<const msl::Decl *>(s);
  const msl::Type::Form form = d->type.form();
  if (form != msl::Type::Form::Scalar && form != msl::Type::Form::Vector)
    return false;
  msl::Expr *init = d->init;
  if (!init)
    return false;
  bool pure = true;
  msl::visitExprs(init, [&](msl::Expr *e) {
    if (e->kind == msl::ExprKind::Subscript || e->kind == msl::ExprKind::Deref)
      pure = false;
    if (e->kind == msl::ExprKind::Call) {
      const msl::Str &f = static_cast<msl::Call *>(e)->callee;
      for (const char *impure :
           {"simd_", "quad_", "atomic", "barrier", "sincos", "modf", "frexp"})
        if (f.find(impure) != msl::Str::npos)
          pure = false;
    }
  });
  return pure;
}

// The names a statement assigns, at any depth.
inline msl::PtrSet<msl::Str> namesWritten(msl::Stmt *s) {
  msl::PtrSet<msl::Str> out;
  msl::visitStmtsOnly(s, [&](msl::Stmt *n) {
    bool escapes = false;
    const msl::Str w = msl::writtenName(n, escapes);
    if (!w.empty())
      out.insert(w);
  });
  return out;
}

} // namespace detail

// Walking up from the end, each declaration moves to right before the first
// statement that now reads or assigns it, unless a statement between writes
// one of its operands. A chain's heavy links, sunk first, end up below the
// cheap ones still waiting above them, so those overlap the long latencies.
inline void sinkToFirstReader(msl::Block &body) {
  for (msl::Stmt *s : body)
    msl::forEachChildBlock(s,
                           [](msl::Block &child) { sinkToFirstReader(child); });

  struct Entry {
    msl::Stmt *stmt;
    msl::PtrSet<msl::Str> reads, writes;
  };
  std::vector<Entry> es;
  es.reserve(body.size());
  for (msl::Stmt *s : body)
    es.push_back(
        {s, msl::collectReads(msl::Block{s}), detail::namesWritten(s)});

  for (std::size_t i = es.size(); i-- > 0;) {
    if (!detail::sinkable(es[i].stmt))
      continue;
    const msl::Str &name = static_cast<msl::Decl *>(es[i].stmt)->name;
    std::size_t j = i + 1;
    bool blocked = false;
    for (;
         j < es.size() && !es[j].reads.count(name) && !es[j].writes.count(name);
         ++j)
      for (const msl::Str &op : es[i].reads)
        blocked = blocked || es[j].writes.count(op);
    if (j == es.size() || j == i + 1 || blocked)
      continue;
    Entry e = std::move(es[i]);
    es.erase(es.begin() + (std::ptrdiff_t)i);
    es.insert(es.begin() + (std::ptrdiff_t)(j - 1), std::move(e));
  }

  body.clear();
  for (Entry &e : es)
    body.push_back(e.stmt);
}

} // namespace agpu

#endif // AGPU_EMIT_SCHEDULE_H
