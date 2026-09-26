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

#include <algorithm>
#include <functional>
#include <unordered_map>
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

// The other statements keep their order. Each is preceded by the not yet
// placed declarations it reads or assigns, each of those by its own, so a
// declaration lands right before its first reader. A statement that writes a
// name is also preceded by every earlier declaration reading that name, which
// must see the value from before the write.
inline void sinkToFirstReader(msl::Block &body) {
  for (msl::Stmt *s : body)
    msl::forEachChildBlock(s,
                           [](msl::Block &child) { sinkToFirstReader(child); });

  const std::size_t n = body.size();
  std::vector<msl::PtrSet<msl::Str>> reads(n), writes(n);
  std::vector<bool> moves(n);
  std::unordered_map<msl::Str, std::size_t> declOf;
  std::unordered_map<msl::Str, std::vector<std::size_t>> readersOf;
  for (std::size_t i = 0; i < n; ++i) {
    reads[i] = msl::collectReads(msl::Block{body[i]});
    writes[i] = detail::namesWritten(body[i]);
    moves[i] = detail::sinkable(body[i]);
    if (!moves[i])
      continue;
    declOf[static_cast<msl::Decl *>(body[i])->name] = i;
    for (const msl::Str &op : reads[i])
      readersOf[op].push_back(i);
  }

  msl::Block out;
  out.reserve(n);
  std::vector<bool> placed(n, false);
  const std::function<void(std::size_t)> place = [&](std::size_t x) {
    placed[x] = true;
    std::vector<std::size_t> first;
    const auto need = [&](std::size_t d) {
      if (d < x && moves[d] && !placed[d])
        first.push_back(d);
    };
    for (const msl::PtrSet<msl::Str> *names : {&reads[x], &writes[x]})
      for (const msl::Str &v : *names)
        if (const auto it = declOf.find(v); it != declOf.end())
          need(it->second);
    for (const msl::Str &v : writes[x])
      if (const auto it = readersOf.find(v); it != readersOf.end())
        for (std::size_t d : it->second)
          need(d);
    std::sort(first.begin(), first.end());
    for (std::size_t d : first)
      if (!placed[d])
        place(d);
    out.push_back(body[x]);
  };
  for (std::size_t i = 0; i < n; ++i)
    if (!moves[i])
      place(i);
  for (std::size_t i = 0; i < n; ++i)
    if (!placed[i])
      place(i);
  body = std::move(out);
}

} // namespace agpu

#endif // AGPU_EMIT_SCHEDULE_H
