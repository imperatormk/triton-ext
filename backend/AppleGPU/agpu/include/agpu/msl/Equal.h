// Equal.h - structural equality of two expressions.
#ifndef AGPU_MSL_EQUAL_H
#define AGPU_MSL_EQUAL_H

#include "Ast.h"

namespace agpu::msl {

inline bool exprsEqual(const Expr *a, const Expr *b) {
  if (a == b)
    return true;
  if (!a || !b || a->kind != b->kind)
    return false;

  switch (a->kind) {
  case ExprKind::VarRef:
    return static_cast<const VarRef *>(a)->name ==
           static_cast<const VarRef *>(b)->name;
  case ExprKind::Literal: {
    const auto *la = static_cast<const Literal *>(a);
    const auto *lb = static_cast<const Literal *>(b);
    return la->form == lb->form && la->intValue == lb->intValue &&
           la->floatValue == lb->floatValue;
  }
  case ExprKind::Binary: {
    const auto *ba = static_cast<const Binary *>(a);
    const auto *bb = static_cast<const Binary *>(b);
    return ba->op == bb->op && exprsEqual(ba->lhs, bb->lhs) &&
           exprsEqual(ba->rhs, bb->rhs);
  }
  case ExprKind::Cast: {
    const auto *ca = static_cast<const Cast *>(a);
    const auto *cb = static_cast<const Cast *>(b);
    return ca->style == cb->style && exprsEqual(ca->operand, cb->operand);
  }
  case ExprKind::Call: {
    const auto *ca = static_cast<const Call *>(a);
    const auto *cb = static_cast<const Call *>(b);
    if (ca->callee != cb->callee || ca->args.size() != cb->args.size())
      return false;
    for (std::size_t i = 0; i < ca->args.size(); ++i)
      if (!exprsEqual(ca->args[i], cb->args[i]))
        return false;
    return true;
  }
  case ExprKind::Subscript: {
    const auto *sa = static_cast<const Subscript *>(a);
    const auto *sb = static_cast<const Subscript *>(b);
    return exprsEqual(sa->base, sb->base) && exprsEqual(sa->index, sb->index);
  }
  case ExprKind::Member: {
    const auto *ma = static_cast<const Member *>(a);
    const auto *mb = static_cast<const Member *>(b);
    return ma->field == mb->field && exprsEqual(ma->base, mb->base);
  }
  case ExprKind::Deref:
    return exprsEqual(static_cast<const Deref *>(a)->operand,
                      static_cast<const Deref *>(b)->operand);
  case ExprKind::AddrOf:
    return exprsEqual(static_cast<const AddrOf *>(a)->operand,
                      static_cast<const AddrOf *>(b)->operand);
  }
  return false;
}

} // namespace agpu::msl

#endif // AGPU_MSL_EQUAL_H
