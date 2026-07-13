// MSLPrinter.cpp - see MSLPrinter.h.
#include "MSLPrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::applegpu::msl {

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

static llvm::StringRef scalarName(Scalar s) {
  switch (s) {
  case Scalar::F32: return "float";
  case Scalar::F16: return "half";
  case Scalar::BF16: return "bfloat";
  case Scalar::I1: return "bool";
  case Scalar::I8: return "char";
  case Scalar::I16: return "short";
  case Scalar::I32: return "int";
  case Scalar::I64: return "long";
  case Scalar::U8: return "uchar";
  case Scalar::U16: return "ushort";
  case Scalar::U32: return "uint";
  case Scalar::U64: return "ulong";
  case Scalar::Void: return "void";
  case Scalar::SizeT: return "size_t";
  }
  llvm_unreachable("bad Scalar");
}

static llvm::StringRef atomicName(Scalar s) {
  switch (s) {
  case Scalar::F32: return "atomic_float";
  case Scalar::I32: return "atomic_int";
  case Scalar::U32: return "atomic_uint";
  case Scalar::I64: return "atomic_long";
  case Scalar::U64: return "atomic_ulong";
  default: return "atomic_int";
  }
}

static llvm::StringRef matrixName(MatrixType::Elem e) {
  switch (e) {
  case MatrixType::Elem::Half: return "simdgroup_half8x8";
  case MatrixType::Elem::Bfloat: return "simdgroup_bfloat8x8";
  case MatrixType::Elem::Float: return "simdgroup_float8x8";
  }
  llvm_unreachable("bad Matrix elem");
}

static llvm::StringRef addrSpaceName(AddrSpace as) {
  switch (as) {
  case AddrSpace::None: return "";
  case AddrSpace::Device: return "device ";
  case AddrSpace::Threadgroup: return "threadgroup ";
  case AddrSpace::Constant: return "constant ";
  case AddrSpace::Thread: return "thread ";
  }
  llvm_unreachable("bad AddrSpace");
}

void MSLPrinter::printType(const Type *t) {
  switch (t->kind) {
  case Type::Kind::Scalar:
    os << scalarName(llvm::cast<ScalarType>(t)->s);
    return;
  case Type::Kind::Vector: {
    auto *v = llvm::cast<VectorType>(t);
    os << scalarName(v->elem) << v->n;
    return;
  }
  case Type::Kind::Matrix:
    os << matrixName(llvm::cast<MatrixType>(t)->elem);
    return;
  case Type::Kind::Atomic:
    os << atomicName(llvm::cast<AtomicType>(t)->elem);
    return;
  case Type::Kind::Pointer: {
    auto *p = llvm::cast<PointerType>(t);
    if (p->coherent && p->as == AddrSpace::Device)
      os << "coherent(device) ";
    else
      os << addrSpaceName(p->as);
    if (p->volatile_)
      os << "volatile ";
    printType(p->pointee);
    os << "*";
    return;
  }
  case Type::Kind::Named:
    os << llvm::cast<NamedType>(t)->name;
    return;
  }
  llvm_unreachable("bad Type kind");
}

//===----------------------------------------------------------------------===//
// Expressions (printer is dumb: grouping comes from explicit Paren nodes)
//===----------------------------------------------------------------------===//

static llvm::StringRef binOpText(BinOp op) {
  switch (op) {
  case BinOp::Add: return " + ";
  case BinOp::Sub: return " - ";
  case BinOp::Mul: return " * ";
  case BinOp::Div: return " / ";
  case BinOp::Rem: return " % ";
  case BinOp::And: return " & ";
  case BinOp::Or: return " | ";
  case BinOp::Xor: return " ^ ";
  case BinOp::Shl: return " << ";
  case BinOp::Shr: return " >> ";
  case BinOp::Lt: return " < ";
  case BinOp::Le: return " <= ";
  case BinOp::Gt: return " > ";
  case BinOp::Ge: return " >= ";
  case BinOp::Eq: return " == ";
  case BinOp::Ne: return " != ";
  case BinOp::LAnd: return " && ";
  case BinOp::LOr: return " || ";
  }
  llvm_unreachable("bad BinOp");
}

void MSLPrinter::printExpr(const Expr *e) {
  switch (e->kind) {
  case Expr::Kind::VarRef:
    os << llvm::cast<VarRef>(e)->name;
    return;
  case Expr::Kind::Literal:
    os << llvm::cast<Literal>(e)->text;
    return;
  case Expr::Kind::Binary: {
    auto *b = llvm::cast<Binary>(e);
    printExpr(b->lhs);
    os << binOpText(b->op);
    printExpr(b->rhs);
    return;
  }
  case Expr::Kind::Unary: {
    auto *u = llvm::cast<Unary>(e);
    switch (u->op) {
    case UnOp::Neg: os << "-"; break;
    case UnOp::Not: os << "~"; break;
    case UnOp::LNot: os << "!"; break;
    }
    printExpr(u->x);
    return;
  }
  case Expr::Kind::Cast: {
    auto *c = llvm::cast<Cast>(e);
    switch (c->style) {
    case Cast::Style::CStyle:
      os << "(";
      printType(c->to);
      os << ")";
      printExpr(c->x);
      return;
    case Cast::Style::Static:
      os << "static_cast<";
      printType(c->to);
      os << ">(";
      printExpr(c->x);
      os << ")";
      return;
    case Cast::Style::AsType:
      os << "as_type<";
      printType(c->to);
      os << ">(";
      printExpr(c->x);
      os << ")";
      return;
    }
    return;
  }
  case Expr::Kind::Call: {
    auto *c = llvm::cast<Call>(e);
    os << c->callee;
    if (!c->templateArgs.empty()) {
      os << "<";
      for (auto [i, ta] : llvm::enumerate(c->templateArgs)) {
        if (i)
          os << ", ";
        printType(ta);
      }
      os << ">";
    }
    os << "(";
    for (auto [i, a] : llvm::enumerate(c->args)) {
      if (i)
        os << ", ";
      printExpr(a);
    }
    os << ")";
    return;
  }
  case Expr::Kind::Ternary: {
    auto *t = llvm::cast<Ternary>(e);
    printExpr(t->c);
    os << " ? ";
    printExpr(t->a);
    os << " : ";
    printExpr(t->b);
    return;
  }
  case Expr::Kind::Subscript: {
    auto *s = llvm::cast<Subscript>(e);
    printExpr(s->base);
    os << "[";
    printExpr(s->idx);
    os << "]";
    return;
  }
  case Expr::Kind::Member: {
    auto *m = llvm::cast<Member>(e);
    printExpr(m->base);
    os << "." << m->field;
    return;
  }
  case Expr::Kind::Deref:
    os << "*";
    printExpr(llvm::cast<Deref>(e)->x);
    return;
  case Expr::Kind::AddrOf:
    os << "&";
    printExpr(llvm::cast<AddrOf>(e)->x);
    return;
  case Expr::Kind::Paren:
    os << "(";
    printExpr(llvm::cast<Paren>(e)->x);
    os << ")";
    return;
  case Expr::Kind::Raw:
    os << llvm::cast<Raw>(e)->text;
    return;
  }
  llvm_unreachable("bad Expr kind");
}

//===----------------------------------------------------------------------===//
// Attributes & params
//===----------------------------------------------------------------------===//

void MSLPrinter::printAttr(const Attr *a) {
  switch (a->kind) {
  case Attr::Kind::Buffer:
    os << "[[buffer(" << a->n << ")]]";
    return;
  case Attr::Kind::ThreadgroupPosInGrid:
    os << "[[threadgroup_position_in_grid]]";
    return;
  case Attr::Kind::ThreadPosInThreadgroup:
    os << "[[thread_position_in_threadgroup]]";
    return;
  case Attr::Kind::ThreadgroupsPerGrid:
    os << "[[threadgroups_per_grid]]";
    return;
  case Attr::Kind::MaxTotalThreads:
    os << "[[max_total_threads_per_threadgroup(" << a->n << ")]]";
    return;
  }
  llvm_unreachable("bad Attr kind");
}

void MSLPrinter::printParam(const Param &p) {
  printType(p.ty);
  os << " " << p.name;
  if (p.attr) {
    os << " ";
    printAttr(p.attr);
  }
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void MSLPrinter::flushBarrier() {
  if (!barrierPending)
    return;
  bool device = barrierPendingDevice;
  barrierPending = false;
  barrierPendingDevice = false;
  if (device)
    ind() << "threadgroup_barrier(mem_flags::mem_threadgroup | "
             "mem_flags::mem_device);\n";
  else
    ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
}

// Render decl/assign as a fragment (no indent, no trailing `;`/newline) for use
// inside a `for (...)` header.
void MSLPrinter::printStmtInline(const Stmt *s) {
  if (auto *d = llvm::dyn_cast<DeclStmt>(s)) {
    printType(d->ty);
    os << " " << d->name;
    if (d->init) {
      os << " = ";
      printExpr(d->init);
    }
    return;
  }
  if (auto *a = llvm::dyn_cast<AssignStmt>(s)) {
    printExpr(a->lhs);
    os << " = ";
    printExpr(a->rhs);
    return;
  }
  llvm_unreachable("printStmtInline: unsupported stmt");
}

void MSLPrinter::printBlock(const Block &b) {
  for (const Stmt *s : b) {
    if (auto *bar = llvm::dyn_cast<BarrierStmt>(s)) {
      barrierPending = true;
      barrierPendingDevice = barrierPendingDevice || bar->device;
      continue;
    }
    flushBarrier();
    printStmt(s);
  }
}

void MSLPrinter::printStmt(const Stmt *s) {
  switch (s->kind) {
  case Stmt::Kind::Decl: {
    auto *d = llvm::cast<DeclStmt>(s);
    ind();
    printType(d->ty);
    os << " " << d->name;
    if (d->init) {
      os << " = ";
      printExpr(d->init);
    }
    os << ";\n";
    return;
  }
  case Stmt::Kind::ArrayDecl: {
    auto *d = llvm::cast<ArrayDeclStmt>(s);
    ind();
    printType(d->elem);
    os << " " << d->name << "[" << d->n << "]";
    if (!d->init.empty()) {
      os << " = {";
      for (auto [i, e] : llvm::enumerate(d->init)) {
        if (i)
          os << ", ";
        printExpr(e);
      }
      os << "}";
    }
    os << ";\n";
    return;
  }
  case Stmt::Kind::Assign: {
    auto *a = llvm::cast<AssignStmt>(s);
    ind();
    printExpr(a->lhs);
    os << " = ";
    printExpr(a->rhs);
    os << ";\n";
    return;
  }
  case Stmt::Kind::Expr:
    ind();
    printExpr(llvm::cast<ExprStmt>(s)->e);
    os << ";\n";
    return;
  case Stmt::Kind::Return: {
    auto *r = llvm::cast<ReturnStmt>(s);
    ind() << "return";
    if (!r->structFields.empty()) {
      os << " { ";
      for (auto [i, f] : llvm::enumerate(r->structFields)) {
        if (i)
          os << ", ";
        printExpr(f);
      }
      os << " }";
    } else if (r->val) {
      os << " ";
      printExpr(r->val);
    }
    os << ";\n";
    return;
  }
  case Stmt::Kind::Break:
    ind() << "break;\n";
    return;
  case Stmt::Kind::Continue:
    ind() << "continue;\n";
    return;
  case Stmt::Kind::Barrier:
    // Barriers only ever go through printBlock's peephole.
    barrierPending = true;
    barrierPendingDevice =
        barrierPendingDevice || llvm::cast<BarrierStmt>(s)->device;
    return;
  case Stmt::Kind::Raw: {
    auto *r = llvm::cast<RawStmt>(s);
    if (r->verbatim)
      os << r->text;
    else
      ind() << r->text << "\n";
    return;
  }
  case Stmt::Kind::KernelFn: {
    auto *fn = llvm::cast<KernelFn>(s);
    if (fn->maxThreads) {
      printAttr(fn->maxThreads);
      os << "\n";
    }
    os << "kernel void " << fn->name << "(\n";
    for (auto [i, p] : llvm::enumerate(fn->params)) {
      os << "    ";
      printParam(p);
      if (i + 1 < fn->params.size())
        os << ",";
      os << "\n";
    }
    os << ") {\n";
    ++indent;
    printBlock(fn->body);
    flushBarrier();
    --indent;
    os << "}\n";
    return;
  }
  case Stmt::Kind::DeviceFn: {
    auto *fn = llvm::cast<DeviceFn>(s);
    printType(fn->retType);
    os << " " << fn->name << "(";
    for (auto [i, p] : llvm::enumerate(fn->params)) {
      if (i)
        os << ", ";
      printParam(p);
    }
    os << ") {\n";
    ++indent;
    printBlock(fn->body);
    flushBarrier();
    --indent;
    os << "}\n";
    return;
  }
  case Stmt::Kind::ForScope: {
    auto *f = llvm::cast<ForScope>(s);
    ind() << "for (";
    printStmtInline(f->initDecl);
    os << "; ";
    printExpr(f->cond);
    os << "; ";
    printStmtInline(f->step);
    os << ") {\n";
    ++indent;
    printBlock(f->body);
    flushBarrier();
    --indent;
    ind() << "}\n";
    return;
  }
  case Stmt::Kind::TripCountForScope: {
    auto *f = llvm::cast<TripCountForScope>(s);
    ind() << "for (";
    printType(f->counterTy);
    os << " " << f->counter << " = 0; ; " << f->counter << " += 1) {\n";
    ++indent;
    printBlock(f->body);
    flushBarrier();
    --indent;
    ind() << "}\n";
    return;
  }
  case Stmt::Kind::IfScope: {
    auto *f = llvm::cast<IfScope>(s);
    ind() << "if (";
    printExpr(f->cond);
    os << ") {\n";
    ++indent;
    printBlock(f->thenB);
    flushBarrier();
    --indent;
    if (f->hasElse) {
      ind() << "} else {\n";
      ++indent;
      printBlock(f->elseB);
      flushBarrier();
      --indent;
    }
    ind() << "}\n";
    return;
  }
  case Stmt::Kind::WhileScope: {
    auto *f = llvm::cast<WhileScope>(s);
    ind() << "while (";
    if (f->cond)
      printExpr(f->cond);
    else
      os << "true";
    os << ") {\n";
    ++indent;
    printBlock(f->body);
    flushBarrier();
    --indent;
    ind() << "}\n";
    return;
  }
  case Stmt::Kind::StateMachineScope: {
    auto *f = llvm::cast<StateMachineScope>(s);
    ind();
    printType(f->stateTy);
    os << " " << f->stateVar << " = 0;\n";
    ind() << "while (true) {\n";
    ++indent;
    for (auto [i, c] : llvm::enumerate(f->cases)) {
      ind() << (i ? "else if" : "if") << " (" << f->stateVar
            << " == " << c.label << ") {\n";
      ++indent;
      printBlock(c.body);
      flushBarrier();
      --indent;
      ind() << "}\n";
    }
    --indent;
    ind() << "}\n";
    return;
  }
  case Stmt::Kind::PlainScope: {
    auto *f = llvm::cast<PlainScope>(s);
    ind() << "{\n";
    ++indent;
    printBlock(f->body);
    flushBarrier();
    --indent;
    ind() << "}\n";
    return;
  }
  }
  llvm_unreachable("bad Stmt kind");
}

void MSLPrinter::printProto(const DeviceFn *fn) {
  printType(fn->retType);
  os << " " << fn->name << "(";
  for (auto [i, p] : llvm::enumerate(fn->params)) {
    if (i)
      os << ", ";
    printParam(p);
  }
  os << ");\n";
}

//===----------------------------------------------------------------------===//
// Preamble (fixed; verbatim from EmitMSL L126-136)
//===----------------------------------------------------------------------===//

void MSLPrinter::printPreamble() {
  os << "#include <metal_stdlib>\n";
  os << "#include <metal_simdgroup_matrix>\n";
  os << "using namespace metal;\n\n";
  os << "static inline float tt_erf(float x){\n"
        "  float t = 1.0f/(1.0f+0.5f*metal::fabs(x));\n"
        "  float y = t*metal::exp(-x*x-1.26551223f+t*(1.00002368f+t*(0.37409196f"
        "+t*(0.09678418f+t*(-0.18628806f+t*(0.27886807f+t*(-1.13520398f"
        "+t*(1.48851587f+t*(-0.82215223f+t*0.17087277f)))))))));\n"
        "  float r = 1.0f - y;\n"
        "  return x >= 0.0f ? r : -r;\n"
        "}\n\n";
}

} // namespace mlir::triton::applegpu::msl
