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
  case Scalar::F32:
    return "float";
  case Scalar::F16:
    return "half";
  case Scalar::BF16:
    return "bfloat";
  case Scalar::I1:
    return "bool";
  case Scalar::I8:
    return "char";
  case Scalar::I16:
    return "short";
  case Scalar::I32:
    return "int";
  case Scalar::I64:
    return "long";
  case Scalar::U8:
    return "uchar";
  case Scalar::U16:
    return "ushort";
  case Scalar::U32:
    return "uint";
  case Scalar::U64:
    return "ulong";
  case Scalar::Void:
    return "void";
  case Scalar::SizeT:
    return "size_t";
  }
  llvm_unreachable("bad Scalar");
}

static llvm::StringRef atomicName(Scalar s) {
  switch (s) {
  case Scalar::F32:
    return "atomic_float";
  case Scalar::I32:
    return "atomic_int";
  case Scalar::U32:
    return "atomic_uint";
  case Scalar::I64:
    return "atomic_long";
  case Scalar::U64:
    return "atomic_ulong";
  default:
    llvm_unreachable("no MSL atomic_* type for this Scalar");
  }
}

static llvm::StringRef matrixName(MatrixType::Elem e) {
  switch (e) {
  case MatrixType::Elem::Half:
    return "simdgroup_half8x8";
  case MatrixType::Elem::Bfloat:
    return "simdgroup_bfloat8x8";
  case MatrixType::Elem::Float:
    return "simdgroup_float8x8";
  }
  llvm_unreachable("bad Matrix elem");
}

static llvm::StringRef addrSpaceName(AddrSpace as) {
  switch (as) {
  case AddrSpace::None:
    return "";
  case AddrSpace::Device:
    return "device ";
  case AddrSpace::Threadgroup:
    return "threadgroup ";
  case AddrSpace::Constant:
    return "constant ";
  case AddrSpace::Thread:
    return "thread ";
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
    // `volatile` qualifies the pointee and precedes the address space;
    // `coherent(<as>)` is an address-space attribute and follows it. Both
    // compose with the address space rather than replacing it.
    if (p->volatile_)
      os << "volatile ";
    os << addrSpaceName(p->as);
    if (p->coherent)
      os << "coherent(" << addrSpaceName(p->as).rtrim() << ") ";
    printType(p->pointee);
    os << (p->spaceBeforeStar ? " *" : "*");
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
  case BinOp::Add:
    return " + ";
  case BinOp::Sub:
    return " - ";
  case BinOp::Mul:
    return " * ";
  case BinOp::Div:
    return " / ";
  case BinOp::Rem:
    return " % ";
  case BinOp::And:
    return " & ";
  case BinOp::Or:
    return " | ";
  case BinOp::Xor:
    return " ^ ";
  case BinOp::Shl:
    return " << ";
  case BinOp::Shr:
    return " >> ";
  case BinOp::Lt:
    return " < ";
  case BinOp::Le:
    return " <= ";
  case BinOp::Gt:
    return " > ";
  case BinOp::Ge:
    return " >= ";
  case BinOp::Eq:
    return " == ";
  case BinOp::Ne:
    return " != ";
  case BinOp::LAnd:
    return " && ";
  case BinOp::LOr:
    return " || ";
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
    case UnOp::Neg:
      os << "-";
      break;
    case UnOp::Not:
      os << "~";
      break;
    case UnOp::LNot:
      os << "!";
      break;
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
    os << (a->compound == AssignStmt::Compound::Add ? " += " : " = ");
    printExpr(a->rhs);
    return;
  }
  if (auto *e = llvm::dyn_cast<ExprStmt>(s)) {
    printExpr(e->e);
    return;
  }
  if (llvm::isa<BreakStmt>(s)) {
    os << "break";
    return;
  }
  if (llvm::isa<ContinueStmt>(s)) {
    os << "continue";
    return;
  }
  llvm_unreachable("printStmtInline: unsupported stmt");
}

void MSLPrinter::printBlock(const Block &b) {
  for (const Stmt *s : b) {
    if (auto *bar = llvm::dyn_cast<BarrierStmt>(s)) {
      if (bar->hard) {
        // Flush any pending soft barrier, then emit this one verbatim; it never
        // collapses with an adjacent barrier.
        flushBarrier();
        if (bar->simdOnly)
          ind() << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
        else if (bar->device)
          ind() << "threadgroup_barrier(mem_flags::mem_threadgroup | "
                   "mem_flags::mem_device);\n";
        else
          ind() << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        continue;
      }
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
    os << (a->compound == AssignStmt::Compound::Add ? " += " : " = ");
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
    // printBlock intercepts every BarrierStmt before printStmt sees it. Setting
    // barrierPending here without a flush would silently swallow the barrier.
    llvm_unreachable("BarrierStmt must go through printBlock's peephole");
  case Stmt::Kind::CompactIf: {
    auto *c = llvm::cast<CompactIfStmt>(s);
    if (c->parenCond) {
      ind() << "if (";
      printExpr(c->cond);
      os << ") ";
    } else {
      ind() << "if ";
      printExpr(c->cond);
      os << " ";
    }
    printStmtInline(c->then);
    os << ";\n";
    return;
  }
  case Stmt::Kind::Raw: {
    ind() << llvm::cast<RawStmt>(s)->text << "\n";
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
        os << ",\n";
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
    if (f->compact) {
      // `for (...) <expr>;` on one line.
      os << ") ";
      printExpr(llvm::cast<ExprStmt>(f->body.front())->e);
      os << ";\n";
      return;
    }
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
    // iv decl + compact break guard first (never floated into the header).
    printStmt(f->ivDecl);
    ind() << "if (!(";
    printExpr(f->guard);
    os << ")) break;\n";
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
    // A spin-wait with an empty body renders on one line: `while (c) {}`.
    if (f->body.empty()) {
      os << ") {}\n";
      return;
    }
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

void MSLPrinter::printPreamble(const HelperSet &h) {
  os << "#include <metal_stdlib>\n";
  os << "#include <metal_simdgroup_matrix>\n";
  os << "using namespace metal;\n\n";
  // air.simdgroup_async_copy_2d has no MSL spelling; these resolve at metallib
  // link time against the prebuilt AIR shim (async_copy_shim.ll). The presence
  // of this symbol is also what tells the driver to take the linking compile
  // path, so the name must match _ASYNC_COPY_SYM in compiler.py.
  if (h.tgAsyncCopy)
    os << R"MSL(extern "C" ulong __triton_tg_async_copy_begin_4(threadgroup void*, ulong, device const void*, ulong, ulong, ulong);
extern "C" ulong __triton_tg_async_copy_begin_2(threadgroup void*, ulong, device const void*, ulong, ulong, ulong);
extern "C" ulong __triton_tg_async_copy_begin_1(threadgroup void*, ulong, device const void*, ulong, ulong, ulong);
extern "C" ulong __triton_tg_async_copy_begin_4_tr(threadgroup void*, ulong, device const void*, ulong, ulong, ulong);
extern "C" ulong __triton_tg_async_copy_begin_2_tr(threadgroup void*, ulong, device const void*, ulong, ulong, ulong);
extern "C" ulong __triton_tg_async_copy_begin_1_tr(threadgroup void*, ulong, device const void*, ulong, ulong, ulong);
extern "C" void __triton_tg_async_copy_wait(ulong);

)MSL";
  if (h.erf)
    os << R"MSL(static inline float tt_erf(float x){
  float t = 1.0f/(1.0f+0.5f*metal::fabs(x));
  float y = t*metal::exp(-x*x-1.26551223f+t*(1.00002368f+t*(0.37409196f+t*(0.09678418f+t*(-0.18628806f+t*(0.27886807f+t*(-1.13520398f+t*(1.48851587f+t*(-0.82215223f+t*0.17087277f)))))))));
  float r = 1.0f - y;
  return x >= 0.0f ? r : -r;
}

)MSL";
  printNarrowingHelpers(h);
  if (h.fp8)
    printFp8Helpers();
  printAtomicHelpers(h);
}

//===----------------------------------------------------------------------===//
// fp32 -> half/bfloat narrowing helpers (IEEE-exact; used by fp_to_fp and the
// emulated fp atomics). tt_rtne_* is round-to-nearest-even with full
// NaN/Inf/overflow/subnormal handling; tt_rtz_* is round-toward-zero;
// tt_rtne_int_* is the integer round variant the CAS loops narrow with (an
// (half)/(bfloat) cast whose only consumer re-widens gets folded away by Metal
// fast-math, dropping the round).
//===----------------------------------------------------------------------===//

void MSLPrinter::printNarrowingHelpers(const HelperSet &h) {
  if (h.rtneHalf)
    os << R"MSL(static inline half tt_rtne_half(float v){
  uint u = as_type<uint>(v);
  ushort bits;
  uint sgn = (u >> 16) & 0x8000u;
  int e32 = (int)((u >> 23) & 0xffu);
  uint mant = u & 0x7fffffu;
  int ex = e32 - 112;
  if (e32 == 0xff) {
    bits = (ushort)(sgn | 0x7c00u | (mant ? 0x200u : 0u));
  } else if (ex >= 31) {
    bits = (ushort)(sgn | 0x7c00u);
  } else if (ex <= 0) {
    if (ex < -10) { bits = (ushort)sgn; }
    else {
      uint fm = mant | 0x800000u;
      int sh = 14 - ex;
      uint m = fm >> sh;
      uint rem = fm & ((1u << sh) - 1u);
      uint half_ = 1u << (sh - 1);
      if (rem > half_ || (rem == half_ && (m & 1u))) m += 1;
      bits = (ushort)(sgn | m);
    }
  } else {
    uint m = mant >> 13;
    uint rem = mant & 0x1fffu;
    bits = (ushort)(sgn | ((uint)ex << 10) | m);
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) bits += 1;
  }
  return as_type<half>(bits);
}

)MSL";
  if (h.rtneBfloat)
    os << R"MSL(static inline bfloat tt_rtne_bfloat(float v){
  uint u = as_type<uint>(v);
  ushort bits;
  int e32 = (int)((u >> 23) & 0xffu);
  uint mant = u & 0x7fffffu;
  if (e32 == 0xff) {
    bits = (ushort)(((u >> 16) & 0xffffu) | (mant ? 0x40u : 0u));
  } else {
    uint r = (u >> 16) & 1u;
    uint t = (u + 0x7fffu + r);
    bits = (ushort)((t >> 16) & 0xffffu);
  }
  return as_type<bfloat>(bits);
}

)MSL";
  if (h.rtzHalf)
    os << R"MSL(static inline half tt_rtz_half(float v){
  uint u = as_type<uint>(v);
  ushort bits;
  uint sgn = (u >> 16) & 0x8000u;
  int ex = (int)((u >> 23) & 0xffu) - 112;
  uint mant = u & 0x7fffffu;
  if (((u >> 23) & 0xffu) == 0xffu) {
    bits = (ushort)(sgn | 0x7c00u | (mant ? 0x200u : 0u));
  } else if (ex >= 31) {
    bits = (ushort)(sgn | 0x7bffu);
  } else if (ex <= 0) {
    if (ex < -10) { bits = (ushort)sgn; }
    else { uint m = (mant | 0x800000u) >> (14 - ex); bits = (ushort)(sgn | m); }
  } else {
    bits = (ushort)(sgn | ((uint)ex << 10) | (mant >> 13));
  }
  return as_type<half>(bits);
}

)MSL";
  if (h.rtzBfloat)
    os << R"MSL(static inline bfloat tt_rtz_bfloat(float v){
  uint u = as_type<uint>(v);
  return as_type<bfloat>((ushort)((u >> 16) & 0xffffu));
}

)MSL";
  if (h.rtneIntHalf)
    os << R"MSL(static inline half tt_rtne_int_half(float v){
  uint u = as_type<uint>(v);
  ushort bits;
  uint sgn = (u >> 16) & 0x8000u;
  int ex = (int)((u >> 23) & 0xffu) - 112;
  uint mant = u & 0x7fffffu;
  if (ex <= 0) {
    bits = (ushort)sgn;
  } else if (ex >= 31) {
    bits = (ushort)(sgn | 0x7c00u);
  } else {
    uint m = mant >> 13;
    uint rem = mant & 0x1fffu;
    bits = (ushort)(sgn | ((uint)ex << 10) | m);
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) bits += 1;
  }
  return as_type<half>(bits);
}

)MSL";
  if (h.rtneIntBfloat)
    os << R"MSL(static inline bfloat tt_rtne_int_bfloat(float v){
  uint u = as_type<uint>(v);
  uint r = (u >> 16) & 1u;
  return as_type<bfloat>((ushort)(((u + 0x7fffu + r) >> 16) & 0xffffu));
}

)MSL";
}

//===----------------------------------------------------------------------===//
// fp8 (OCP e4m3fn / e5m2) pack/unpack: no native MSL fp8 scalar, so fp8 is
// uchar storage. RTNE + saturation on the way in; e4m3 has no inf (NaN only,
// max 448); e5m2 is ieee-like (inf/nan). Emitted only when a kernel uses fp8.
//===----------------------------------------------------------------------===//

void MSLPrinter::printFp8Helpers() {
  os << R"MSL(static inline uchar tt_f32_to_fp8e4m3_rtne(float v){
  uint u = as_type<uint>(v);
  uint sgn = (u >> 24) & 0x80u;
  int e32 = (int)((u >> 23) & 0xffu);
  uint mant = u & 0x7fffffu;
  if (e32 == 0xff) { return (uchar)(sgn | 0x7fu); }
  int ex = e32 - 127 + 7;
  if (ex >= 16 || (ex == 15 && mant > 0x600000u)) { return (uchar)(sgn | 0x7eu); }
  if (ex <= 0) {
    if (ex < -6) { return (uchar)sgn; }
    uint fm = mant | 0x800000u;
    int sh = 21 - ex;
    uint m = fm >> sh;
    uint rem = fm & ((1u << sh) - 1u);
    uint half_ = 1u << (sh - 1);
    if (rem > half_ || (rem == half_ && (m & 1u))) m += 1;
    return (uchar)(sgn | m);
  }
  uint m = mant >> 20;
  uint rem = mant & 0xfffffu;
  uint bits = sgn | ((uint)ex << 3) | m;
  if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) bits += 1;
  return (uchar)bits;
}

static inline float tt_fp8e4m3_to_f32(uchar b){
  uint sgn = ((uint)b & 0x80u) << 24;
  uint e = ((uint)b >> 3) & 0xfu;
  uint m = (uint)b & 0x7u;
  if (e == 0) {
    if (m == 0) { return as_type<float>(sgn); }
    int sh = 0;
    while ((m & 0x8u) == 0) { m <<= 1; sh += 1; }
    m &= 0x7u;
    uint e32 = (uint)(127 - 6 - sh);
    return as_type<float>(sgn | (e32 << 23) | (m << 20));
  }
  if (e == 0xf && m == 0x7u) { return as_type<float>(sgn | 0x7f800000u | 0x400000u); }
  uint e32 = e + 127 - 7;
  return as_type<float>(sgn | (e32 << 23) | (m << 20));
}

static inline uchar tt_f32_to_fp8e5m2_rtne(float v){
  uint u = as_type<uint>(v);
  uint sgn = (u >> 24) & 0x80u;
  int e32 = (int)((u >> 23) & 0xffu);
  uint mant = u & 0x7fffffu;
  if (e32 == 0xff) { return (uchar)(sgn | 0x7cu | (mant ? 0x2u : 0u)); }
  int ex = e32 - 127 + 15;
  if (ex >= 31) { return (uchar)(sgn | 0x7cu); }
  if (ex <= 0) {
    if (ex < -2) { return (uchar)sgn; }
    uint fm = mant | 0x800000u;
    int sh = 22 - ex;
    uint m = fm >> sh;
    uint rem = fm & ((1u << sh) - 1u);
    uint half_ = 1u << (sh - 1);
    if (rem > half_ || (rem == half_ && (m & 1u))) m += 1;
    return (uchar)(sgn | m);
  }
  uint m = mant >> 21;
  uint rem = mant & 0x1fffffu;
  uint bits = sgn | ((uint)ex << 2) | m;
  if (rem > 0x100000u || (rem == 0x100000u && (m & 1u))) bits += 1;
  return (uchar)bits;
}

static inline float tt_fp8e5m2_to_f32(uchar b){
  uint sgn = ((uint)b & 0x80u) << 24;
  uint e = ((uint)b >> 2) & 0x1fu;
  uint m = (uint)b & 0x3u;
  if (e == 0) {
    if (m == 0) { return as_type<float>(sgn); }
    int sh = 0;
    while ((m & 0x4u) == 0) { m <<= 1; sh += 1; }
    m &= 0x3u;
    uint e32 = (uint)(127 - 14 - sh);
    return as_type<float>(sgn | (e32 << 23) | (m << 21));
  }
  if (e == 0x1f) { return as_type<float>(sgn | 0x7f800000u | (m << 21)); }
  uint e32 = e + 127 - 15;
  return as_type<float>(sgn | (e32 << 23) | (m << 21));
}

)MSL";
}

//===----------------------------------------------------------------------===//
// Emulated fp atomic RMW helpers (Metal's atomic_float is fetch_add-only;
// max/min/xchg and all packed-fp16 RMW need CAS emulation). `op`: 0=add,
// 1=max, 2=min, 3=xchg. Each returns the pre-op value.
//===----------------------------------------------------------------------===//

void MSLPrinter::printAtomicHelpers(const HelperSet &h) {
  if (h.atomicF32)
    os << R"MSL(static inline float tt_atomic_rmw_f32(device atomic_uint *p, float v, int op){
  uint word = atomic_load_explicit(p, memory_order_relaxed);
  while (true) {
    float cur = as_type<float>(word);
    float nv = (op == 0) ? (cur + v) : (op == 1) ? fmax(cur, v) : (op == 2) ? fmin(cur, v) : v;
    uint nw = as_type<uint>(nv);
    if (atomic_compare_exchange_weak_explicit(p, &word, nw, memory_order_relaxed, memory_order_relaxed)) return cur;
  }
}

)MSL";
  if (h.atomicPacked16)
    os << R"MSL(template <typename T, T (*Narrow)(float)>
static inline T tt_atomic_rmw_packed16(device atomic_uint *word, bool isHigh, float v, int op){
  T vh = Narrow(v);
  uint w = atomic_load_explicit(word, memory_order_relaxed);
  while (true) {
    ushort lane = (ushort)((isHigh) ? (w >> 16) : (w & 0xffffu));
    T cur = as_type<T>(lane);
    T nl = (T)((op == 0) ? (cur + vh) : (op == 1) ? fmax(cur, vh) : (op == 2) ? fmin(cur, vh) : vh);
    uint nw = (isHigh) ? ((w & 0x0000ffffu) | ((uint)as_type<ushort>(nl) << 16)) : ((w & 0xffff0000u) | (uint)as_type<ushort>(nl));
    if (atomic_compare_exchange_weak_explicit(word, &w, nw, memory_order_relaxed, memory_order_relaxed)) return cur;
  }
}

)MSL";
}

} // namespace mlir::triton::applegpu::msl
