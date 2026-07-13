// MSLAst.h - typed MSL AST node set + arena context.
//
// Build-once/print-once nodes, arena-allocated on MSLContext. All layout,
// braces, indentation and the barrier peephole live in MSLPrinter, never here.
// See MSL_AST_DESIGN.md for the frozen contract.
#ifndef MSL_AST_H
#define MSL_AST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include <cstring>
#include <utility>

namespace mlir::triton::applegpu::msl {

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

struct Type {
  enum class Kind { Scalar, Vector, Matrix, Atomic, Pointer, Named };
  const Kind kind;
  explicit Type(Kind k) : kind(k) {}
};

enum class Scalar {
  F32, F16, BF16, I1, I8, I16, I32, I64, U8, U16, U32, U64, Void, SizeT
};

struct ScalarType : Type {
  Scalar s;
  explicit ScalarType(Scalar s) : Type(Kind::Scalar), s(s) {}
  static bool classof(const Type *t) { return t->kind == Kind::Scalar; }
};

struct VectorType : Type {
  Scalar elem;
  unsigned n;
  VectorType(Scalar e, unsigned n) : Type(Kind::Vector), elem(e), n(n) {}
  static bool classof(const Type *t) { return t->kind == Kind::Vector; }
};

struct MatrixType : Type {
  enum class Elem { Half, Bfloat, Float };
  Elem elem;
  explicit MatrixType(Elem e) : Type(Kind::Matrix), elem(e) {}
  static bool classof(const Type *t) { return t->kind == Kind::Matrix; }
};

struct AtomicType : Type {
  Scalar elem;
  explicit AtomicType(Scalar e) : Type(Kind::Atomic), elem(e) {}
  static bool classof(const Type *t) { return t->kind == Kind::Atomic; }
};

enum class AddrSpace { None, Device, Threadgroup, Constant, Thread };

struct PointerType : Type {
  Type *pointee;
  AddrSpace as;
  bool coherent;
  bool volatile_;
  PointerType(Type *p, AddrSpace as, bool coherent, bool vol)
      : Type(Kind::Pointer), pointee(p), as(as), coherent(coherent),
        volatile_(vol) {}
  static bool classof(const Type *t) { return t->kind == Kind::Pointer; }
};

struct NamedType : Type {
  llvm::StringRef name;
  explicit NamedType(llvm::StringRef n) : Type(Kind::Named), name(n) {}
  static bool classof(const Type *t) { return t->kind == Kind::Named; }
};

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

struct Expr {
  enum class Kind {
    VarRef, Literal, Binary, Unary, Cast, Call, Ternary, Subscript, Member,
    Deref, AddrOf, Paren, Raw
  };
  const Kind kind;
  explicit Expr(Kind k) : kind(k) {}
};

struct VarRef : Expr {
  llvm::StringRef name;
  explicit VarRef(llvm::StringRef n) : Expr(Kind::VarRef), name(n) {}
  static bool classof(const Expr *e) { return e->kind == Kind::VarRef; }
};

struct Literal : Expr {
  llvm::StringRef text;
  explicit Literal(llvm::StringRef t) : Expr(Kind::Literal), text(t) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Literal; }
};

enum class BinOp {
  Add, Sub, Mul, Div, Rem, And, Or, Xor, Shl, Shr,
  Lt, Le, Gt, Ge, Eq, Ne, LAnd, LOr
};

struct Binary : Expr {
  BinOp op;
  Expr *lhs;
  Expr *rhs;
  Binary(BinOp op, Expr *l, Expr *r)
      : Expr(Kind::Binary), op(op), lhs(l), rhs(r) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Binary; }
};

enum class UnOp { Neg, Not, LNot };

struct Unary : Expr {
  UnOp op;
  Expr *x;
  Unary(UnOp op, Expr *x) : Expr(Kind::Unary), op(op), x(x) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Unary; }
};

struct Cast : Expr {
  enum class Style { CStyle, Static, AsType };
  Style style;
  Type *to;
  Expr *x;
  Cast(Style s, Type *to, Expr *x) : Expr(Kind::Cast), style(s), to(to), x(x) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Cast; }
};

struct Call : Expr {
  llvm::StringRef callee;
  llvm::SmallVector<Type *, 2> templateArgs;
  llvm::SmallVector<Expr *, 4> args;
  Call(llvm::StringRef c, llvm::ArrayRef<Type *> ta, llvm::ArrayRef<Expr *> a)
      : Expr(Kind::Call), callee(c), templateArgs(ta), args(a) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Call; }
};

struct Ternary : Expr {
  Expr *c;
  Expr *a;
  Expr *b;
  Ternary(Expr *c, Expr *a, Expr *b)
      : Expr(Kind::Ternary), c(c), a(a), b(b) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Ternary; }
};

struct Subscript : Expr {
  Expr *base;
  Expr *idx;
  Subscript(Expr *b, Expr *i) : Expr(Kind::Subscript), base(b), idx(i) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Subscript; }
};

struct Member : Expr {
  Expr *base;
  llvm::StringRef field;
  Member(Expr *b, llvm::StringRef f) : Expr(Kind::Member), base(b), field(f) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Member; }
};

struct Deref : Expr {
  Expr *x;
  explicit Deref(Expr *x) : Expr(Kind::Deref), x(x) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Deref; }
};

struct AddrOf : Expr {
  Expr *x;
  explicit AddrOf(Expr *x) : Expr(Kind::AddrOf), x(x) {}
  static bool classof(const Expr *e) { return e->kind == Kind::AddrOf; }
};

struct Paren : Expr {
  Expr *x;
  explicit Paren(Expr *x) : Expr(Kind::Paren), x(x) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Paren; }
};

struct Raw : Expr {
  llvm::StringRef text;
  explicit Raw(llvm::StringRef t) : Expr(Kind::Raw), text(t) {}
  static bool classof(const Expr *e) { return e->kind == Kind::Raw; }
};

//===----------------------------------------------------------------------===//
// Statements & scopes (a Scope is a Stmt that owns a `{ ... }` block)
//===----------------------------------------------------------------------===//

struct Stmt {
  enum class Kind {
    Decl, ArrayDecl, Assign, Expr, Return, Break, Continue, Barrier, Raw,
    // scopes
    KernelFn, DeviceFn, ForScope, TripCountForScope, IfScope, WhileScope,
    StateMachineScope, PlainScope
  };
  const Kind kind;
  explicit Stmt(Kind k) : kind(k) {}
};

using Block = llvm::SmallVector<Stmt *, 8>;

struct DeclStmt : Stmt {
  Type *ty;
  llvm::StringRef name;
  Expr *init; // nullable
  DeclStmt(Type *t, llvm::StringRef n, Expr *i)
      : Stmt(Kind::Decl), ty(t), name(n), init(i) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Decl; }
};

struct ArrayDeclStmt : Stmt {
  Type *elem;
  llvm::StringRef name;
  unsigned n;
  llvm::SmallVector<Expr *, 4> init;
  ArrayDeclStmt(Type *e, llvm::StringRef nm, unsigned n,
                llvm::ArrayRef<Expr *> init)
      : Stmt(Kind::ArrayDecl), elem(e), name(nm), n(n), init(init) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::ArrayDecl; }
};

struct AssignStmt : Stmt {
  Expr *lhs;
  Expr *rhs;
  // compound != None renders `lhs op= rhs` (e.g. the for-step `iv += st`).
  enum class Compound { None, Add } compound;
  AssignStmt(Expr *l, Expr *r, Compound c)
      : Stmt(Kind::Assign), lhs(l), rhs(r), compound(c) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Assign; }
};

struct ExprStmt : Stmt {
  Expr *e;
  explicit ExprStmt(Expr *e) : Stmt(Kind::Expr), e(e) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Expr; }
};

struct ReturnStmt : Stmt {
  Expr *val; // nullable
  llvm::SmallVector<Expr *, 4> structFields;
  ReturnStmt(Expr *v, llvm::ArrayRef<Expr *> fields)
      : Stmt(Kind::Return), val(v), structFields(fields) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Return; }
};

struct BreakStmt : Stmt {
  BreakStmt() : Stmt(Kind::Break) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Break; }
};

struct ContinueStmt : Stmt {
  ContinueStmt() : Stmt(Kind::Continue) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Continue; }
};

struct BarrierStmt : Stmt {
  bool device;
  explicit BarrierStmt(bool d) : Stmt(Kind::Barrier), device(d) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Barrier; }
};

struct RawStmt : Stmt {
  llvm::StringRef text;
  // verbatim => print text exactly (no indent, no trailing newline); used by the
  // transitional capture path where `text` already carries indentation+newlines.
  bool verbatim;
  RawStmt(llvm::StringRef t, bool verbatim)
      : Stmt(Kind::Raw), text(t), verbatim(verbatim) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::Raw; }
};

//===----------------------------------------------------------------------===//
// Params & attributes
//===----------------------------------------------------------------------===//

struct Attr {
  enum class Kind {
    Buffer, ThreadgroupPosInGrid, ThreadPosInThreadgroup, ThreadgroupsPerGrid,
    MaxTotalThreads
  };
  const Kind kind;
  unsigned n; // Buffer index / MaxTotalThreads count; unused otherwise
  Attr(Kind k, unsigned n) : kind(k), n(n) {}
};

struct Param {
  Type *ty;
  llvm::StringRef name;
  Attr *attr; // nullable
  Param(Type *t, llvm::StringRef n, Attr *a) : ty(t), name(n), attr(a) {}
};

//===----------------------------------------------------------------------===//
// Scopes
//===----------------------------------------------------------------------===//

struct KernelFn : Stmt {
  Attr *maxThreads; // nullable
  llvm::StringRef name;
  llvm::SmallVector<Param, 8> params;
  Block body;
  KernelFn(Attr *mt, llvm::StringRef n, llvm::ArrayRef<Param> p, Block b)
      : Stmt(Kind::KernelFn), maxThreads(mt), name(n), params(p),
        body(std::move(b)) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::KernelFn; }
};

struct DeviceFn : Stmt {
  Type *retType;
  llvm::StringRef name;
  llvm::SmallVector<Param, 8> params;
  Block body;
  DeviceFn(Type *rt, llvm::StringRef n, llvm::ArrayRef<Param> p, Block b)
      : Stmt(Kind::DeviceFn), retType(rt), name(n), params(p),
        body(std::move(b)) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::DeviceFn; }
};

struct ForScope : Stmt {
  Stmt *initDecl; // DeclStmt
  Expr *cond;
  Stmt *step; // AssignStmt (iv += st), rendered without trailing `;`
  Block body;
  ForScope(Stmt *init, Expr *c, Stmt *step, Block b)
      : Stmt(Kind::ForScope), initDecl(init), cond(c), step(step),
        body(std::move(b)) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::ForScope; }
};

// Wide-IV i64 loop: `for (long tc = 0; ; tc += 1) { <ivDecl> if(!(...)) break; }`
// The iv decl MUST be the first Stmt of body (never floated into the header) so
// the AGX i65 Gauss-sum fold can't fire.
struct TripCountForScope : Stmt {
  Type *counterTy;
  llvm::StringRef counter;
  Block body;
  TripCountForScope(Type *ct, llvm::StringRef c, Block b)
      : Stmt(Kind::TripCountForScope), counterTy(ct), counter(c),
        body(std::move(b)) {}
  static bool classof(const Stmt *s) {
    return s->kind == Kind::TripCountForScope;
  }
};

struct IfScope : Stmt {
  Expr *cond;
  Block thenB;
  Block elseB;
  bool hasElse;
  IfScope(Expr *c, Block t, Block e, bool hasElse)
      : Stmt(Kind::IfScope), cond(c), thenB(std::move(t)), elseB(std::move(e)),
        hasElse(hasElse) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::IfScope; }
};

struct WhileScope : Stmt {
  Expr *cond; // nullable => while (true)
  Block body;
  WhileScope(Expr *c, Block b)
      : Stmt(Kind::WhileScope), cond(c), body(std::move(b)) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::WhileScope; }
};

// `int <state> = 0; while (true) { if (state==L0){..} else if (state==L1){..} }`
struct StateMachineScope : Stmt {
  Type *stateTy;
  llvm::StringRef stateVar;
  struct Case {
    llvm::StringRef label;
    Block body;
  };
  llvm::SmallVector<Case, 4> cases;
  StateMachineScope(Type *st, llvm::StringRef sv,
                    llvm::SmallVector<Case, 4> cases)
      : Stmt(Kind::StateMachineScope), stateTy(st), stateVar(sv),
        cases(std::move(cases)) {}
  static bool classof(const Stmt *s) {
    return s->kind == Kind::StateMachineScope;
  }
};

struct PlainScope : Stmt {
  Block body;
  explicit PlainScope(Block b) : Stmt(Kind::PlainScope), body(std::move(b)) {}
  static bool classof(const Stmt *s) { return s->kind == Kind::PlainScope; }
};

//===----------------------------------------------------------------------===//
// Arena / factory
//===----------------------------------------------------------------------===//

class MSLContext {
public:
  // Strings are copied into the arena so callers may pass temporaries.
  llvm::StringRef save(llvm::StringRef s) {
    if (s.empty())
      return s;
    char *mem = alloc.Allocate<char>(s.size());
    std::memcpy(mem, s.data(), s.size());
    return llvm::StringRef(mem, s.size());
  }

  // --- Types ---
  ScalarType *scalar(Scalar s) { return make<ScalarType>(s); }
  VectorType *vector(Scalar e, unsigned n) { return make<VectorType>(e, n); }
  MatrixType *matrix(MatrixType::Elem e) { return make<MatrixType>(e); }
  AtomicType *atomic(Scalar e) { return make<AtomicType>(e); }
  PointerType *ptr(Type *p, AddrSpace as, bool coherent = false,
                   bool vol = false) {
    return make<PointerType>(p, as, coherent, vol);
  }
  NamedType *named(llvm::StringRef n) { return make<NamedType>(save(n)); }

  // --- Exprs ---
  VarRef *var(llvm::StringRef n) { return make<VarRef>(save(n)); }
  Literal *lit(llvm::StringRef t) { return make<Literal>(save(t)); }
  Binary *binary(BinOp op, Expr *l, Expr *r) { return make<Binary>(op, l, r); }
  Unary *unary(UnOp op, Expr *x) { return make<Unary>(op, x); }
  Cast *cast(Cast::Style s, Type *to, Expr *x) { return make<Cast>(s, to, x); }
  Call *call(llvm::StringRef c, llvm::ArrayRef<Type *> ta,
             llvm::ArrayRef<Expr *> a) {
    return make<Call>(save(c), ta, a);
  }
  Call *call(llvm::StringRef c, llvm::ArrayRef<Expr *> a) {
    return make<Call>(save(c), llvm::ArrayRef<Type *>{}, a);
  }
  Ternary *ternary(Expr *c, Expr *a, Expr *b) {
    return make<Ternary>(c, a, b);
  }
  Subscript *subscript(Expr *b, Expr *i) { return make<Subscript>(b, i); }
  Member *member(Expr *b, llvm::StringRef f) {
    return make<Member>(b, save(f));
  }
  Deref *deref(Expr *x) { return make<Deref>(x); }
  AddrOf *addrOf(Expr *x) { return make<AddrOf>(x); }
  Paren *paren(Expr *x) { return make<Paren>(x); }
  Raw *raw(llvm::StringRef t) { return make<Raw>(save(t)); }

  // --- Stmts ---
  DeclStmt *declStmt(Type *t, llvm::StringRef n, Expr *init = nullptr) {
    return make<DeclStmt>(t, save(n), init);
  }
  ArrayDeclStmt *arrayDeclStmt(Type *e, llvm::StringRef n, unsigned cnt,
                               llvm::ArrayRef<Expr *> init = {}) {
    return make<ArrayDeclStmt>(e, save(n), cnt, init);
  }
  AssignStmt *assignStmt(Expr *l, Expr *r) {
    return make<AssignStmt>(l, r, AssignStmt::Compound::None);
  }
  AssignStmt *addAssignStmt(Expr *l, Expr *r) {
    return make<AssignStmt>(l, r, AssignStmt::Compound::Add);
  }
  ExprStmt *exprStmt(Expr *e) { return make<ExprStmt>(e); }
  ReturnStmt *returnStmt(Expr *v = nullptr,
                         llvm::ArrayRef<Expr *> fields = {}) {
    return make<ReturnStmt>(v, fields);
  }
  BreakStmt *breakStmt() { return make<BreakStmt>(); }
  ContinueStmt *continueStmt() { return make<ContinueStmt>(); }
  BarrierStmt *barrier(bool device) { return make<BarrierStmt>(device); }
  RawStmt *rawStmt(llvm::StringRef t) { return make<RawStmt>(save(t), false); }
  RawStmt *rawVerbatim(llvm::StringRef t) {
    return make<RawStmt>(save(t), true);
  }

  // --- Attrs / Params ---
  Attr *bufferAttr(unsigned n) { return make<Attr>(Attr::Kind::Buffer, n); }
  Attr *tgPosAttr() {
    return make<Attr>(Attr::Kind::ThreadgroupPosInGrid, 0);
  }
  Attr *threadPosAttr() {
    return make<Attr>(Attr::Kind::ThreadPosInThreadgroup, 0);
  }
  Attr *tgsPerGridAttr() {
    return make<Attr>(Attr::Kind::ThreadgroupsPerGrid, 0);
  }
  Attr *maxThreadsAttr(unsigned n) {
    return make<Attr>(Attr::Kind::MaxTotalThreads, n);
  }
  Param param(Type *t, llvm::StringRef n, Attr *a = nullptr) {
    return Param(t, save(n), a);
  }

  // --- Scopes ---
  KernelFn *kernelFn(Attr *mt, llvm::StringRef n, llvm::ArrayRef<Param> p,
                     Block b) {
    return make<KernelFn>(mt, save(n), p, std::move(b));
  }
  DeviceFn *deviceFn(Type *rt, llvm::StringRef n, llvm::ArrayRef<Param> p,
                     Block b) {
    return make<DeviceFn>(rt, save(n), p, std::move(b));
  }
  ForScope *forScope(Stmt *init, Expr *c, Stmt *step, Block b) {
    return make<ForScope>(init, c, step, std::move(b));
  }
  TripCountForScope *tripCountForScope(Type *ct, llvm::StringRef c, Block b) {
    return make<TripCountForScope>(ct, save(c), std::move(b));
  }
  IfScope *ifScope(Expr *c, Block t) {
    return make<IfScope>(c, std::move(t), Block{}, false);
  }
  IfScope *ifElseScope(Expr *c, Block t, Block e) {
    return make<IfScope>(c, std::move(t), std::move(e), true);
  }
  WhileScope *whileScope(Expr *c, Block b) {
    return make<WhileScope>(c, std::move(b));
  }
  StateMachineScope *stateMachineScope(
      Type *st, llvm::StringRef sv,
      llvm::SmallVector<StateMachineScope::Case, 4> cases) {
    return make<StateMachineScope>(st, save(sv), std::move(cases));
  }
  PlainScope *plainScope(Block b) { return make<PlainScope>(std::move(b)); }

private:
  template <typename T, typename... Args> T *make(Args &&...args) {
    void *mem = alloc.Allocate(sizeof(T), alignof(T));
    return new (mem) T(std::forward<Args>(args)...);
  }
  llvm::BumpPtrAllocator alloc;
};

} // namespace mlir::triton::applegpu::msl

#endif // MSL_AST_H
