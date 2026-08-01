// MSLPrinter.h - walks the msl:: AST and emits MSL text.
//
// THE ONLY place `{`/`}` and indentation are emitted. Owns the barrier
// peephole (moved out of the emitter).
#ifndef MSL_PRINTER_H
#define MSL_PRINTER_H

#include "MSLAst.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::applegpu::msl {

// Which preamble helper groups a module actually needs. Computed by an IR
// pre-scan (MSLEmitter::emit) so only referenced helpers are emitted.
// Dependency: the packed16 atomics call tt_rtne_int_half/bfloat, so requesting
// an fp16/bf16 atomic implies the matching narrowing helper.
struct HelperSet {
  bool erf = false;
  bool rtneHalf = false;
  bool rtneBfloat = false;
  bool rtzHalf = false;
  bool rtzBfloat = false;
  bool rtneIntHalf = false;
  bool rtneIntBfloat = false;
  bool atomicF32 = false;
  bool atomicPacked16 = false;
  bool fp8 = false;
  bool tgAsyncCopy = false;
};

class MSLPrinter {
public:
  explicit MSLPrinter(llvm::raw_ostream &os) : os(os) {}

  void printPreamble(const HelperSet &h = HelperSet{});
  void printNarrowingHelpers(const HelperSet &h);
  void printFp8Helpers();
  void printAtomicHelpers(const HelperSet &h);
  void printType(const Type *t);
  void printExpr(const Expr *e);
  void printStmt(const Stmt *s);
  void printBlock(const Block &b);
  void printParam(const Param &p);
  void printAttr(const Attr *a);

  // A top-level function is itself a Scope stmt (KernelFn/DeviceFn).
  void print(const Stmt *topLevel) { printStmt(topLevel); }
  // Device-function prototype (no body): `<ret> <name>(<params>);`
  void printProto(const DeviceFn *fn);

private:
  llvm::raw_ostream &os;
  unsigned indent = 0;

  // Barrier peephole: a BarrierStmt is held pending rather than printed, so
  // adjacent barriers collapse into one keeping the stronger scope.
  // flushBarrier MUST run before any non-barrier output and at every scope
  // boundary.
  bool barrierPending = false;
  bool barrierPendingDevice = false;
  void flushBarrier();

  llvm::raw_ostream &ind() {
    for (unsigned i = 0; i < indent * 4; ++i)
      os << ' ';
    return os;
  }

  void
  printStmtInline(const Stmt *s); // decl/assign without newline (for-header)
};

} // namespace mlir::triton::applegpu::msl

#endif // MSL_PRINTER_H
