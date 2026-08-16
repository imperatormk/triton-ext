// MSLFuncBudget.h - size measurement and shrink policy for one emitted function.
//
// Apple's Metal compiler dies with std::bad_alloc under PromoteMemToReg/SROA on
// an oversized function. Measure, decide, mitigate -- all of it lives here.
#ifndef MSL_FUNC_BUDGET_H
#define MSL_FUNC_BUDGET_H

#include "MSLAst.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::applegpu {

struct FuncSize {
  int64_t stmts = 0;
  int64_t decls = 0;
  int64_t fragDecls = 0;
  int64_t branches = 0;
  int64_t mma = 0;
  int64_t barriers = 0;
  int64_t loops = 0;

  // Declarations are what the optimiser allocates state per, and measured over
  // a whole crashing run they rank the ten largest kernels exactly, where
  // fragment counts rank two of ten.
  int64_t optimiserLoad() const { return decls; }
};

// Counted off the built AST: the per-dot counts cannot be predicted from the IR
// because operand fragments are re-declared per cache window, and the windows
// are chosen during emission.
FuncSize measureFunc(const msl::Block &body);

struct ShrinkPlan {
  bool rollKSteps = false;
  bool fuseGuards = false;

  bool any() const { return rollKSteps || fuseGuards; }
};

ShrinkPlan planShrink(const FuncSize &s);

// rollKSteps is a re-walk, so the emitter drives that one; this applies the
// post-emission mitigations.
void shrink(msl::KernelFn *fn, msl::MSLContext &ctx, const ShrinkPlan &plan);

void debugBudget(llvm::StringRef fn, llvm::StringRef stage, const FuncSize &s,
                 const ShrinkPlan &plan);

} // namespace mlir::triton::applegpu

#endif // MSL_FUNC_BUDGET_H
