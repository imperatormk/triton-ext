// A K loop's dot operands are loaded, staged and multiplied in one iteration,
// so every iteration waits on its own loads. Loading them num_stages - 1
// iterations ahead into loop-carried registers hides that wait behind the dots
// of the iterations in between. The expansion itself is upstream's pipeliner;
// this pass only decides which ops run ahead.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SetVector.h"

namespace tt = mlir::triton;
using namespace mlir;

namespace mlir::triton::applegpu {

#define GEN_PASS_DECL_PREFETCHLOADS
#define GEN_PASS_DEF_PREFETCHLOADS
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

// Loads in `loop`'s body whose value reaches a dot operand there through
// side-effect-free ops.
SmallVector<Operation *> dotOperandLoads(scf::ForOp loop) {
  SmallVector<Operation *> loads;
  Block *body = loop.getBody();
  for (auto dot : body->getOps<tt::DotOp>())
    for (Value operand : {dot.getA(), dot.getB()}) {
      SmallVector<Value> work{operand};
      DenseSet<Operation *> seen;
      while (!work.empty()) {
        Operation *def = work.pop_back_val().getDefiningOp();
        if (!def || def->getBlock() != body || !seen.insert(def).second)
          continue;
        if (isa<tt::LoadOp>(def)) {
          loads.push_back(def);
          continue;
        }
        if (!isMemoryEffectFree(def))
          continue;
        for (Value v : def->getOperands())
          work.push_back(v);
      }
    }
  return loads;
}

// Whether `loop` writes memory anywhere in its body. Loads moved ahead would
// pass those writes, and nothing proves the two never alias.
bool writesMemory(scf::ForOp loop) {
  return loop.getBody()
      ->walk([](Operation *op) {
        auto effects = dyn_cast<MemoryEffectOpInterface>(op);
        const bool writes =
            effects ? effects.hasEffect<MemoryEffects::Write>()
                    : !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
        return writes ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

// The ops that must run ahead with `loads`: everything in the body they
// depend on, including what computes the loop-carried values they read.
// Empty when that would carry a dot or any other side effect ahead.
llvm::SetVector<Operation *> aheadSet(scf::ForOp loop,
                                      ArrayRef<Operation *> loads) {
  Block *body = loop.getBody();
  auto yield = cast<scf::YieldOp>(body->getTerminator());
  llvm::SetVector<Operation *> ahead;
  SmallVector<Operation *> work(loads.begin(), loads.end());
  while (!work.empty()) {
    Operation *op = work.pop_back_val();
    if (!ahead.insert(op))
      continue;
    if (op->getNumRegions() != 0 || isa<tt::DotOp>(op) ||
        (!isa<tt::LoadOp>(op) && !isMemoryEffectFree(op)))
      return {};
    for (Value v : op->getOperands()) {
      if (Operation *def = v.getDefiningOp()) {
        if (def->getBlock() == body)
          work.push_back(def);
        continue;
      }
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg || arg.getOwner() != body || arg == loop.getInductionVar())
        continue;
      Value next = yield.getOperand(arg.getArgNumber() - 1);
      if (Operation *def = next.getDefiningOp(); def && def->getBlock() == body)
        work.push_back(def);
    }
  }
  return ahead;
}

struct PrefetchLoadsPass : public impl::PrefetchLoadsBase<PrefetchLoadsPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops) {
      // An outer loop would carry its tiles across the whole inner loop.
      const int stages = tt::getNumStagesOrDefault(loop, numStages);
      if (stages < 2 || tt::isOuterLoop(loop) || writesMemory(loop))
        continue;
      const SmallVector<Operation *> loads = dotOperandLoads(loop);
      if (loads.empty())
        continue;
      const llvm::SetVector<Operation *> ahead = aheadSet(loop, loads);
      if (ahead.empty())
        continue;

      // Ahead ops first, so the loads issue before this iteration's dots.
      std::vector<std::pair<Operation *, unsigned>> schedule;
      const unsigned last = stages - 1;
      for (Operation &op : loop.getBody()->without_terminator())
        if (ahead.contains(&op))
          schedule.emplace_back(&op, 0);
      for (Operation &op : loop.getBody()->without_terminator())
        if (!ahead.contains(&op))
          schedule.emplace_back(&op, last);

      tt::PipeliningOption options;
      options.supportDynamicLoops = true;
      options.peelEpilogue = true;
      options.predicateFn = tt::wrapInMaskOp;
      options.getScheduleFn =
          [&](scf::ForOp, std::vector<std::pair<Operation *, unsigned>> &s) {
            s = schedule;
          };
      IRRewriter rewriter(loop);
      (void)tt::pipelineForLoop(rewriter, loop, options);
    }
    tt::resolveMaskOp(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> createPrefetchLoadsPass(int numStages) {
  PrefetchLoadsOptions options;
  options.numStages = numStages;
  return std::make_unique<PrefetchLoadsPass>(options);
}

} // namespace mlir::triton::applegpu
