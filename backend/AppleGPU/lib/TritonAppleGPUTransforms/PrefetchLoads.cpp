// A K loop's dot operands are loaded, staged and multiplied in one iteration,
// so every iteration waits on its own loads. Loading them num_stages - 1
// iterations ahead into loop-carried registers hides that wait behind the dots
// of the iterations in between. The expansion itself is upstream's pipeliner;
// this pass only decides which ops run ahead.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
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

// The kernel arguments `value` can point into, or nullopt when part of it
// does not come from one.
std::optional<DenseSet<Value>> argumentsBehind(Value value) {
  DenseSet<Value> args, seen;
  SmallVector<Value> work{value};
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      Operation *owner = arg.getOwner()->getParentOp();
      if (isa<tt::FuncOp>(owner)) {
        args.insert(arg);
        continue;
      }
      auto loop = dyn_cast<scf::ForOp>(owner);
      if (!loop || arg == loop.getInductionVar())
        return std::nullopt;
      work.push_back(loop.getInitArgs()[arg.getArgNumber() - 1]);
      work.push_back(loop.getYieldedValues()[arg.getArgNumber() - 1]);
      continue;
    }
    Operation *def = v.getDefiningOp();
    const unsigned i = cast<OpResult>(v).getResultNumber();
    if (auto loop = dyn_cast<scf::ForOp>(def)) {
      work.push_back(loop.getInitArgs()[i]);
      work.push_back(loop.getYieldedValues()[i]);
    } else if (auto branch = dyn_cast<scf::IfOp>(def)) {
      work.push_back(branch.thenYield().getOperand(i));
      work.push_back(branch.elseYield().getOperand(i));
    } else if (!isa<ub::PoisonOp>(def)) {
      const size_t before = work.size();
      if (isPure(def))
        for (Value operand : def->getOperands())
          if (isa<tt::PointerType>(getElementTypeOrSelf(operand)))
            work.push_back(operand);
      if (work.size() == before)
        return std::nullopt;
    }
  }
  return args;
}

// One memory effect's resource and the kernel arguments its address can come
// from (nullopt when not known).
struct Access {
  SideEffects::Resource *resource;
  std::optional<DenseSet<Value>> args;
};

// Every access of `kind` in `op` and the ops nested in it; nullopt when some
// op's effects are not known.
template <typename Kind>
std::optional<SmallVector<Access>> accessesOf(Operation *op) {
  SmallVector<Access> accesses;
  const bool unknown =
      op->walk([&](Operation *nested) {
          auto effects = dyn_cast<MemoryEffectOpInterface>(nested);
          if (!effects)
            return nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>()
                       ? WalkResult::advance()
                       : WalkResult::interrupt();
          SmallVector<MemoryEffects::EffectInstance> instances;
          effects.getEffects(instances);
          for (const MemoryEffects::EffectInstance &e : instances) {
            if (!isa<Kind>(e.getEffect()))
              continue;
            Value target = e.getValue();
            accesses.push_back({e.getResource(), target
                                                     ? argumentsBehind(target)
                                                     : std::nullopt});
          }
          return WalkResult::advance();
        }).wasInterrupted();
  if (unknown)
    return std::nullopt;
  return accesses;
}

// Kernel arguments are taken not to alias one another.
bool mayAlias(const Access &a, const Access &b) {
  auto *any = SideEffects::DefaultResource::get();
  if (a.resource != b.resource && a.resource != any && b.resource != any)
    return false;
  if (!a.args || !b.args)
    return true;
  return llvm::any_of(*a.args, [&](Value v) { return b.args->contains(v); });
}

// Whether a write anywhere in `loop` may touch what `ahead` reads: those
// reads would pass it.
bool aheadReadsMayBeWritten(scf::ForOp loop,
                            const llvm::SetVector<Operation *> &ahead) {
  SmallVector<Access> reads;
  for (Operation *op : ahead) {
    std::optional<SmallVector<Access>> r = accessesOf<MemoryEffects::Read>(op);
    if (!r)
      return true;
    reads.append(*r);
  }
  for (Operation &op : loop.getBody()->without_terminator()) {
    std::optional<SmallVector<Access>> writes =
        accessesOf<MemoryEffects::Write>(&op);
    if (!writes)
      return true;
    for (const Access &w : *writes)
      for (const Access &r : reads)
        if (mayAlias(w, r))
          return true;
  }
  return false;
}

// The ops that must run ahead with `loads`: everything in the body they
// depend on, including what computes the loop-carried values they read.
// Empty when that would carry a dot or a write ahead.
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
    std::optional<SmallVector<Access>> writes =
        accessesOf<MemoryEffects::Write>(op);
    if (!writes || !writes->empty() || op->walk([](tt::DotOp) {
                                           return WalkResult::interrupt();
                                         }).wasInterrupted())
      return {};
    llvm::SetVector<Value> used(op->getOperands().begin(),
                                op->getOperands().end());
    getUsedValuesDefinedAbove(op->getRegions(), used);
    for (Value v : used) {
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
      if (stages < 2 || tt::isOuterLoop(loop))
        continue;
      const SmallVector<Operation *> loads = dotOperandLoads(loop);
      if (loads.empty())
        continue;
      const llvm::SetVector<Operation *> ahead = aheadSet(loop, loads);
      if (ahead.empty() || aheadReadsMayBeWritten(loop, ahead))
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
