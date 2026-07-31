// PrefetchDotOperand: rotate the smaller dot operand's global load one
// iteration ahead of the K-loop so its latency overlaps the MMA block.
//
// The MSL path stages both dot operands through threadgroup memory, and the
// loop body runs strictly serially: load A/B -> store to threadgroup ->
// barrier -> simdgroup MMAs. Measured on 16384x768x768 (32x64x16, 2 warps) the
// MMAs alone saturate the machine (5.3 TFLOP/s, ~100% of the fp32 peak) while
// the 1.6ms of load traffic around them is pure serial overhead.
//
// Rotating a load means: peel iteration 0's load in front of the loop, carry
// its result as an iter-arg, and inside the body advance the pointer and issue
// the *next* iteration's load before the dot. The emitter then sinks that load
// past the operand-staging barrier (see deferPrefetchLoad in
// EmitMSLControlFlow.cpp), giving it the whole MMA block to retire under.
//
// Only ONE operand is rotated, the one with fewer registers. Rotating both
// keeps every staged element live across the MMAs, and the register pressure
// costs more than the overlap wins (measured +12%; A alone measured -3.8%).

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_PREFETCHDOTOPERAND
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

// A tile-swizzled launch (Triton's GROUP_M grouping) flattens the tile grid
// onto one program-id axis and recovers the 2-D coordinates arithmetically.
// The grouping already orders threadgroups so that the tiles resident together
// share operand rows, which hides the same load latency prefetching would.
// Doing both then costs register pressure with nothing left to cover:
// measured on 16384x768x768 (32x64x16, 2 warps) prefetch moves a plain 2-D
// grid 5.258 -> 5.057 ms but a GROUP_M grid 5.018 -> 5.263 ms.
static bool usesSwizzledGrid(tt::FuncOp fn) {
  bool sawX = false, sawY = false;
  fn.walk([&](tt::GetProgramIdOp pid) {
    if (pid.getAxis() == tt::ProgramIDDim::X)
      sawX = true;
    if (pid.getAxis() == tt::ProgramIDDim::Y)
      sawY = true;
  });
  return sawX && !sawY;
}

// Number of elements a tensor value occupies, used to pick the cheaper operand.
static int64_t tensorElems(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t)
    return 0;
  int64_t n = 1;
  for (int64_t d : t.getShape())
    n *= d;
  return n;
}

// The load feeding a dot operand, looking through layout conversions. Matched
// only when the load is unmasked, has one use, and reads a pointer the loop
// carries as an iter-arg advanced by a loop-invariant step.
struct RotatableLoad {
  tt::LoadOp load;
  BlockArgument ptrArg;
  tt::AddPtrOp advance;
  unsigned ptrIdx;
};

static std::optional<RotatableLoad> matchRotatable(Value operand,
                                                   scf::ForOp forOp) {
  Value v = operand;
  while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = cvt.getSrc();
  auto load = v.getDefiningOp<tt::LoadOp>();
  if (!load || load.getMask() || load.getOther())
    return std::nullopt;
  if (!load.getResult().hasOneUse())
    return std::nullopt;
  auto ptrArg = dyn_cast<BlockArgument>(load.getPtr());
  if (!ptrArg || ptrArg.getOwner() != forOp.getBody())
    return std::nullopt;
  unsigned idx = ptrArg.getArgNumber();
  if (idx == 0)
    return std::nullopt;
  unsigned iterIdx = idx - 1;
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  auto advance = yield.getOperand(iterIdx).getDefiningOp<tt::AddPtrOp>();
  if (!advance || advance.getPtr() != ptrArg)
    return std::nullopt;
  // The step must be computed outside the loop or the peeled copy cannot use
  // it.
  if (advance.getOffset().getParentBlock() == forOp.getBody())
    return std::nullopt;
  for (Operation *u : ptrArg.getUsers())
    if (u != load.getOperation() && u != advance.getOperation())
      return std::nullopt;
  return RotatableLoad{load, ptrArg, advance, iterIdx};
}

class PrefetchDotOperandPass
    : public impl::PrefetchDotOperandBase<PrefetchDotOperandPass> {

  // Rotate `r` one iteration ahead: peel its load before the loop, carry the
  // value, and re-issue the load for the next iteration inside the body.
  void rotate(scf::ForOp forOp, tt::DotOp dot, RotatableLoad r) {
    OpBuilder b(forOp);
    Location loc = r.load.getLoc();

    Value initPtr = forOp.getInitArgs()[r.ptrIdx];
    auto peeled = tt::LoadOp::create(b, loc, initPtr, r.load.getCache(),
                                     r.load.getEvict(), r.load.getIsVolatile());

    SmallVector<Value> newInits(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
    newInits.push_back(peeled.getResult());
    auto newFor = scf::ForOp::create(b, forOp.getLoc(), forOp.getLowerBound(),
                                     forOp.getUpperBound(), forOp.getStep(),
                                     newInits);
    Block *body = forOp.getBody();
    Block *newBody = newFor.getBody();

    // Re-point the old block arguments at the new block's *before* moving the
    // ops across: after the splice the old block is empty and its arguments
    // are no longer a valid iteration source.
    for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i)
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    if (newBody->mightHaveTerminator())
      newBody->getTerminator()->erase();
    newBody->getOperations().splice(newBody->end(), body->getOperations());
    BlockArgument carried = newBody->getArguments().back();

    r.load.getResult().replaceAllUsesWith(carried);

    // Issue the next iteration's load from the advanced pointer immediately
    // before the dot. Anchoring at the old load instead would leave it at the
    // top of the body, in front of the operand staging and its barrier, where
    // it stays serial with the MMAs and buys nothing.
    OpBuilder ib(dot);
    auto advanced = tt::AddPtrOp::create(
        ib, loc, r.advance.getResult().getType(),
        newBody->getArgument(r.ptrIdx + 1), r.advance.getOffset());
    auto next = tt::LoadOp::create(ib, loc, advanced.getResult(),
                                   r.load.getCache(), r.load.getEvict(),
                                   r.load.getIsVolatile());
    r.load.erase();

    auto yield = cast<scf::YieldOp>(newBody->getTerminator());
    SmallVector<Value> newYields(yield.getOperands().begin(),
                                 yield.getOperands().end());
    newYields.push_back(next.getResult());
    OpBuilder yb(yield);
    scf::YieldOp::create(yb, yield.getLoc(), newYields);
    yield.erase();

    for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i)
      forOp.getResult(i).replaceAllUsesWith(newFor.getResult(i));
    forOp.erase();
  }

public:
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](tt::FuncOp fn) {
      if (usesSwizzledGrid(fn))
        return;
      fn.walk([&](scf::ForOp f) { loops.push_back(f); });
    });

    for (scf::ForOp forOp : loops) {
      // Exactly one dot, accumulating into an iter-arg (the fused GEMM shape).
      tt::DotOp dot;
      int nDots = 0;
      for (Operation &o : forOp.getBody()->without_terminator())
        if (auto d = dyn_cast<tt::DotOp>(&o)) {
          dot = d;
          ++nDots;
        }
      if (nDots != 1 || !dot)
        continue;

      auto ra = matchRotatable(dot.getA(), forOp);
      auto rb = matchRotatable(dot.getB(), forOp);
      if (!ra && !rb)
        continue;
      // Rotate the operand with fewer elements: it holds fewer registers live
      // across the MMAs.
      RotatableLoad pick =
          (ra && rb)
              ? (tensorElems(dot.getA()) <= tensorElems(dot.getB()) ? *ra : *rb)
              : (ra ? *ra : *rb);
      rotate(forOp, dot, pick);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPrefetchDotOperandPass() {
  return std::make_unique<PrefetchDotOperandPass>();
}

} // namespace mlir::triton::applegpu
