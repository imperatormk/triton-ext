// ShareDotOperands: route K-loop dot operands through shared memory so
// Triton's software pipeliner will multi-buffer them.
//
// The pipeliner only multi-buffers a load whose single user is a
// ttg.local_alloc (mustLoadToRegisters in LowerLoops.cpp); everything else is
// left in registers and never rotated. The MSL path kept dot operands in
// blocked layouts, so no load qualified and num_stages was silently inert --
// ns=2/3/4 produced byte-identical code while still minting three cache
// entries and three autotune candidates.
//
// Rewriting `load -> dot` into `load -> local_alloc -> local_load -> dot` is
// the shape the pipeliner recognises. The allocation is unswizzled
// (vec/perPhase/maxPhase = 1) because the emitter stages operands as plain
// padded row-major tiles and reads them back with simdgroup_load at that
// pitch; a swizzle here would have to be undone there.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_SHAREDOTOPERANDS
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

// The load feeding a dot operand, looking through layout converts. Only an
// unmasked load inside `forOp` qualifies: a masked load cannot lower to an
// async copy, and one outside the loop has nothing to pipeline against.
tt::LoadOp operandLoad(Value operand, scf::ForOp forOp) {
  Value v = operand;
  while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = cvt.getSrc();
  auto load = v.getDefiningOp<tt::LoadOp>();
  if (!load || load.getMask() || load.getOther())
    return nullptr;
  if (load->getParentOp() != forOp.getOperation())
    return nullptr;
  // local_alloc consumes the load's result, so anything else reading it would
  // observe a value that no longer exists in registers.
  if (!load.getResult().hasOneUse())
    return nullptr;
  return load;
}

// `load -> local_alloc -> local_load` in place of the raw load.
void routeThroughShared(tt::LoadOp load) {
  auto ty = cast<RankedTensorType>(load.getResult().getType());
  MLIRContext *ctx = ty.getContext();

  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
  if (!blocked)
    return;

  // The emitter addresses every threadgroup tile row-major (flatTileOffset
  // folds the out-dims in shape order and ignores the shared encoding's
  // `order`), so the allocation must say row-major too. Inheriting the
  // blocked layout's order instead makes a column-major operand be written
  // one way and read another -- measured as wrong answers on every dtype with
  // a transposed B, while row-major B stayed correct.
  SmallVector<unsigned> order;
  for (int d = (int)ty.getRank() - 1; d >= 0; --d)
    order.push_back((unsigned)d);
  (void)blocked;
  auto ctaLayout = ttg::getCGALayout(ty.getEncoding());
  auto shared = ttg::SwizzledSharedEncodingAttr::get(ctx, /*vec=*/1,
                                                     /*perPhase=*/1,
                                                     /*maxPhase=*/1, order,
                                                     ctaLayout);
  auto space = ttg::SharedMemorySpaceAttr::get(ctx);
  auto memTy = ttg::MemDescType::get(ty.getShape(), ty.getElementType(), shared,
                                     space);

  OpBuilder b(load);
  b.setInsertionPointAfter(load);
  Location loc = load.getLoc();
  Value alloc =
      ttg::LocalAllocOp::create(b, loc, memTy, load.getResult()).getResult();
  Value back = ttg::LocalLoadOp::create(b, loc, ty, alloc).getResult();
  load.getResult().replaceAllUsesExcept(back, alloc.getDefiningOp());
}

struct ShareDotOperandsPass
    : public impl::ShareDotOperandsBase<ShareDotOperandsPass> {
  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp forOp) {
      SmallVector<tt::LoadOp> todo;
      forOp.walk([&](tt::DotOp dot) {
        if (dot->getParentOp() != forOp.getOperation())
          return;
        for (Value operand : {dot.getA(), dot.getB()})
          if (tt::LoadOp ld = operandLoad(operand, forOp))
            if (!llvm::is_contained(todo, ld))
              todo.push_back(ld);
      });
      for (tt::LoadOp ld : todo)
        routeThroughShared(ld);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createShareDotOperandsPass() {
  return std::make_unique<ShareDotOperandsPass>();
}

} // namespace mlir::triton::applegpu
