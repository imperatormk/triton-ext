// UnpipelineDotA: keep the A operand out of the software pipeline.
//
// The pipeliner turns every latency-carrying load into an allocation plus an
// async copy (createAsyncCopy in LowerLoops.cpp runs unconditionally on
// everything that reached asyncLoads). For a GEMM that means both operands make
// a device->threadgroup round trip, which is exactly the trip dotADirect
// removes on the default path by letting simdgroup_load read A straight out of
// device memory at its runtime stride.
//
// A load only reaches asyncLoads when its latency attribute puts its definition
// and its first use in different stages, so clearing the attribute on A's load
// leaves A as a plain load feeding the dot -- the shape dotADirect matches --
// while B keeps its latency and still rotates.

#include "TritonAppleGPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;

namespace mlir::triton::applegpu {

#define GEN_PASS_DEF_UNPIPELINEDOTA
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace {

// The load behind a dot operand, looking through the layout converts and the
// local_alloc/local_load pair ShareDotOperands may already have inserted.
tt::LoadOp operandLoad(Value operand) {
  Value v = operand;
  for (;;) {
    if (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>()) {
      v = cvt.getSrc();
      continue;
    }
    if (auto ll = v.getDefiningOp<ttg::LocalLoadOp>()) {
      v = ll.getSrc();
      continue;
    }
    if (auto la = v.getDefiningOp<ttg::LocalAllocOp>()) {
      if (!la.getSrc())
        return nullptr;
      v = la.getSrc();
      continue;
    }
    break;
  }
  return v.getDefiningOp<tt::LoadOp>();
}

struct UnpipelineDotAPass
    : public impl::UnpipelineDotABase<UnpipelineDotAPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto helper = tt::TritonDialect::getLoaded(mod)->getLatencyAttrHelper();
    mod.walk([&](scf::ForOp forOp) {
      forOp.walk([&](tt::DotOp dot) {
        if (dot->getParentOp() != forOp.getOperation())
          return;
        tt::LoadOp a = operandLoad(dot.getA());
        tt::LoadOp b = operandLoad(dot.getB());
        // Dropping A's latency only pays for itself while B still rotates; with
        // neither operand pipelined the loop is the unpipelined one already.
        if (!a || !b || a == b)
          return;
        if (!helper.getAttr(b))
          return;
        helper.removeAttr(a);
      });
    });
  }
};

} // namespace

std::unique_ptr<Pass> createUnpipelineDotAPass() {
  return std::make_unique<UnpipelineDotAPass>();
}

} // namespace mlir::triton::applegpu
