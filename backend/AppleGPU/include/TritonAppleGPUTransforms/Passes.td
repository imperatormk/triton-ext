#ifndef TRITON_APPLE_GPU_TRANSFORMS_PASSES
#define TRITON_APPLE_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AccelerateAppleMatmul : Pass<"triton-apple-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "Replace blocked dot ops with Apple simdgroup MMA encoding";

  let description = [{
    Rewrites `tt.dot` to AppleMmaEncodingAttr so it lowers to
    `simdgroup_multiply_accumulate`.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

def StoreShuffleLayout : Pass<"triton-apple-store-shuffle-layout", "mlir::ModuleOp"> {
  let summary = "Re-lay MMA epilogue stores into the simdgroup-shuffle layout";

  let description = [{
    The mma C result and the coalesced store layout use different warp tilings,
    so the `#mma -> #blocked` convert crosses warps and spills through
    threadgroup memory. This re-lays the store into the blocked layout that
    shares the mma warp tiling, making the convert a within-simdgroup shuffle.

    A store whose address math uses an op outside the relayable allowlist keeps
    the threadgroup convert.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

def AtomicLaneLayout : Pass<"triton-apple-atomic-lane-layout", "mlir::ModuleOp"> {
  let summary = "Give each lane one element of a tensor atomic's fastest axis";

  let description = [{
    Coalesce sizes a thread's elements for vector memory ops. Metal has no
    vector atomics, so with several elements per thread one atomic
    instruction strides across cache lines. Converts in and out let
    remove_layout_conversions push the lane layout onto the producers.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def MaskSelectArmLoads : Pass<"triton-apple-mask-select-arm-loads", "mlir::ModuleOp"> {
  let summary = "Mask loads that only feed one arm of a select with its condition";

  let description = [{
    Inductor lowers select_backward, stack and similar ops to a sum of
    `where(c_i, f_i(load_i), 0)`, loading every input for every element. A
    load whose value only reaches one arm of a select gets that arm's
    condition in its mask, so lanes that discard the arm skip the access.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect"
  ];
}

def EmitMSL : Pass<"triton-apple-emit-msl", "mlir::ModuleOp"> {
  let summary = "Emit Metal Shading Language from TTGIR";

  let description = [{
    Terminal codegen. Writes MSL text for the module to the path given in the
    pass options.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

#endif // TRITON_APPLE_GPU_TRANSFORMS_PASSES
