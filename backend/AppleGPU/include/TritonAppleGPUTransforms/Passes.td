#ifndef TRITON_APPLE_GPU_TRANSFORMS_PASSES
#define TRITON_APPLE_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AccelerateAppleMatmul : Pass<"triton-apple-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "Replace blocked dot ops with Apple simdgroup MMA encoding";

  let description = [{
    Rewrites tt.dot operations to use AppleMmaEncodingAttr, enabling
    simdgroup_multiply_accumulate code generation on Apple GPUs.

    For each tt.dot with BlockedEncoding output:
      1. Select warpsPerCTA based on output shape and numWarps budget
      2. Create AppleMmaEncodingAttr with selected warpsPerCTA
      3. Insert ConvertLayoutOp on A, B, C operands
      4. Create new dot with AppleMma output encoding
      5. Convert output back to original encoding for downstream users
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
    The AppleMma C result and the natural coalesced store layout use different
    warp tilings, so the #mma -> #blocked C-output convert_layout is cross-warp
    and spills through threadgroup memory + barriers (the dominant epilogue cost
    for large GEMM tiles). For stores whose value is an #mma -> #blocked convert,
    this pass re-lays the store (value, pointer, and mask address math) into the
    one blocked layout that shares the mma warp tiling and maps every C element
    to a lane in the same simdgroup, making the convert a within-simdgroup
    register+shuffle move with no shared memory. Stores whose address math uses
    ops outside a fixed relayable allowlist are left untouched (the threadgroup
    convert path stays as the always-correct fallback).
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::applegpu::TritonAppleGPUDialect"
  ];
}

def PrefetchDotOperand : Pass<"triton-apple-prefetch-dot-operand", "mlir::ModuleOp"> {
  let summary = "Rotate the smaller dot operand's load ahead of the K-loop";

  let description = [{
    The MSL path stages both dot operands through threadgroup memory and runs
    the loop body serially: load -> threadgroup store -> barrier -> MMAs. The
    MMAs alone already saturate the fp32 issue rate, so the surrounding load
    traffic is pure serial overhead.

    This peels iteration 0's load for one operand in front of the loop, carries
    the loaded value as an iter-arg, and issues the next iteration's load
    before the dot so it retires underneath the MMA block.

    Only the operand with fewer elements is rotated. Rotating both keeps every
    staged element live across the MMAs and the added register pressure costs
    more than the overlap wins.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::scf::SCFDialect"
  ];
}

def SimplifyGatherLayout : Pass<"triton-apple-simplify-gather", "mlir::ModuleOp"> {
  let summary = "Strip efficient_layout from large gather ops to avoid IR explosion";

  let description = [{
    The upstream OptimizeThreadLocality pass marks gather ops with
    efficient_layout, forcing warp-local shuffle lowering. For large tensors
    this generates thousands of unrolled shuffle+select instructions that
    exceed the Metal GPU JIT compiler's instruction limit. This pass clears
    the flag so the gather falls back to shared memory, and the subsequent
    RemoveLayoutConversions pass eliminates the now-unnecessary convert_layouts.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif // TRITON_APPLE_GPU_TRANSFORMS_PASSES
