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

def WidenPipelinedStaging : Pass<"triton-apple-widen-staging", "mlir::ModuleOp"> {
  let summary = "Grow num_stages=2 dot staging from 1 to 2 rotating SMEM slots";

  let description = [{
    The upstream pipeliner allocates (num_stages - 1) staging slots, so at
    num_stages=2 every K-step's async copy overwrites the strip the in-flight
    dot still reads, and the DotOp lowering must keep a per-dot post-load
    threadgroup barrier. This pass doubles such single-slot staging allocs to
    2 rotating slots and bumps the pipeliner's already-present rotation
    counters' wrap bound from 1 to 2, making the IR shape identical to
    num_stages=3 so the barrier elision fires. Bails when the doubled staging
    would exceed the 32KB threadgroup budget.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def DemoteWideAccumulator : Pass<"triton-apple-demote-wide-accumulator", "mlir::ModuleOp"> {
  let summary = "Scalarize a wide loop-carried literal-struct phi to per-element allocas";

  let description = [{
    A rolled scf.for whose iter_arg is a large per-thread tensor slice lowers to
    a loop with a single literal-struct block argument (the phi),
    !llvm.struct<(f32 x N)> for large N, rebuilt every iteration via N
    insertvalue and read via N extractvalue. LLVM's IPSCCP (part of
    optimize_module(O3)) non-terminates on such a wide aggregate phi.

    This pass demotes the wide struct phi to N scalar allocas the loop
    load/modify/stores in place: the field value stored on the latch edge is
    pulled directly out of the producing insertvalue chain, so no wide aggregate
    SSA value is ever materialized. mem2reg/SROA then handle the scalar slots
    trivially and IPSCCP sees N independent scalar lattices instead of one
    N-field aggregate. Only fires above a field-count threshold, so ordinary
    small loop-carried tensors are untouched.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif // TRITON_APPLE_GPU_TRANSFORMS_PASSES
