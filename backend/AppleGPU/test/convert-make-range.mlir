// RUN: triton-opt %s --convert-triton-apple-gpu-to-llvm | FileCheck %s

// Regression test for the WarpId lowering. The upstream MakeRange pattern
// emits ttg.warp_id; without an Apple lowering the conversion rolls back
// and falsely reports "tt.make_range failed to legalize".

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "mps:apple_m", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @make_range_kernel
  // CHECK: llvm.call @air.thread_position_in_threadgroup
  tt.func public @make_range_kernel() attributes {noinline = false} {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    tt.return
  }
}
