// A rank-1 tile is a single row, so this encoding's phase is always zero and
// the offset stays unpermuted; `test_swizzle.cpp` covers the permutation.
//
// RUN: msl_driver.py %s | %filecheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: kernel void triton_rt
// CHECK: threadgroup float [[MD:md[0-9]+]][128];
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK: [[MD]][{{[a-z0-9_]+}}] =
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK: = [[MD]][{{[a-z0-9_]+}}];
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "mps:apple_m", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rt(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_0 = tt.addptr %x, %off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_1 = tt.load %x_0 : tensor<128x!tt.ptr<f32>, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttg.local_store %x_1, %smem : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    %0 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %2 = ttg.local_load %smem : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    tt.store %1, %2 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
