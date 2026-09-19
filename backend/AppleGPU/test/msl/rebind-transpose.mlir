// A transpose moves elements between threads, so the tile goes through
// threadgroup memory: scatter by the source's coordinates, gather by the
// result's. The convert the frontend pairs with it keeps every element in its
// own lane, so it emits nothing and one buffer serves both.
//
// RUN: msl_driver.py %s | %filecheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "mps:apple_m", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_rt(%x_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
    %m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m_e = tt.expand_dims %m {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %n_e = tt.expand_dims %n {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %stride = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %row = arith.muli %m_e, %stride : tensor<32x1xi32, #blocked>
    %row_b = tt.broadcast %row : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %col_b = tt.broadcast %n_e : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %off = arith.addi %row_b, %col_b : tensor<32x32xi32, #blocked>

    %px = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %qx = tt.addptr %px, %off : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %tile = tt.load %qx : tensor<32x32x!tt.ptr<f32>, #blocked>

    %t = tt.trans %tile {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    %c = ttg.convert_layout %t : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #blocked>

    %po = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %qo = tt.addptr %po, %off : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    tt.store %qo, %c : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: kernel void triton_rt
// CHECK: threadgroup float [[RD:rd[0-9]+]][1024];
// CHECK: threadgroup_barrier
// CHECK: [[RD]][
// CHECK: threadgroup_barrier
// CHECK: = *(threadgroup float4 *)&[[RD]][
// CHECK: threadgroup_barrier
// CHECK-NOT: threadgroup float rd
