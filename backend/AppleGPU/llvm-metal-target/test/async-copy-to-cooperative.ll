; A kernel that reads a threadgroup arena with air.simdgroup_matrix_8x8_load
; while that same arena is written by air.simdgroup_async_copy_2d fails the AGX
; PSO compiler with the opaque "Failed to materializeAll".  MetalAsyncCopyToCo-
; operative lowers the async copy to a cooperative threadgroup copy and erases
; the toxic async-copy / wait-event symbols.  See METALLC_NOTES.md.
; RUN: %metal-llc -mtriple=air -filetype=asm %s -o - | FileCheck %s

; CHECK-NOT: air.simdgroup_async_copy_2d
; CHECK-NOT: air.wait_simdgroup_events
; The cooperative copy reads device src and stores into the threadgroup arena.
; CHECK: store float
; The simdgroup-matrix load survives.
; CHECK: air.simdgroup_matrix_8x8_load

@smem = internal addrspace(3) global [256 x i8] undef, align 16

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)
declare ptr addrspace(3) @air.simdgroup_async_copy_2d.p3i8.p1i8(i64, i64, ptr addrspace(3), i64, i64, <2 x i64>, ptr addrspace(1), i64, i64, <2 x i64>, <2 x i64>, i32)
declare void @air.wait_simdgroup_events(i32, ptr)
declare [3 x i32] @air.thread_position_in_threadgroup()

define void @kernel(ptr addrspace(1) %in) {
entry:
  %ev = alloca ptr addrspace(3), align 8
  ; async copy device -> threadgroup arena (8x8 f32 tile, 32-byte pitch)
  %e = call ptr addrspace(3) @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 1, i64 1, ptr addrspace(3) @smem, i64 32, i64 1,
      <2 x i64> <i64 32, i64 8>, ptr addrspace(1) %in, i64 32, i64 1,
      <2 x i64> <i64 32, i64 8>, <2 x i64> zeroinitializer, i32 0)
  store ptr addrspace(3) %e, ptr %ev, align 8
  call void @air.wait_simdgroup_events(i32 1, ptr %ev)
  ; MMA reads the same arena -> would trigger materializeAll without the pass.
  %m = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(
      ptr addrspace(3) @smem, <2 x i64> <i64 8, i64 8>,
      <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
      <64 x float> %m, ptr addrspace(3) @smem, <2 x i64> <i64 8, i64 8>,
      <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{}
