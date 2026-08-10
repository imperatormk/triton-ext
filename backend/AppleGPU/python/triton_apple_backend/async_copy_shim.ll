; Threadgroup DMA shim for the MSL backend.
;
; air.simdgroup_async_copy_2d is not expressible in MSL source, but a metallib
; may link an AIR object that calls it, and MSL may call that object's function
; through a plain extern "C" declaration. The AIR version metadata below is
; mandatory: without it `metallib` rejects the object with "Module is missing
; AIR version number".
;
; The parameter order of the intrinsic is taken verbatim from Apple's shipping
; MPSMatrix MatrixMultiplyA14X kernel, so it is correct by construction rather
; than by guess.
;
; The element-size operands must be literal constants: passing them as runtime
; values assembles and links cleanly but silently copies garbage (measured --
; a 64x64 fp32 copy came back all-zero). Strides and dimensions are safe to pass
; dynamically. Hence one entry point per element size rather than a single
; parameterised one. Element size is a compile-time property of the dot operand,
; so this costs the emitter nothing.
;
; No target triple is pinned here: metal-as stamps the active toolchain's AIR
; version, and a pinned one is silently overridden, leaving the object at an AIR
; version the kernels it links against may not match.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

%struct._simdgroup_event_t = type opaque

declare %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(i64, i64, i8 addrspace(3)*, i64, i64, <2 x i64>, i8 addrspace(1)*, i64, i64, <2 x i64>, <2 x i64>, i32)
declare void @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t**)

define i64 @__triton_tg_async_copy_begin_4(i8 addrspace(3)* %dst, i64 %dstStride,
                                           i8 addrspace(1)* %src, i64 %srcStride,
                                           i64 %rows, i64 %cols) {
  %d0 = insertelement <2 x i64> undef, i64 %cols, i32 0
  %dims = insertelement <2 x i64> %d0, i64 %rows, i32 1
  %e = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 4, i64 4,
      i8 addrspace(3)* %dst, i64 %dstStride, i64 1, <2 x i64> %dims,
      i8 addrspace(1)* %src, i64 %srcStride, i64 1, <2 x i64> %dims,
      <2 x i64> zeroinitializer, i32 0)
  %h = ptrtoint %struct._simdgroup_event_t* %e to i64
  ret i64 %h
}

define i64 @__triton_tg_async_copy_begin_2(i8 addrspace(3)* %dst, i64 %dstStride,
                                           i8 addrspace(1)* %src, i64 %srcStride,
                                           i64 %rows, i64 %cols) {
  %d0 = insertelement <2 x i64> undef, i64 %cols, i32 0
  %dims = insertelement <2 x i64> %d0, i64 %rows, i32 1
  %e = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 2, i64 2,
      i8 addrspace(3)* %dst, i64 %dstStride, i64 1, <2 x i64> %dims,
      i8 addrspace(1)* %src, i64 %srcStride, i64 1, <2 x i64> %dims,
      <2 x i64> zeroinitializer, i32 0)
  %h = ptrtoint %struct._simdgroup_event_t* %e to i64
  ret i64 %h
}

define i64 @__triton_tg_async_copy_begin_1(i8 addrspace(3)* %dst, i64 %dstStride,
                                           i8 addrspace(1)* %src, i64 %srcStride,
                                           i64 %rows, i64 %cols) {
  %d0 = insertelement <2 x i64> undef, i64 %cols, i32 0
  %dims = insertelement <2 x i64> %d0, i64 %rows, i32 1
  %e = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 1, i64 1,
      i8 addrspace(3)* %dst, i64 %dstStride, i64 1, <2 x i64> %dims,
      i8 addrspace(1)* %src, i64 %srcStride, i64 1, <2 x i64> %dims,
      <2 x i64> zeroinitializer, i32 0)
  %h = ptrtoint %struct._simdgroup_event_t* %e to i64
  ret i64 %h
}

; Transposing variants. The two stride operands after each pointer are the row
; stride and the element stride; swapping them on the source side walks the
; device tile down its columns while the destination stays row-major, so a
; column-major operand (inductor hands B in with strides [1, N]) lands in
; threadgroup memory in the layout simdgroup_load already expects. Verified on
; device: the non-transposing form mis-stages the same tile at element (0,1).
;
; This is what keeps the transpose out of the MMA: no transpose flag has to be
; threaded through simdgroup_load and no fragment addressing changes.

define i64 @__triton_tg_async_copy_begin_4_tr(i8 addrspace(3)* %dst, i64 %dstStride,
                                              i8 addrspace(1)* %src, i64 %srcStride,
                                              i64 %rows, i64 %cols) {
  %d0 = insertelement <2 x i64> undef, i64 %cols, i32 0
  %dims = insertelement <2 x i64> %d0, i64 %rows, i32 1
  %e = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 4, i64 4,
      i8 addrspace(3)* %dst, i64 %dstStride, i64 1, <2 x i64> %dims,
      i8 addrspace(1)* %src, i64 1, i64 %srcStride, <2 x i64> %dims,
      <2 x i64> zeroinitializer, i32 0)
  %h = ptrtoint %struct._simdgroup_event_t* %e to i64
  ret i64 %h
}

define i64 @__triton_tg_async_copy_begin_2_tr(i8 addrspace(3)* %dst, i64 %dstStride,
                                              i8 addrspace(1)* %src, i64 %srcStride,
                                              i64 %rows, i64 %cols) {
  %d0 = insertelement <2 x i64> undef, i64 %cols, i32 0
  %dims = insertelement <2 x i64> %d0, i64 %rows, i32 1
  %e = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 2, i64 2,
      i8 addrspace(3)* %dst, i64 %dstStride, i64 1, <2 x i64> %dims,
      i8 addrspace(1)* %src, i64 1, i64 %srcStride, <2 x i64> %dims,
      <2 x i64> zeroinitializer, i32 0)
  %h = ptrtoint %struct._simdgroup_event_t* %e to i64
  ret i64 %h
}

define i64 @__triton_tg_async_copy_begin_1_tr(i8 addrspace(3)* %dst, i64 %dstStride,
                                              i8 addrspace(1)* %src, i64 %srcStride,
                                              i64 %rows, i64 %cols) {
  %d0 = insertelement <2 x i64> undef, i64 %cols, i32 0
  %dims = insertelement <2 x i64> %d0, i64 %rows, i32 1
  %e = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 1, i64 1,
      i8 addrspace(3)* %dst, i64 %dstStride, i64 1, <2 x i64> %dims,
      i8 addrspace(1)* %src, i64 1, i64 %srcStride, <2 x i64> %dims,
      <2 x i64> zeroinitializer, i32 0)
  %h = ptrtoint %struct._simdgroup_event_t* %e to i64
  ret i64 %h
}

; The event token crosses the ABI as an integer: MSL requires an explicit
; address space on every pointer type, so it cannot be declared as an opaque
; pointer on the MSL side.
define void @__triton_tg_async_copy_wait(i64 %h) {
  %e = inttoptr i64 %h to %struct._simdgroup_event_t*
  %slot = alloca %struct._simdgroup_event_t*
  store %struct._simdgroup_event_t* %e, %struct._simdgroup_event_t** %slot
  call void @air.wait_simdgroup_events(i32 1, %struct._simdgroup_event_t** %slot)
  ret void
}

define void @__triton_tg_async_copy_wait2(i64 %h0, i64 %h1) {
  %e0 = inttoptr i64 %h0 to %struct._simdgroup_event_t*
  %e1 = inttoptr i64 %h1 to %struct._simdgroup_event_t*
  %slots = alloca [2 x %struct._simdgroup_event_t*]
  %p0 = getelementptr [2 x %struct._simdgroup_event_t*], [2 x %struct._simdgroup_event_t*]* %slots, i64 0, i64 0
  %p1 = getelementptr [2 x %struct._simdgroup_event_t*], [2 x %struct._simdgroup_event_t*]* %slots, i64 0, i64 1
  store %struct._simdgroup_event_t* %e0, %struct._simdgroup_event_t** %p0
  store %struct._simdgroup_event_t* %e1, %struct._simdgroup_event_t** %p1
  call void @air.wait_simdgroup_events(i32 2, %struct._simdgroup_event_t** %p0)
  ret void
}

!llvm.module.flags = !{!0, !1}
!air.version = !{!2}
!air.language_version = !{!3}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 9, i32 0}
!3 = !{!"Metal", i32 3, i32 2, i32 0}
