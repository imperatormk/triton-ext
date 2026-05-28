; Phase 1 smoke test: the Metal/AIR target loads and llc runs its (currently
; empty) pass pipeline, printing the module back out.
; RUN: %metal-llc -mtriple=air -filetype=asm %s -o - | FileCheck %s

; CHECK: define void @kernel
define void @kernel(ptr %out) {
entry:
  store i32 0, ptr %out
  ret void
}
