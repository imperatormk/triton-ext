//===- IntegerLegalize.h - AGX-JIT integer/intrinsic legalization *- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_INTEGERLEGALIZE_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_INTEGERLEGALIZE_H

#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

// Drop lifetime.start/end intrinsics (and dangling decls); Apple's metallib
// carries none and the shared void(ptr) signature collides with typed pointers.
void stripLifetimeIntrinsics(Module &M);

// Expand >64-bit integer arithmetic chains into (lo, hi) i64 limb pairs; the
// AGX JIT cannot legalize any iN > 64. Unsupported wide ops fail loud.
void expandWideIntegers(Module &M);

// AIR v1 bitcode has no freeze opcode; replace freeze with its operand.
void lowerFreezeInsts(Module &M);

// Strip 'disjoint' flag from 'or' instructions (Metal v1 bitcode).
void stripDisjointFlags(Module &M);

// `zext nneg` equals `sext`; emit the sext form (the AGX JIT mis-legalizes
// zext-fed 64-bit multiplies into s65).
void canonicalizeNNegZExt(Module &M);

// Expand llvm.scmp/llvm.ucmp inline; the AIR backend has no lowering for them.
void lowerCmpIntrinsics(Module &M);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_INTEGERLEGALIZE_H
