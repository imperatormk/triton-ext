//===- LowerVectorSelect.h - Branchless vector-select lowering --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_LOWERVECTORSELECT_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_LOWERVECTORSELECT_H

#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

// Lowers vector-condition selects (VSELECT) into a branchless per-lane bitmask
// blend, staying fully vectorized. The AGX JIT rejects `select <N x i1>`, so
// each such select is rewritten as `(a & mask) | (b & ~mask)` where the mask is
// `sext <N x i1> cond` over the bit-reinterpreted operands.
void lowerVectorSelects(Module &M);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_LOWERVECTORSELECT_H
