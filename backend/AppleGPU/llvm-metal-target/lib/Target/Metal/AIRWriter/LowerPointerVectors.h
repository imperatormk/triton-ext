//===- LowerPointerVectors.h - Lower vector-of-pointer values ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_LOWERPOINTERVECTORS_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_LOWERPOINTERVECTORS_H

#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

// AIR/Metal has no vector-of-pointer values: a `<N x ptr>` gets a single
// pointee slot in the AIR type table, yet its scalar lanes can carry
// conflicting pointees. Lower the whole pointer-vector web (phis, selects,
// shuffles, insert/extractelement, bitcasts, ptrtoint/inttoptr) to `<N x i64>`,
// reconstructing pointer semantics only at the consuming edges.
void lowerVectorPointerToInt(Module &M);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_LOWERPOINTERVECTORS_H
