//===- ConstantExprLower.h - Lower ConstantExprs to instructions *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_CONSTANTEXPRLOWER_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_CONSTANTEXPRLOWER_H

#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

/// Lower ConstantExpr operands to real instructions. The writer pipeline runs
/// it twice: once upstream before buildPointeeTypeMap (so materialized GEPs get
/// PointeeTypeMap entries) and again as emit stage D (to catch constexprs that
/// stages A-C introduce). A constexpr GEP reaching the typed emitter produces a
/// record the Metal reader rejects (PSO "Failed to materializeAll").
void lowerConstantExprs(llvm::Module &M);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_CONSTANTEXPRLOWER_H
