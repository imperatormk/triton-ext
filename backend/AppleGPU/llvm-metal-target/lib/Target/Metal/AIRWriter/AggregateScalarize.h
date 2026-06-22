//===- AggregateScalarize.h - Scalarize aggregate/bool-vec ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_AGGREGATESCALARIZE_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_AGGREGATESCALARIZE_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

// The AGX JIT cannot legalize bool-vector bitcasts; expand them into per-bit
// shifts.
void scalarizeBoolVectorCasts(Module &M);

// AIR v1 bitcode has no aggregate load; expand `[N x T]` loads into per-element
// GEP+load+insertvalue.
void scalarizeAggregateLoads(Module &M, PointeeTypeMap &PTM);

// Split array-valued stores and vector-through-aggregate stores into element
// stores (an array can't be a Metal store value type).
void scalarizeAggregateStores(Module &M, PointeeTypeMap &PTM);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_AGGREGATESCALARIZE_H
