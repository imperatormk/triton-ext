//===- PointerPointeeRepair.h - Pointer/pointee type agreement --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_POINTERPOINTEEREPAIR_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_POINTERPOINTEEREPAIR_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

// Remove ptr-to-ptr bitcasts only where the PTM records the same pointee on
// both sides.
void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM);

// Make a pointer phi's incomings agree with the phi's record pointee.
void fixPhiIncomingTypes(Module &M, PointeeTypeMap &PTM);

// Give each simdgroup-matrix call a pointer whose pointee matches the
// intrinsic's element suffix.
void fixMMAPointerSuffixMismatch(Module &M, PointeeTypeMap &PTM);

// Make both arms of a pointer select agree on a single pointee.
void fixSelectPointerArms(Module &M, PointeeTypeMap &PTM);

// Make every load/store's pointer pointee equal its access type.
void fixAccessTypeMismatch(Module &M, PointeeTypeMap &PTM);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_POINTERPOINTEEREPAIR_H
