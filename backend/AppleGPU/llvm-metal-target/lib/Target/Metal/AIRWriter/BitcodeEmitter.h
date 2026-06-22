//===- BitcodeEmitter.h - Metal v1 bitcode emitter --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEEMITTER_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEEMITTER_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace llvm {
namespace metal {

/// Emit the Module as bitcode with Metal-compatible typed pointers, using the
/// PointeeTypeMap to emit typed POINTER records (code 8) instead of opaque
/// pointer records (code 25). Replaces LLVM's WriteBitcodeToFile.
std::vector<uint8_t> emitMetalBitcode(llvm::Module &M, PointeeTypeMap &PTM);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEEMITTER_H
