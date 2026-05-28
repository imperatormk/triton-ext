//===- MetalTripleCompat.h - Out-of-tree Triple::air shim -----*- C++ -*-===//
//
// Out-of-tree compat layer: the stock LLVM prebuilt that Triton ships has no
// `Triple::air` ArchType enumerator, no `Triple::MetalLib` ObjectFormatType,
// and no AIR data-layout dispatch in TargetParser. This header provides
// string-based equivalents and "hijack" enum slots so the existing Metal
// target sources compile against unmodified upstream LLVM headers.
//
//===---------------------------------------------------------------------===//
#ifndef LLVM_METAL_TRIPLECOMPAT_H
#define LLVM_METAL_TRIPLECOMPAT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace metal_compat {

/// String-based detection of an AIR triple ("air", "air64", "air64_vNN").
inline bool isAirTriple(const Triple &T) {
  StringRef A = T.getArchName();
  return A == "air" || A == "air64" || A.starts_with("air64_v");
}

/// In-tree the target is registered via `RegisterTarget<Triple::air, ...>`.
/// Out-of-tree we hijack `Triple::UnknownArch`: the prebuilt's Triple parser
/// returns `UnknownArch` for any unrecognized arch string (including "air"),
/// so `lookupTarget("air", ...)` will dispatch to whatever target registered
/// itself for `UnknownArch`. We are the only target in this dylib that does.
inline constexpr Triple::ArchType kAirArchHijack = Triple::UnknownArch;

/// The AIR data layout string. Copied verbatim from the in-tree
/// `Triple::computeDataLayout` `case Triple::air` block so the out-of-tree
/// pipeline produces byte-identical metallib output.
inline constexpr const char *kAirDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
    "f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-"
    "v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-"
    "v512:512:512-v1024:1024:1024-n8:16:32";

} // namespace metal_compat
} // namespace llvm

#endif // LLVM_METAL_TRIPLECOMPAT_H
