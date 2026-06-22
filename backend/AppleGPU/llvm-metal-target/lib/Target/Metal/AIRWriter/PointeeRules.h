//===- PointeeRules.h - Single-authority pointee-typing rules ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pointee-typing rules that both the one-shot analysis
// (buildPointeeTypeMap) and the post-mutation repairs (PointerPointeeRepair)
// must agree on, expressed ONCE here. Each rule maps a Value (or an intrinsic
// name / arg position) to the pointee it requires, or null when the rule does
// not apply. The analysis calls these to populate the map; the repairs call the
// SAME functions to re-assert the rule after the IR is rewritten into final
// shape. This is the single authority that removes the analysis-vs-repair
// duplication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_POINTEERULES_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_POINTEERULES_H

#include "PointeeTypeMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

namespace llvm {
namespace metal {

// The pointer operand of an `air.atomic.*` call pins to the intrinsic's element
// type (i32/f32), overriding any GEP/byte-buffer source type. Null if Ptr is
// not an atomic pointer operand.
Type *atomicPointeeFromUsers(Value *Ptr);

// simdgroup-matrix intrinsic suffix → element type (i8/f16/bf16/f32). Null when
// the name carries no recognized suffix. Tests bf16 before f16 (substring
// overlap). This is the ONE suffix→element mapping shared by the analysis call
// site and fixMMAPointerSuffixMismatch.
Type *mmaElemFromName(StringRef Name, LLVMContext &Ctx);

// The single pointee a pointer-typed select must carry on both arms + itself:
// prefer the non-inttoptr arm, else either typed arm, else inferred-from-usage.
// Null when neither arm has a known pointee. Shared by analysis Phase 8b and
// fixSelectPointerArms.
Type *requiredSelectPointee(SelectInst *Sel, const PointeeTypeMap &PTM);

// The pointee a pointer phi's record must carry: a device phi takes the atomic
// element type if it feeds one, else float*; a non-device phi (or one with no
// existing entry) gap-fills from the first incoming with a concrete pointee.
// Null when no incoming yields a concrete pointee. Shared by analysis Phase
// 2b/4 and fixPhiIncomingTypes.
Type *requiredPhiPointee(PHINode *PN, const PointeeTypeMap &PTM);

// Reconcile a GEP's base pointee with its source element type: if the base
// (non-global, non-bitcast) carries a different pointee than the GEP's source
// element type, route the base through an identity bitcast pinned to the source
// type. Returns true if it retyped. The ONE base-vs-source rule shared by
// normalizeGEPs' final arm and fixGEPBaseTypeMismatch.
bool reconcileGEPBaseType(GetElementPtrInst *GEP, PointeeTypeMap &PTM);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_POINTEERULES_H
