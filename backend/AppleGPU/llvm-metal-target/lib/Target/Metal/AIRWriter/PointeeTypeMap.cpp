//===- PointeeTypeMap.cpp - Typed-pointer reconstruction --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reconstructs per-value pointee types for opaque LLVM pointers so the AIR
// bitcode writer can emit typed POINTER records (code 8) that the Metal GPU
// JIT requires. The map is populated by walking memory ops and GEPs, then
// consulted by BitcodeEmitter / TypeTableWriter / FunctionWriter when they
// need a concrete pointee for a Value or function-parameter type.
//
//===----------------------------------------------------------------------===//

#include "PointeeTypeMap.h"
#include "CoopTensorLowering.h"
#include "MetalConstraints.h"
#include "PointeeRules.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

AnalysisKey PointeeTypeAnalysis::Key;

StructType *getOrCreateEventType(LLVMContext &Ctx) {
  if (StructType *EventTy = StructType::getTypeByName(Ctx, kEventTypeName))
    return EventTy;
  return StructType::create(Ctx, kEventTypeName);
}

static bool isNonFloatScalarDevicePointer(Value *Ptr);

// Infer pointee from usage: load/store/GEP, then GEP source type and atomic
// intrinsic name inference.
Type *PointeeTypeMap::inferFromUsage(Value *Ptr) {
  SmallPtrSet<Value *, 8> Visited;
  return inferFromUsage(Ptr, Visited);
}

Type *PointeeTypeMap::inferFromUsage(Value *Ptr,
                                     SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(Ptr).second)
    return nullptr;
  // users() asserts without a use-list; such values carry no usage evidence.
  if (!Ptr->hasUseList())
    return nullptr;
  // Prioritize load/store over GEP source types; recurse through GEP chains.
  // Do NOT follow atomic intrinsics through GEP chains: a float buffer through
  // a float GEP into an i32 CAS must keep the GEP source type (float); the
  // atomic mismatch is handled by InferTypedPointersPass Phase 1b.
  Type *GepType = nullptr;
  for (auto *U : Ptr->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U))
      return LI->getType();
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() == Ptr)
        return SI->getValueOperand()->getType();
    }
    // Follow identity ptr->ptr bitcasts only on DEVICE (addrspace 1) pointers.
    // NOT for addrspace(3): smem array globals must keep their array GEP
    // element type, else the writer rejects "gep type does not match pointee
    // type".
    if (auto *BC = dyn_cast<BitCastInst>(U)) {
      if (BC->getType()->isPointerTy() &&
          BC->getType()->getPointerAddressSpace() == AS::Device)
        if (Type *T = inferFromUsage(BC, Visited)) {
          auto *GEPPtr = dyn_cast<GetElementPtrInst>(Ptr);
          auto *VT = dyn_cast<VectorType>(T);
          if (GEPPtr && VT &&
              VT->getElementType() == GEPPtr->getResultElementType())
            return GEPPtr->getResultElementType();
          return T;
        }
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (Type *T = inferFromUsage(GEP, Visited)) {
        if (auto *VT = dyn_cast<VectorType>(T))
          if (VT->getElementType() == GEP->getSourceElementType())
            return GEP->getSourceElementType();
        return T;
      }
      if (!GepType)
        GepType = GEP->getSourceElementType();
    }
    // Follow pointer-typed select users so both arms get a typed-pointer slot
    // (writer cannot emit typed `select ptr` otherwise).
    if (auto *Sel = dyn_cast<SelectInst>(U)) {
      if (Sel->getType()->isPointerTy())
        if (Type *T = inferFromUsage(Sel, Visited))
          return T;
    }
    // Follow pointer-typed phi users (loop-carried pointers): loads often
    // happen only through the phi, so without this the base infers the byte-GEP
    // type and the phi record's operand types disagree. Visited guards cycles.
    if (auto *PN = dyn_cast<PHINode>(U)) {
      if (PN->getType()->isPointerTy())
        if (Type *T = inferFromUsage(PN, Visited))
          return T;
    }
    if (auto *CI = dyn_cast<CallInst>(U)) {
      if (auto *Callee = CI->getCalledFunction()) {
        StringRef Name = Callee->getName();
        auto &Ctx = Ptr->getContext();
        // The simdgroup-matrix intrinsic's pointer suffix is definitive
        // element-type evidence, outvoting byte-form GEP source types.
        if (Type *Elem = mmaElemFromName(Name, Ctx))
          return Elem;
        // Tensor-handle pinning for cooperative_tensor / matmul2d builtins.
        {
          unsigned ArgNo = ~0u;
          for (unsigned J = 0; J < CI->arg_size(); ++J)
            if (CI->getArgOperand(J) == Ptr) {
              ArgNo = J;
              break;
            }
          if (Type *Want = tensorHandlePointee(Name, ArgNo, Ptr, Ctx))
            return Want;
        }
        // Only use atomic type when the pointer is NOT a GEP result; GEP
        // results keep their source element type (atomic mismatch handled via
        // ptrtoint+inttoptr).
        if (!isa<GetElementPtrInst>(Ptr) && Name.starts_with("air.atomic.")) {
          if (Name.ends_with(".i32"))
            return Type::getInt32Ty(Ptr->getContext());
          else if (Name.ends_with(".f32"))
            return Type::getFloatTy(Ptr->getContext());
        }
      }
    }
  }
  return GepType;
}

// With MMA intrinsics present the JIT rejects an ambiguously-typed device
// pointer; pin each addrspace(1) entry to float* EXCEPT genuine non-float
// scalar buffers (int8-dot/half/bfloat), which keep their element type.
void PointeeTypeMap::collapseDevicePointersToFloat(Module &M) {
  Type *F32 = Type::getFloatTy(M.getContext());
  for (auto &[Ptr, Ty] : map) {
    auto *PtrTy = Ptr->getType();
    if (auto *PT = dyn_cast<PointerType>(PtrTy)) {
      if (PT->getAddressSpace() == AS::Device &&
          !isNonFloatScalarDevicePointer(Ptr))
        Ty = F32;
    }
  }
}

// Metal has no i1 memory type (pointers to i1 crash the GPU JIT); remap to i8.
void PointeeTypeMap::remapI1ToI8(Module &M) {
  Type *I8 = Type::getInt8Ty(M.getContext());
  for (auto &[Ptr, Ty] : map) {
    if (Ty && Ty->isIntegerTy(1))
      Ty = I8;
  }
}

// This analysis MUST be self-contained - it may be re-run after pipeline passes
// invalidate it, so all Metal-specific overrides (MMA, async copy) live here.

static bool functionUsesMMA(const Function &F) {
  for (const auto &BB : F)
    for (const auto &I : BB)
      if (const auto *CI = dyn_cast<CallInst>(&I))
        if (const auto *Callee = CI->getCalledFunction())
          if (Callee->getName().starts_with(mma_intrinsics::kPrefix))
            return true;
  return false;
}

// MMA collapse forces device pointers to float*, but an MMA kernel can have a
// genuine non-float scalar device buffer (i32 int8-dot output, half/bfloat
// input). Collapsing those to float* makes the writer reject "load/store type
// does not match pointee type". Preserve any device pointer whose inferred
// usage is a concrete non-float scalar (int/half/bfloat).
static bool isNonFloatScalar(Type *Ty) {
  if (!Ty)
    return false;
  if (Ty->isIntegerTy() && !Ty->isIntegerTy(1))
    return true;
  return Ty->isHalfTy() || Ty->isBFloatTy();
}

static bool isNonFloatScalarDevicePointer(Value *Ptr) {
  if (isNonFloatScalar(PointeeTypeMap::inferFromUsage(Ptr)))
    return true;
  // An identity ptr->ptr device bitcast feeding a non-float load/store must not
  // collapse to float*, or the writer rejects "load/store type does not match
  // pointee type". Catch the access type at the immediate use directly.
  if (auto *BC = dyn_cast<BitCastInst>(Ptr))
    if (BC->getType()->isPointerTy() && BC->getSrcTy() == BC->getDestTy())
      for (auto *U : BC->users()) {
        if (auto *LI = dyn_cast<LoadInst>(U))
          if (isNonFloatScalar(LI->getType()))
            return true;
        if (auto *SI = dyn_cast<StoreInst>(U))
          if (SI->getPointerOperand() == BC &&
              isNonFloatScalar(SI->getValueOperand()->getType()))
            return true;
      }
  return false;
}

PointeeTypeMap buildPointeeTypeMap(Module &M) {
  PointeeTypeMap PTM;
  Type *F32 = Type::getFloatTy(M.getContext());
  Type *I8 = Type::getInt8Ty(M.getContext());

  // Detect MMA and async copy presence
  bool HasMMA = false;
  bool HasAsyncCopy = false;
  DenseMap<const Function *, bool> FunctionHasMMA;
  for (auto &F : M) {
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      HasMMA = true;
    if (F.getName().starts_with("air.simdgroup_async_copy_2d."))
      HasAsyncCopy = true;
    if (!F.isDeclaration())
      FunctionHasMMA[&F] = functionUsesMMA(F);
  }

  // Phase 1: Function parameters - infer from usage
  for (auto &F : M)
    for (auto &Arg : F.args())
      if (Arg.getType()->isPointerTy())
        if (auto *Ty = PointeeTypeMap::inferFromUsage(&Arg))
          PTM.set(&Arg, Ty);

  // Phase 2: Instructions that produce pointers
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy())
          continue;
        if (auto *Ty = PointeeTypeMap::inferFromUsage(&I))
          PTM.set(&I, Ty);
      }

  // Phase 2b: device pointer phis carry the atomic element type (if they feed
  // an atomic) else float*; the shared phi rule enforces this.
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN || !PN->getType()->isPointerTy())
          continue;
        if (PN->getType()->getPointerAddressSpace() != AS::Device)
          continue;
        if (Type *Ty = requiredPhiPointee(PN, PTM))
          PTM.set(PN, Ty);
      }

  // Phase 3: Global variables
  for (auto &GV : M.globals())
    if (GV.getType()->isPointerTy())
      PTM.set(&GV, GV.getValueType());

  // Phase 4: Fill gaps - phi incoming, inttoptr, GEP
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        if (!I.getType()->isPointerTy() || PTM.has(&I))
          continue;
        if (auto *PHI = dyn_cast<PHINode>(&I))
          if (Type *Ty = requiredPhiPointee(PHI, PTM))
            PTM.set(&I, Ty);
        if (isa<IntToPtrInst>(&I))
          if (auto *Ty = PointeeTypeMap::inferFromUsage(&I))
            PTM.set(&I, Ty);
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          PTM.set(&I, GEP->getResultElementType());
        if (auto *AI = dyn_cast<AllocaInst>(&I))
          PTM.set(&I, AI->getAllocatedType());
      }

  // Phase 5: i1* → i8*
  PTM.remapI1ToI8(M);

  // Phase 6: MMA - collapse device pointers to float* only in functions that
  // actually use MMA intrinsics.
  if (HasMMA) {
    for (auto &F : M) {
      if (F.isDeclaration() || !FunctionHasMMA.lookup(&F))
        continue;

      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device &&
            !isNonFloatScalarDevicePointer(&Arg))
          PTM.set(&Arg, F32);

      for (auto &BB : F)
        for (auto &I : BB)
          if (I.getType()->isPointerTy() &&
              I.getType()->getPointerAddressSpace() == AS::Device &&
              !isNonFloatScalarDevicePointer(&I))
            PTM.set(&I, F32);

      for (auto &Arg : F.args())
        if (Arg.getType()->isPointerTy() &&
            Arg.getType()->getPointerAddressSpace() == AS::Device &&
            !PTM.has(&Arg) && !isNonFloatScalarDevicePointer(&Arg))
          PTM.set(&Arg, F32);
    }

    // MMA declaration params → typed pointer (float*/half*/bfloat*)
    for (auto &F : M) {
      if (!F.isDeclaration())
        continue;
      if (Type *PtrPointee = mmaElemFromName(F.getName(), M.getContext()))
        for (auto &Arg : F.args())
          if (Arg.getType()->isPointerTy())
            PTM.set(&Arg, PtrPointee);
    }

    // MMA kernel device pointer args → float*
    for (auto &F : M)
      if (!F.isDeclaration() && FunctionHasMMA.lookup(&F))
        for (auto &Arg : F.args())
          if (Arg.getType()->isPointerTy() &&
              Arg.getType()->getPointerAddressSpace() == AS::Device &&
              !isNonFloatScalarDevicePointer(&Arg))
            PTM.set(&Arg, F32);

    // MMA call site pointer operands → typed pointer
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction())
            continue;
          StringRef Name = CI->getCalledFunction()->getName();
          Type *PtrPointee = mmaElemFromName(Name, M.getContext());
          if (PtrPointee)
            for (unsigned J = 0; J < CI->arg_size(); J++)
              if (CI->getArgOperand(J)->getType()->isPointerTy())
                PTM.set(CI->getArgOperand(J), PtrPointee);
        }
  }

  // Phase 7: Async copy overrides (AFTER MMA collapse, re-applies i8*)
  if (HasAsyncCopy) {
    StructType *EventTy = getOrCreateEventType(M.getContext());

    for (auto &F : M) {
      if (!F.isDeclaration())
        continue;
      StringRef Name = F.getName();

      if (Name.starts_with("air.simdgroup_async_copy_2d.")) {
        // Return type → event_t (set on call results)
        for (auto &FN : M)
          for (auto &BB : FN)
            for (auto &I : BB)
              if (auto *CI = dyn_cast<CallInst>(&I))
                if (CI->getCalledFunction() == &F)
                  PTM.set(CI, EventTy);
        // Pointer params → i8* (byte copy)
        for (unsigned J = 0; J < F.arg_size(); J++) {
          auto &Arg = *F.getArg(J);
          if (!Arg.getType()->isPointerTy())
            continue;
          PTM.set(&Arg, I8);
          for (auto *U : F.users())
            if (auto *CI = dyn_cast<CallInst>(U))
              if (J < CI->arg_size())
                PTM.set(CI->getArgOperand(J), I8);
        }
      }

      if (Name == "air.wait_simdgroup_events") {
        // Param 1: pointer to event_t pointer storage
        Type *PtrAs3 = PointerType::get(M.getContext(), 3);
        if (F.arg_size() >= 2) {
          auto &Arg = *F.getArg(1);
          if (Arg.getType()->isPointerTy()) {
            PTM.set(&Arg, PtrAs3);
            for (auto *U : F.users())
              if (auto *CI = dyn_cast<CallInst>(U))
                if (CI->arg_size() >= 2)
                  PTM.set(CI->getArgOperand(1), PtrAs3);
          }
        }
      }
    }

    // Event alloca: stores event_t pointers
    Type *PtrAs3 = PointerType::get(M.getContext(), 3);
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *AI = dyn_cast<AllocaInst>(&I))
            if (AI->getAllocatedType()->isPointerTy() &&
                AI->getAllocatedType()->getPointerAddressSpace() == 3)
              PTM.set(AI, PtrAs3);

    // Propagate event_t through identity bitcasts
    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          auto *BC = dyn_cast<BitCastInst>(&I);
          if (!BC || BC->getSrcTy() != BC->getDestTy())
            continue;
          if (!BC->getType()->isPointerTy())
            continue;
          if (auto *SrcTy = PTM.get(BC->getOperand(0)))
            if (SrcTy == EventTy || isa<PointerType>(SrcTy))
              PTM.set(BC, SrcTy);
        }
  }

  // Phase 8b: Unify pointer-typed select arms. The Metal GPU JIT refuses to
  // materialize a `select i1, ptr, ptr` whose arm pointee types disagree. Force
  // both arms and the select to one pointee, preferring a concrete
  // (non-inttoptr) arm.
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *Sel = dyn_cast<SelectInst>(&I);
        if (!Sel || !Sel->getType()->isPointerTy())
          continue;
        Value *TV = Sel->getTrueValue();
        Value *FV = Sel->getFalseValue();
        if (PTM.get(TV) == PTM.get(FV))
          continue;
        Type *Unified = requiredSelectPointee(Sel, PTM);
        if (!Unified)
          continue;
        PTM.set(TV, Unified);
        PTM.set(FV, Unified);
        PTM.set(Sel, Unified);
      }

  // Phase 9: Function pointer
  for (auto &F : M)
    if (!F.isDeclaration()) {
      PTM.set(&F, F.getFunctionType());
      break;
    }

  return PTM;
}

PointeeTypeMap PointeeTypeAnalysis::run(Module &M, ModuleAnalysisManager &) {
  return buildPointeeTypeMap(M);
}

} // namespace metal
} // namespace llvm
