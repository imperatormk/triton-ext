//===- MetalAsyncCopyToCooperative.cpp - lower async copy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX bug worked around here: a module that uses air.simdgroup_matrix_8x8_load
// and merely contains an air.simdgroup_async_copy_2d declaration fails Metal
// PSO creation with the opaque "Failed to materializeAll" (no diagnostic, even
// under MTL_DEBUG_LAYER / MTL_SHADER_VALIDATION).  Minimal repro: the failure
// reproduces with the async-copy DECLARATION present and zero call sites, and
// disappears the instant that declaration is erased.
//
// Fix: when (and only when) a module uses BOTH simdgroup-matrix ops and
// air.simdgroup_async_copy_2d, lower every async-copy call to an inline
// cooperative threadgroup copy and every air.wait_simdgroup_events to a
// threadgroup barrier, then drop the now-unused async/wait declarations.  This
// removes the toxic declaration while preserving semantics.  Modules that use
// async copy WITHOUT simdgroup-matrix ops (e.g. a flash-attention-style copy
// pipeline) are left untouched and keep the hardware async DMA.
//
//===----------------------------------------------------------------------===//

#include "MetalAsyncCopyToCooperative.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include <cstdlib>

using namespace llvm;

#define DEBUG_TYPE "metal-async-copy-to-cooperative"

static constexpr StringLiteral kAsyncCopyPrefix("air.simdgroup_async_copy_2d.");
static constexpr StringLiteral kWaitEvents("air.wait_simdgroup_events");
static constexpr StringLiteral kMMAPrefix("air.simdgroup_matrix_8x8_");
static constexpr StringLiteral kTidTG("air.thread_position_in_threadgroup");
// This pass runs AFTER MetalBarrierRename, so emit the already-renamed
// air.wg.barrier directly (a stray air.threadgroup.barrier declaration left
// past the rename pass is itself an AGX materializeAll trigger).
static constexpr StringLiteral kBarrier("air.wg.barrier");

static bool moduleUsesMMA(Module &M) {
  for (Function &F : M)
    if (F.getName().starts_with(kMMAPrefix))
      return true;
  return false;
}

static bool moduleHasAsyncCopy(Module &M) {
  for (Function &F : M)
    if (F.getName().starts_with(kAsyncCopyPrefix))
      return true;
  return false;
}

// Walk GEP/bitcast/addrspacecast/select chains to the underlying threadgroup
// GlobalVariable a pointer refers to (or null if it doesn't reach one).
static GlobalVariable *threadgroupGlobalOf(Value *V) {
  SmallVector<Value *, 8> Work{V};
  SmallPtrSet<Value *, 8> Seen;
  while (!Work.empty()) {
    Value *Cur = Work.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    Cur = Cur->stripPointerCasts();
    if (auto *GV = dyn_cast<GlobalVariable>(Cur)) {
      if (GV->getAddressSpace() == metal::AS::Threadgroup)
        return GV;
      continue;
    }
    if (auto *GEP = dyn_cast<GEPOperator>(Cur))
      Work.push_back(GEP->getPointerOperand());
    else if (auto *Sel = dyn_cast<SelectInst>(Cur)) {
      Work.push_back(Sel->getTrueValue());
      Work.push_back(Sel->getFalseValue());
    } else if (auto *PN = dyn_cast<PHINode>(Cur)) {
      for (Value *In : PN->incoming_values())
        Work.push_back(In);
    }
  }
  return nullptr;
}

// The AGX "materializeAll" trap fires precisely when an
// air.simdgroup_matrix_8x8_load reads from the same threadgroup arena that an
// air.simdgroup_async_copy_2d writes (the @global_smem pipeline buffer).  A
// dedicated dot scratch global read by MMA is fine.  Detect that exact overlap
// so the lowering only fires on the offending kernels and the default GEMM
// keeps its hardware async DMA untouched.
static bool mmaReadsAsyncArena(Module &M) {
  // Collect threadgroup globals written by async copies.
  SmallPtrSet<GlobalVariable *, 4> AsyncTargets;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        if (!CI->getCalledFunction()->getName().starts_with(kAsyncCopyPrefix))
          continue;
        if (CI->arg_size() > 2)
          if (auto *GV = threadgroupGlobalOf(CI->getArgOperand(2)))
            AsyncTargets.insert(GV);
      }
  }
  if (AsyncTargets.empty())
    return false;
  // Any MMA load/store whose pointer reaches one of those globals?
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        if (!Name.starts_with(kMMAPrefix) || !Name.contains("_load"))
          if (!Name.starts_with(kMMAPrefix) || !Name.contains("_store"))
            continue;
        // ptr arg is 0 for load, 1 for store.
        unsigned PtrArg = Name.contains("_store") ? 1u : 0u;
        if (PtrArg >= CI->arg_size())
          continue;
        if (auto *GV = threadgroupGlobalOf(CI->getArgOperand(PtrArg)))
          if (AsyncTargets.count(GV))
            return true;
      }
  }
  return false;
}

// Declare (or fetch) air.thread_position_in_threadgroup : () -> [3 x i32].
static Function *getTidFn(Module &M) {
  if (auto *F = M.getFunction(kTidTG))
    return F;
  auto &Ctx = M.getContext();
  auto *RetTy = ArrayType::get(Type::getInt32Ty(Ctx), 3);
  auto *FT = FunctionType::get(RetTy, {}, false);
  return Function::Create(FT, Function::ExternalLinkage, kTidTG, &M);
}

// Declare (or fetch) air.threadgroup.barrier : (i32, i32) -> void.
static Function *getBarrierFn(Module &M) {
  if (auto *F = M.getFunction(kBarrier))
    return F;
  auto &Ctx = M.getContext();
  auto *I32 = Type::getInt32Ty(Ctx);
  auto *FT = FunctionType::get(Type::getVoidTy(Ctx), {I32, I32}, false);
  return Function::Create(FT, Function::ExternalLinkage, kBarrier, &M);
}

// Get the flattened thread id (x lane of thread_position_in_threadgroup).
// All cooperative loops here are 1-D over the threadgroup x dimension, which
// matches how the Triton frontend lays out warps (32-lane simdgroups stacked
// along x).
static Value *getFlatTid(IRBuilder<> &B, Module &M) {
  auto *TidFn = getTidFn(M);
  Value *Tid = B.CreateCall(TidFn, {}, "tid");
  return B.CreateExtractValue(Tid, {0}, "tid.x");
}

// Lower one air.simdgroup_async_copy_2d call to a cooperative copy.
//
// Call ABI (as emitted by the Triton plugin):
//   (i64 sizeof, i64 alignof, p3 dst, i64 dstStrideBytes, i64 dstElemStride,
//    <2 x i64> dstTile=<widthBytes,rows>, p1 src, i64 srcStrideBytes,
//    i64 srcElemStride, <2 x i64> srcTile, <2 x i64> offset, i32 clamp)
//
// The copy is byte-exact: rows * widthBytes bytes, dst row r at
// dst + r*dstStrideBytes, src row r at src + r*srcStrideBytes.  We copy in
// 4-byte (float) chunks when widthBytes is a multiple of 4 (always true for the
// f32/f16/bf16 MMA tiles), else byte by byte.  The destination total
// threadgroup element span is treated as a flat <rows*widthBytes> region
// addressed per row so the (possibly padded) dst pitch is honoured. Returns the
// constant element of a <2 x i64> vector operand, or -1 if not a compile-time
// constant.
static int64_t constVecElem(Value *V, unsigned Idx) {
  if (auto *CV = dyn_cast<Constant>(V)) {
    if (auto *CE = CV->getAggregateElement(Idx))
      if (auto *CI = dyn_cast<ConstantInt>(CE))
        return (int64_t)CI->getZExtValue();
  }
  return -1;
}

static bool lowerAsyncCopy(CallInst *CI, Module &M, unsigned TGSize) {
  auto &Ctx = M.getContext();
  IRBuilder<> B(CI);
  auto *I64 = Type::getInt64Ty(Ctx);
  auto *F32 = Type::getFloatTy(Ctx);

  Value *Dst = CI->getArgOperand(2);
  Value *DstStrideBytes = CI->getArgOperand(3);
  Value *DstTile = CI->getArgOperand(5);
  Value *Src = CI->getArgOperand(6);
  Value *SrcStrideBytes = CI->getArgOperand(7);

  // The tile geometry (<widthBytes, rows>) is a compile-time constant for every
  // Triton-emitted async copy.  Fully unroll the copy so it is straight-line:
  // a loop here would get a threadgroup barrier inserted into its body by the
  // later TGBarrierInsert pass, and a barrier inside a per-thread-divergent
  // loop is itself an AGX "materializeAll" trigger.
  int64_t WidthBytes = constVecElem(DstTile, 0);
  int64_t Rows = constVecElem(DstTile, 1);
  if (WidthBytes < 0 || Rows < 0 || (WidthBytes % 4) != 0)
    return false; // leave it to the (rare) non-constant fallback: keep async
  int64_t WidthF = WidthBytes / 4;
  int64_t Total = Rows * WidthF;
  unsigned Step = TGSize ? TGSize : 32;

  Value *Flat = getFlatTid(B, M);
  Value *FlatI64 = B.CreateZExt(Flat, I64, "tid64");
  Value *Four = ConstantInt::get(I64, 4);
  Value *WF = ConstantInt::get(I64, WidthF);
  Value *TotalC = ConstantInt::get(I64, Total);

  // Fully-unrolled, branch-free strided cooperative copy:
  //   for k in 0..ceil(Total/Step): i = tid + k*Step;
  //     valid = i < Total; ii = valid ? i : 0;     // clamp out-of-range to 0
  //     r = ii / WidthF; c = ii % WidthF;
  //     dst[r,c] = src[r,c];                        // element 0 is copied by a
  //                                                 // real thread too ->
  //                                                 idempotent
  // No loop and no per-element if-region, so the later barrier-insertion pass
  // has nothing to wrap, avoiding the divergent-loop materializeAll trap.
  // Element (float) strides.  The threadgroup dst is tightly packed in the
  // Triton MMA path (dstPitchBytes == widthBytes), so the dst flat float index
  // is exactly II; address it with a FLOAT gep (matching the form the MMA loads
  // and the frontend sync copy use) rather than an i8 byte gep, so the AGX
  // dependency analysis sees a homogeneous typed access to the arena and can
  // order the store before the simdgroup-matrix load.  The device src keeps its
  // (possibly wider) row pitch.
  Value *SrcPitchF = B.CreateUDiv(SrcStrideBytes, Four, "ci.srcpitchf");
  Value *DstPitchF = B.CreateUDiv(DstStrideBytes, Four, "ci.dstpitchf");
  int64_t KMax = (Total + (int64_t)Step - 1) / (int64_t)Step;
  for (int64_t k = 0; k < KMax; ++k) {
    Value *I =
        B.CreateAdd(FlatI64, ConstantInt::get(I64, k * (int64_t)Step), "ci.i");
    Value *Valid = B.CreateICmpULT(I, TotalC, "ci.valid");
    Value *II = B.CreateSelect(Valid, I, ConstantInt::get(I64, 0), "ci.ii");
    Value *R = B.CreateUDiv(II, WF, "ci.r");
    Value *C = B.CreateURem(II, WF, "ci.c");
    // dst float index = R*dstPitchF + C ; src float index = R*srcPitchF + C
    Value *DstIdx = B.CreateAdd(B.CreateMul(R, DstPitchF), C, "ci.dstidx");
    Value *SrcIdx = B.CreateAdd(B.CreateMul(R, SrcPitchF), C, "ci.srcidx");
    Value *DstP = B.CreateGEP(F32, Dst, DstIdx, "ci.dstp");
    Value *SrcP = B.CreateGEP(F32, Src, SrcIdx, "ci.srcp");
    Value *V = B.CreateAlignedLoad(F32, SrcP, Align(4), "ci.v");
    B.CreateAlignedStore(V, DstP, Align(4));
  }

  // Replace the event-pointer result with the dst pointer; the wait is lowered
  // to a barrier and any event store becomes dead.
  CI->replaceAllUsesWith(Dst);
  CI->eraseFromParent();
  return true;
}

static bool asyncCopyToCooperative(Module &M, unsigned TGSize) {
  // Disabled: keep the genuine air.simdgroup_async_copy_2d hardware DMA. The AGX
  // materializeAll trap this pass worked around no longer reproduces here.
  return false;
  if (!moduleUsesMMA(M) || !moduleHasAsyncCopy(M))
    return false;
  // Only fire on the exact AGX trap: an MMA load/store that touches the same
  // threadgroup arena an async copy writes.  Leaves async DMA intact otherwise.
  if (!mmaReadsAsyncArena(M))
    return false;

  bool Changed = false;

  // 1. Lower wait calls to threadgroup barriers.
  SmallVector<CallInst *, 8> Waits;
  SmallVector<CallInst *, 8> Copies;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        if (Name == kWaitEvents)
          Waits.push_back(CI);
        else if (Name.starts_with(kAsyncCopyPrefix))
          Copies.push_back(CI);
      }
  }

  auto *I32 = Type::getInt32Ty(M.getContext());
  for (CallInst *W : Waits) {
    IRBuilder<> B(W);
    Function *Barr = getBarrierFn(M);
    // air.wg.barrier(2, 1): the post-rename threadgroup-execution-barrier form
    // the rest of the pipeline emits.
    B.CreateCall(Barr, {ConstantInt::get(I32, 2), ConstantInt::get(I32, 1)});
    W->eraseFromParent();
    Changed = true;
  }

  // 2. Lower async copies to cooperative copies.
  bool AllLowered = true;
  for (CallInst *C : Copies) {
    if (lowerAsyncCopy(C, M, TGSize))
      Changed = true;
    else
      AllLowered = false;
  }

  // 3. Erase now-dead event allocas + their dead stores (the event storage the
  // async copies fed and the waits drained).  An "alloca ptr addrspace(3)" with
  // only store users is dead once the wait/copy calls are gone.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<AllocaInst *, 2> DeadAllocas;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          if (!AI->getAllocatedType()->isPointerTy())
            continue;
          if (AI->getAllocatedType()->getPointerAddressSpace() != 3)
            continue;
          bool OnlyStores = true;
          for (User *U : AI->users())
            if (!isa<StoreInst>(U)) {
              OnlyStores = false;
              break;
            }
          if (OnlyStores)
            DeadAllocas.push_back(AI);
        }
    for (AllocaInst *AI : DeadAllocas) {
      SmallVector<Instruction *, 4> Stores;
      for (User *U : AI->users())
        Stores.push_back(cast<Instruction>(U));
      for (Instruction *S : Stores)
        S->eraseFromParent();
      AI->eraseFromParent();
      Changed = true;
    }
  }

  // 4. Erase the now-unused async-copy / wait declarations (the toxic symbols).
  // Only safe if every copy was lowered; otherwise a live call remains.
  if (AllLowered) {
    SmallVector<Function *, 4> Dead;
    for (Function &F : M)
      if ((F.getName().starts_with(kAsyncCopyPrefix) ||
           F.getName() == kWaitEvents) &&
          F.use_empty())
        Dead.push_back(&F);
    for (Function *F : Dead) {
      F->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

// Best-effort threadgroup size from !air.max_threads_per_threadgroup or a
// kernel's reqd metadata; default to 32 (one simdgroup) which is always safe
// (the loop just iterates more times per thread, never skips elements).
static unsigned getThreadgroupSize(Module &M) {
  // Triton kernels carry num-warps via the launch, not reliably in IR here;
  // 32 is a correct lower bound (over-iterates harmlessly).  Using a larger
  // power-of-two would skip elements if the real TG is smaller, so keep 32.
  (void)M;
  return 32;
}

PreservedAnalyses
MetalAsyncCopyToCooperativePass::run(Module &M, ModuleAnalysisManager &AM) {
  return asyncCopyToCooperative(M, getThreadgroupSize(M))
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}

bool MetalAsyncCopyToCooperativeLegacy::runOnModule(Module &M) {
  return asyncCopyToCooperative(M, getThreadgroupSize(M));
}

char MetalAsyncCopyToCooperativeLegacy::ID = 0;

INITIALIZE_PASS(MetalAsyncCopyToCooperativeLegacy, DEBUG_TYPE,
                "Metal Async Copy To Cooperative", false, false)

ModulePass *llvm::createMetalAsyncCopyToCooperativeLegacyPass() {
  return new MetalAsyncCopyToCooperativeLegacy();
}
