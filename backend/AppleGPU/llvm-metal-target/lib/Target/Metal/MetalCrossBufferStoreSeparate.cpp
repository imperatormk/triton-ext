//===- MetalCrossBufferStoreSeparate.cpp - Re-separate device stores ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX-1 root-fix. See MetalCrossBufferStoreSeparate.h and AGX_BUGS.md.
//
// The Apple AGX JIT coalesces device stores by *address*, ignoring source
// order. When two stores to different buffer args land at the same per-thread
// element offset in one straight-line block, the JIT fuses them and drops
// warp 0's value. At -O0 the bug never fires because each masked store sits in
// its own predicated basic block; -O1+ flattens those into one run.
//
// This pass restores the -O0 structure: it groups conflicting device stores by
// their per-thread offset SSA and hoists each offset group into its own
// predicated block guarded by THAT group's in-bounds mask (the dominating
// `icmp slt offset, N`). The guard is always taken for in-bounds threads
// (semantics-identical) but is a real conditional branch, so the JIT can no
// longer address-coalesce the cross-buffer pair. Per-offset grouping keeps a
// store under a different predicate from being pulled under the wrong guard.
//
// A block is rewritten only when it contains a cross-buffer offset group (>=2
// stores reaching DIFFERENT addrspace(1) Args at the same SSA offset), so
// single-output kernels are a byte-identical no-op.
//
//===----------------------------------------------------------------------===//

#include "MetalCrossBufferStoreSeparate.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <functional>

using namespace llvm;

#define DEBUG_TYPE "metal-cross-buffer-store-separate"

namespace {

// Device (AS1) address space, per the Metal address-space map.
static constexpr unsigned kDeviceAS = metal::AS::Device;

// Walk a store-pointer expression back through pointer-preserving ops to the
// reaching base value (ideally an addrspace(1) Argument) and collect the SSA
// offset that selects the per-thread element. Returns the base; OffsetOut is
// the dominating index value (or null if a constant-only / unrecognised shape).
static Value *traceBaseAndOffset(Value *Ptr, Value *&OffsetOut) {
  OffsetOut = nullptr;
  Value *Cur = Ptr;
  while (Cur) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
      // Last non-constant index is the per-thread offset.
      for (Value *Idx : GEP->indices()) {
        if (!isa<ConstantInt>(Idx))
          OffsetOut = Idx;
      }
      Cur = GEP->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastInst>(Cur)) {
      Cur = BC->getOperand(0);
      continue;
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(Cur)) {
      Cur = ASC->getOperand(0);
      continue;
    }
    if (auto *GEPOp = dyn_cast<GEPOperator>(Cur)) {
      for (Value *Idx : GEPOp->indices())
        if (!isa<ConstantInt>(Idx))
          OffsetOut = Idx;
      Cur = GEPOp->getPointerOperand();
      continue;
    }
    break;
  }
  return Cur;
}

// Is this a plain (non-volatile, non-atomic) store to device memory whose
// pointer reaches a kernel Argument? Fills Base / Offset on success.
static bool isDeviceArgStore(StoreInst *SI, Value *&Base, Value *&Offset) {
  if (SI->isVolatile() || !SI->isSimple())
    return false;
  if (SI->getPointerAddressSpace() != kDeviceAS)
    return false;
  Base = traceBaseAndOffset(SI->getPointerOperand(), Offset);
  return isa<Argument>(Base);
}

// Find the in-bounds mask (an i1 `icmp slt/ult offset, N`) that dominates SI.
// Returns null if none is found - the caller then skips the group, since a
// constant-true guard would fold away and the AGX bug would return.
static Value *findMaskForOffset(Value *Offset, Instruction *InsertPt) {
  // Peel sext/zext: the GEP index is often `sext %i32off` while the icmp
  // compares the i32 offset directly.
  SmallVector<Value *, 3> Cands;
  if (Offset) {
    Cands.push_back(Offset);
    if (auto *Ext = dyn_cast<CastInst>(Offset))
      if (Ext->getOpcode() == Instruction::SExt ||
          Ext->getOpcode() == Instruction::ZExt)
        Cands.push_back(Ext->getOperand(0));
  }
  for (Value *Off : Cands) {
    // Accept an slt/ult icmp user defined before the store, so it dominates the
    // conditional branch we emit at InsertPt.
    for (User *U : Off->users()) {
      auto *IC = dyn_cast<ICmpInst>(U);
      if (!IC || !IC->getType()->isIntegerTy(1))
        continue;
      if (IC->getPredicate() != ICmpInst::ICMP_SLT &&
          IC->getPredicate() != ICmpInst::ICMP_ULT)
        continue;
      if (IC->getParent() == &IC->getFunction()->getEntryBlock())
        return IC;
      if (IC->getParent() == InsertPt->getParent() && IC->comesBefore(InsertPt))
        return IC;
    }
  }
  return nullptr;
}

// Hoist a set of conflicting stores (sharing one per-thread offset) into one
// predicated block guarded by `Guard`. The stores (possibly non-adjacent) are
// moved in program order into the new block along with the address GEPs they
// depend on; non-conflicting stores stay in the continuation, so a store under
// a different offset is never pulled under this guard. The real conditional
// branch stops the AGX JIT from treating warp-0 lane stores as unconditional,
// so it no longer address-coalesces the cross-buffer pair. Returns the
// continuation block for re-scanning.
static BasicBlock *guardStoreGroup(ArrayRef<StoreInst *> Group, Value *Guard) {
  assert(!Group.empty() && "empty store group");
  BasicBlock *Pre = Group.front()->getParent();
  // Anchor at the earliest store; the caller guarantees every store's value is
  // defined before this point.
  Instruction *At = Group.front();
  for (StoreInst *SI : Group)
    if (SI->comesBefore(At))
      At = SI;

  // Split before At into Tail, then peel an empty StoreBB between Pre and Tail.
  BasicBlock *Tail = Pre->splitBasicBlock(At, Pre->getName() + ".cbssep.tail");
  BasicBlock *StoreBB = BasicBlock::Create(
      Pre->getContext(), Pre->getName() + ".cbssep.st", Pre->getParent(), Tail);

  // Replace Pre's unconditional branch to Tail with a guarded branch.
  Instruction *PreTerm = Pre->getTerminator();
  IRBuilder<> B(PreTerm);
  B.CreateCondBr(Guard, StoreBB, Tail);
  PreTerm->eraseFromParent();

  // Move each store into StoreBB in order, together with the pure operand-chain
  // instructions (address GEP/bitcast, value glue) that live in Tail and are
  // used only by this group. `Moved` dedups glue shared across the group's
  // stores (e.g. a duplicated output).
  SmallPtrSet<Instruction *, 16> Moved;
  IRBuilder<> SB(StoreBB);
  for (StoreInst *SI : Group) {
    // DFS the pure operand chain in Tail, collected operands-before-users.
    SmallVector<Instruction *, 8> Ordered;
    SmallPtrSet<Instruction *, 8> Seen;
    std::function<void(Value *)> pull = [&](Value *V) {
      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getParent() != Tail || I->mayHaveSideEffects() ||
          isa<StoreInst>(I) || Moved.count(I) || Seen.count(I))
        return;
      // Relocate only when every user is also hoisted; otherwise a user left in
      // Tail would lose dominance over its def sunk into the predicated
      // StoreBB.
      for (User *U : I->users()) {
        auto *UI = dyn_cast<Instruction>(U);
        bool UserHoisted = (UI && (Moved.count(UI) || Seen.count(UI))) ||
                           (isa<StoreInst>(U) &&
                            llvm::is_contained(Group, cast<StoreInst>(U)));
        if (!UserHoisted)
          return; // a non-hoisted user remains -> leave this def in place.
      }
      Seen.insert(I);
      for (Value *Op : I->operands())
        pull(Op);
      Ordered.push_back(I); // post-order: operands first
    };
    pull(SI->getPointerOperand());
    pull(SI->getValueOperand());
    for (Instruction *I : Ordered) {
      I->moveBefore(*StoreBB, StoreBB->end());
      Moved.insert(I);
    }
    SI->moveBefore(*StoreBB, StoreBB->end());
  }
  // Terminate StoreBB into Tail.
  SB.SetInsertPoint(StoreBB);
  SB.CreateBr(Tail);
  return Tail;
}

// Process one function. Returns true if any store was separated.
static bool separateInFunction(Function &F) {
  bool Changed = false;
  // Iterate over a worklist of blocks; sinking splits a block, so we re-scan.
  SmallVector<BasicBlock *, 16> Worklist;
  for (BasicBlock &BB : F)
    Worklist.push_back(&BB);

  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();

    // Collect device-arg stores in this block, in program order, with their
    // (base, offset).
    struct Rec {
      StoreInst *SI;
      Value *Base;
      Value *Offset;
    };
    SmallVector<Rec, 8> Stores;
    for (Instruction &I : *BB)
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Value *Base, *Offset;
        if (isDeviceArgStore(SI, Base, Offset))
          Stores.push_back({SI, Base, Offset});
      }
    if (Stores.size() < 2)
      continue;

    // Group conflicting stores by offset SSA: same per-thread offset but
    // DIFFERENT base args (the AGX coalescing trigger). Process one group per
    // scan, then re-scan the continuation for the next.
    DenseMap<Value *, SmallVector<StoreInst *, 4>> ByOffset;
    DenseMap<Value *, bool> CrossBuffer;
    DenseMap<Value *, Value *> AnyBaseForOffset;
    for (auto &R : Stores) {
      if (!R.Offset)
        continue;
      auto &Vec = ByOffset[R.Offset];
      Vec.push_back(R.SI);
      auto It = AnyBaseForOffset.find(R.Offset);
      if (It == AnyBaseForOffset.end())
        AnyBaseForOffset[R.Offset] = R.Base;
      else if (It->second != R.Base)
        CrossBuffer[R.Offset] = true;
    }

    // Pick one offset group that is cross-buffer, has a real mask, and is safe
    // to hoist. Hoisting moves the group to its first store, so every stored
    // value/pointer must already be available there; a value sunk after the
    // first store would forward-reference and make the AIR writer emit an
    // invalid record, so such a group is skipped rather than miscompiled.
    SmallVector<Value *, 8> OffsetOrder;
    for (auto &R : Stores)
      if (R.Offset && ByOffset.count(R.Offset) &&
          !llvm::is_contained(OffsetOrder, R.Offset))
        OffsetOrder.push_back(R.Offset);

    SmallVector<StoreInst *, 4> *Group = nullptr;
    Value *Guard = nullptr;
    for (Value *Offset : OffsetOrder) {
      auto &Vec = ByOffset[Offset];
      if (!CrossBuffer.lookup(Offset))
        continue; // single-buffer offset: not the trigger, leave it.
      Value *G = findMaskForOffset(Offset, Vec.front());
      if (!G)
        continue; // no real in-bounds mask: never emit a constant guard.
      // Hoist target = the group's first store (earliest in program order).
      StoreInst *FirstSI = Vec.front();
      for (StoreInst *SI : Vec)
        if (SI->comesBefore(FirstSI))
          FirstSI = SI;
      // A value defined after FirstSI is safe only if it is pure and every user
      // is a store in THIS group - then it moves into StoreBB with them and
      // still precedes every use. An impure value, or one with a user outside
      // the group, cannot be relocated, so skip the group rather than emit an
      // invalid forward reference.
      auto allUsersInGroup = [&](Instruction *V) {
        for (User *U : V->users()) {
          auto *SU = dyn_cast<StoreInst>(U);
          if (!SU || !llvm::is_contained(Vec, SU))
            return false;
        }
        return true;
      };
      bool Safe = true;
      for (StoreInst *SI : Vec) {
        auto *V = dyn_cast<Instruction>(SI->getValueOperand());
        if (!V || V->getParent() != BB || V->comesBefore(FirstSI))
          continue; // defined before the hoist point (or non-instr) -> safe.
        if (V->mayHaveSideEffects() || !allUsersInGroup(V)) {
          Safe = false;
          break;
        }
      }
      if (!Safe)
        continue; // value-after-store would forward-reference; leave it.
      Group = &Vec;
      Guard = G;
      break;
    }
    if (!Group)
      continue;

    // The group's stores are in program order within BB (ByOffset preserves
    // insertion order per key). Hoist them behind their shared guard.
    BasicBlock *Cont = guardStoreGroup(*Group, Guard);
    Changed = true;
    // Re-scan the continuation for further offset groups.
    Worklist.push_back(Cont);
  }
  return Changed;
}

static bool runImpl(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Changed |= separateInFunction(F);
  }
  return Changed;
}

} // namespace

PreservedAnalyses
MetalCrossBufferStoreSeparatePass::run(Module &M, ModuleAnalysisManager &AM) {
  return runImpl(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool MetalCrossBufferStoreSeparateLegacy::runOnModule(Module &M) {
  return runImpl(M);
}

char MetalCrossBufferStoreSeparateLegacy::ID = 0;

INITIALIZE_PASS(MetalCrossBufferStoreSeparateLegacy, DEBUG_TYPE,
                "Metal Cross-Buffer Store Separate", false, false)

ModulePass *llvm::createMetalCrossBufferStoreSeparateLegacyPass() {
  return new MetalCrossBufferStoreSeparateLegacy();
}
