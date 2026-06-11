//===- BitcodeEmitter.cpp - Metal v1 bitcode emitter ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level orchestrator for Metal v1 bitcode emission. Delegates to
// ValueEnumerator, TypeTableWriter, ConstantsWriter, MetadataWriter,
// and FunctionWriter.
//
//===----------------------------------------------------------------------===//

#include "BitcodeEmitter.h"
#include "BitcodeEncoding.h"
#include "MetadataWriter.h"
#include "MetalConstraints.h"
#include "MetalVersion.h"
#include "ValueEnumerator.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <functional>

using namespace llvm;

namespace llvm {
namespace metal {

// Lower all ConstantExpr operands in instructions to real instructions.
// Metal's GPU JIT doesn't handle constant expression records in bitcode,
// so they must be materialized as instructions before serialization.
//
// For byte-stride GEPs on threadgroup float globals (e.g.,
// gep i8, @tg_global, i64 byte_offset where @tg_global is [N x float]),
// converts to float-element GEPs (gep float, @base, i64 float_index)
// because Metal v1 typed-pointer bitcode requires GEP source type to match
// the pointer's pointee type.
void lowerConstantExprs(Module &M) {
  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  unsigned FloatSize = M.getDataLayout().getTypeAllocSize(FloatTy);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<std::pair<Instruction *, unsigned>, 32> Worklist;
    bool Changed = true;
    while (Changed) {
      Changed = false;
      Worklist.clear();
      for (auto &BB : F)
        for (auto &I : BB)
          for (unsigned J = 0; J < I.getNumOperands(); J++)
            if (isa<ConstantExpr>(I.getOperand(J)))
              Worklist.push_back({&I, J});
      for (auto &[I, OpIdx] : Worklist) {
        auto *CE = cast<ConstantExpr>(I->getOperand(OpIdx));
        Instruction *NewI = CE->getAsInstruction();

        // Convert byte-stride GEPs on TG float globals to float-element GEPs.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(NewI)) {
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == metal::AS::Threadgroup &&
              GEP->getNumIndices() == 1) {
            // Check if base is a TG global with float array element type
            Value *Base = GEP->getPointerOperand();
            GlobalVariable *GV = dyn_cast<GlobalVariable>(Base);
            if (GV) {
              Type *ElemTy = nullptr;
              if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                ElemTy = AT->getElementType();
              if (ElemTy && ElemTy->isFloatTy()) {
                Value *ByteIdx = GEP->idx_begin()->get();
                if (auto *CI = dyn_cast<ConstantInt>(ByteIdx)) {
                  uint64_t ByteOff = CI->getZExtValue();
                  if (ByteOff % FloatSize == 0) {
                    // Find or create a base float* from the GV.
                    // Look for an existing gep [N x float], @GV, 0, 0 in the
                    // function's entry block.
                    Value *FloatBase = nullptr;
                    for (auto *U : GV->users()) {
                      auto *BaseGEP = dyn_cast<GetElementPtrInst>(U);
                      if (!BaseGEP || !BaseGEP->getParent())
                        continue;
                      if (BaseGEP->getFunction() == &F &&
                          BaseGEP->getSourceElementType() ==
                              GV->getValueType() &&
                          BaseGEP->getNumIndices() == 2) {
                        FloatBase = BaseGEP;
                        break;
                      }
                    }
                    if (!FloatBase) {
                      // Create base GEP: gep [N x float], @GV, 0, 0
                      auto *NewBaseGEP = GetElementPtrInst::CreateInBounds(
                          GV->getValueType(), GV,
                          {ConstantInt::get(I64Ty, 0),
                           ConstantInt::get(I64Ty, 0)});
                      NewBaseGEP->insertBefore(
                          F.getEntryBlock().getFirstInsertionPt());
                      FloatBase = NewBaseGEP;
                    }
                    // Create: gep float, %base, i64 (byteOff/4)
                    auto *FloatGEP = GetElementPtrInst::CreateInBounds(
                        ElemTy, FloatBase,
                        {ConstantInt::get(I64Ty, ByteOff / FloatSize)});
                    FloatGEP->insertBefore(I->getIterator());
                    I->setOperand(OpIdx, FloatGEP);
                    NewI->deleteValue(); // discard the byte GEP
                    Changed = true;
                    continue;
                  }
                }
              }
            }
          }
        }

        NewI->insertBefore(I->getIterator());
        I->setOperand(OpIdx, NewI);
        Changed = true;
      }
    }

    // Identity ptr-to-ptr bitcasts (same opaque type, different typed pointer
    // semantics) are kept - the FunctionWriter handles them by emitting a
    // bitcast to the correct typed pointer type inferred from PTM/usage.
  }
}

// Fix GEP source type / pointer pointee mismatches for Metal typed bitcode.
//
// Metal v1 typed-pointer bitcode requires the GEP source element type to match
// the pointer's pointee type. When a GEP uses a different element type than the
// pointer (e.g., `gep half, float*3 %ptr`), the GPU JIT rejects it.
//
// For TG (AS3) pointers where the buffer is float-typed (from MMA merge) but
// accessed with half/i8 GEPs (from pipelined loads), we insert identity
// bitcasts before the GEP. The bitcast creates a new pointer value that the
// PTM can type as half* instead of float*, making the GEP consistent:
// %bc = bitcast float*3 %ptr to float*3 (identity in opaque-ptr IR)
// %p = gep half, float*3 %bc, i32 %idx
// Then PTM sets %bc → half, so typed bitcode sees: gep half, half*3 %bc, idx
//
// For device (AS1) pointers with i8 GEPs (from async copy byte offsets),
// convert to float-stride GEPs since all device pointers are float*.
static void fixGEPTypeMismatches(Module &M, PointeeTypeMap &PTM) {
  bool HasMMA = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      HasMMA = true;
  if (!HasMMA)
    return;

  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Type *SrcTy = GEP->getSourceElementType();
          if (SrcTy == FloatTy)
            continue;
          if (GEP->getNumIndices() != 1)
            continue;
          if (!SrcTy->isIntegerTy() && !SrcTy->isHalfTy() &&
              !SrcTy->isBFloatTy())
            continue;
          unsigned AS = GEP->getPointerAddressSpace();
          if (AS != metal::AS::Device && AS != metal::AS::Threadgroup)
            continue;
          // Don't float-ify a GEP whose pointer is a genuine integer buffer
          // (e.g. the i32 output of an int8 dot, whose f32 accumulator is
          // fptosi'd to i32). The collapse-to-float assumption holds only for
          // buffers fed to float MMA intrinsics; rewriting an i32 buffer's GEP
          // source to float leaves its pointee i32 → "Explicit gep type does
          // not match pointee type" → materializeAll failure.
          if (Type *Pointee = PTM.get(GEP->getPointerOperand()))
            if (Pointee->isIntegerTy() &&
                Pointee == GEP->getSourceElementType())
              continue;
          ToFix.push_back(GEP);
        }

    for (auto *GEP : ToFix) {
      Type *SrcTy = GEP->getSourceElementType();
      unsigned SrcSize = SrcTy->getPrimitiveSizeInBits();
      Value *Ptr = GEP->getPointerOperand();

      // Same-size types (e.g., i32 vs float, both 4 bytes):
      // Just change the GEP source element type to float. The stride is
      // identical so the pointer arithmetic doesn't change.
      if (SrcSize == 32) {
        GEP->setSourceElementType(FloatTy);
        GEP->setResultElementType(FloatTy);
        continue;
      }

      // Different-size types (half=16bit, i8=8bit vs float=32bit):
      // Insert identity bitcast before GEP to create a new pointer value
      // with the correct PTM entry. The bitcast is a no-op in opaque-ptr IR
      // but gives the serializer a different typed pointer for the GEP source.
      auto *BC = CastInst::Create(Instruction::BitCast, Ptr, Ptr->getType(), "",
                                  GEP->getIterator());
      GEP->setOperand(0, BC);
      PTM.set(BC, SrcTy);
    }
  }
}

// The mid-end's VectorCombine scalarizes wide loads into vector-typed
// two-index GEPs (`gep <N x T>, p, i, j`). The typed-pointer machinery has
// no notion of vector pointees and silently mis-addresses them; rewrite to
// the equivalent element-typed single-index GEP (`gep T, p, i*N+j`).
static void normalizeVectorGEPs(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
          if (isa<FixedVectorType>(GEP->getSourceElementType()) &&
              GEP->getNumIndices() == 2)
            ToFix.push_back(GEP);
    for (auto *GEP : ToFix) {
      auto *VT = cast<FixedVectorType>(GEP->getSourceElementType());
      IRBuilder<> B(GEP);
      Value *I0 = B.CreateSExtOrTrunc(GEP->getOperand(1), B.getInt64Ty());
      Value *I1 = B.CreateSExtOrTrunc(GEP->getOperand(2), B.getInt64Ty());
      Value *Lin =
          B.CreateAdd(B.CreateMul(I0, B.getInt64(VT->getNumElements())), I1);
      auto *NewGEP = cast<GetElementPtrInst>(
          B.CreateGEP(VT->getElementType(), GEP->getPointerOperand(), Lin));
      NewGEP->setIsInBounds(GEP->isInBounds());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
  }
}

// A pointer phi's record carries one pointee type; every incoming value must
// resolve to it. Globals (typed as their value type) and differently-typed
// GEP chains as incomings make the reader reject the record ("Invalid phi
// record"); wrap such incomings in an identity bitcast typed to the phi's
// pointee. The bitcast lands in the incoming block before its terminator.
static void fixPhiIncomingTypes(Module &M, PointeeTypeMap &PTM) {
  Type *FloatTy = Type::getFloatTy(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break; // phis are at block start
        if (!PN->getType()->isPointerTy())
          continue;
        // Only intervene when an incoming carries a CONCRETE pointee that the
        // phi record must be emitted against. If every incoming defaults to
        // the per-AS fallback (e.g. all-null, or args used only by the phi),
        // doing nothing keeps the phi, its incomings, and any consumer (e.g.
        // an insertelement into a <N x ptr>) all resolving to the same
        // default — touching it would create a spurious mismatch.
        Type *PhiPointee = PTM.get(PN);
        if (!PhiPointee) {
          for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
            Value *In = PN->getIncomingValue(J);
            if (isa<Constant>(In))
              continue;
            if (auto *GV = dyn_cast<GlobalVariable>(In))
              PhiPointee = GV->getValueType();
            else if (auto *G = dyn_cast<GetElementPtrInst>(In))
              PhiPointee = G->getResultElementType();
            else if (Type *T = PTM.get(In))
              PhiPointee = T;
            else
              continue;
            break;
          }
        }
        if (!PhiPointee)
          continue; // no concrete pointee anywhere — leave the phi alone
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          Value *In = PN->getIncomingValue(J);
          Type *InPointee = nullptr;
          if (isa<ConstantPointerNull>(In)) {
            // A typed null is emitted via SETTYPE against its own pointer
            // type's default pointee; when that disagrees with the phi
            // pointee the record is invalid. Replace with an inttoptr(0)
            // pinned to the phi pointee so the incoming is a real typed value.
            auto &Ctx = M.getContext();
            auto *Zero = ConstantInt::get(Type::getInt64Ty(Ctx), 0);
            auto *I2P = new IntToPtrInst(
                Zero, In->getType(), "",
                PN->getIncomingBlock(J)->getTerminator()->getIterator());
            PTM.set(I2P, PhiPointee);
            PN->setIncomingValue(J, I2P);
            continue;
          }
          if (isa<Constant>(In))
            continue; // other constants: leave untouched
          if (auto *GV = dyn_cast<GlobalVariable>(In))
            InPointee = GV->getValueType();
          else if (auto *G = dyn_cast<GetElementPtrInst>(In))
            InPointee = G->getResultElementType();
          else
            InPointee = PTM.get(In);
          if (InPointee == PhiPointee)
            continue;
          auto *BC = CastInst::Create(
              Instruction::BitCast, In, In->getType(), "",
              PN->getIncomingBlock(J)->getTerminator()->getIterator());
          PTM.set(BC, PhiPointee);
          PN->setIncomingValue(J, BC);
        }
        // Pin the phi's own pointee so the record's type index matches the
        // (now-consistent) incomings.
        PTM.set(PN, PhiPointee);
      }
  }
}

// Normalize the mid-end optimizer's byte-stride GEPs (`gep [N x i8]`,
// single index) against the typed pointee the PTM assigned to their base.
// When the pointee's alloc size equals N the rewrite to `gep <pointee>` is a
// pure retype (identical stride); otherwise an identity bitcast gives the
// GEP a base typed to its own source type. Without this the reader rejects
// the module ("Explicit gep type does not match pointee type of pointer
// operand"). Unlike fixGEPTypeMismatches this applies to every module, not
// just MMA users — elementwise kernels hit it too.
static void normalizeByteArrayGEPs(Module &M, PointeeTypeMap &PTM) {
  const DataLayout &DL = M.getDataLayout();
  // What the reader believes a base points to: for GEP-defined bases the
  // type is derived structurally from the producing GEP record; everything
  // else (args, call results) comes from the PTM-driven declarations.
  auto effectivePointee = [&](Value *Base) -> Type * {
    if (auto *G = dyn_cast<GetElementPtrInst>(Base))
      return G->getResultElementType();
    return PTM.get(Base);
  };
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // Rewriting a GEP changes the structural pointee its dependent GEPs see,
    // so iterate to a fixpoint (chains are shallow; converges in 2-3 rounds).
    bool Changed = true;
    while (Changed) {
      Changed = false;
      SmallVector<GetElementPtrInst *, 8> ToFix;
      for (auto &BB : F)
        for (auto &I : BB)
          if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
            if (GEP->getNumIndices() != 1)
              continue;
            Type *SrcTy = GEP->getSourceElementType();
            auto *AT = dyn_cast<ArrayType>(SrcTy);
            bool IsByteArray = AT && AT->getElementType()->isIntegerTy(8);
            // Two byte-stride forms: `[N x i8]` (index = element count) and
            // plain `i8` (index = byte offset).
            if (!IsByteArray && !SrcTy->isIntegerTy(8))
              continue;
            // Array globals are normalizeArrayGlobalGEPs' job (2-index form).
            if (isa<GlobalVariable>(GEP->getPointerOperand()))
              continue;
            Type *Pointee = effectivePointee(GEP->getPointerOperand());
            if (!Pointee || Pointee == SrcTy)
              continue;
            ToFix.push_back(GEP);
          }
      for (auto *GEP : ToFix) {
        Type *SrcTy = GEP->getSourceElementType();
        Type *Pointee = effectivePointee(GEP->getPointerOperand());
        bool PointeeOK = Pointee->isSized() && !Pointee->isAggregateType() &&
                         DL.getTypeAllocSize(Pointee) > 0;
        if (auto *AT = dyn_cast<ArrayType>(SrcTy)) {
          // [N x i8]: same stride as the pointee iff alloc sizes match.
          if (PointeeOK &&
              DL.getTypeAllocSize(Pointee) == AT->getNumElements()) {
            GEP->setSourceElementType(Pointee);
            GEP->setResultElementType(Pointee);
            Changed = true;
            continue;
          }
        } else {
          // i8: a constant byte offset divisible by the pointee size can be
          // rescaled into an element-typed GEP.
          auto *CI = dyn_cast<ConstantInt>(GEP->idx_begin()->get());
          uint64_t ESz = PointeeOK ? DL.getTypeAllocSize(Pointee) : 0;
          if (CI && ESz && CI->getSExtValue() % (int64_t)ESz == 0) {
            GEP->setSourceElementType(Pointee);
            GEP->setResultElementType(Pointee);
            GEP->setOperand(1,
                            ConstantInt::get(CI->getType(), CI->getSExtValue() /
                                                                (int64_t)ESz));
            Changed = true;
            continue;
          }
        }
        // Fallback: retype the base to the GEP's own source type.
        Value *Ptr = GEP->getPointerOperand();
        auto *BC = CastInst::Create(Instruction::BitCast, Ptr, Ptr->getType(),
                                    "", GEP->getIterator());
        GEP->setOperand(0, BC);
        PTM.set(BC, SrcTy);
        Changed = true;
      }
    }
  }
}

// INST_SELECT in the AIR-era encoding requires a scalar i1 condition (the
// O3 canonicalizes 3-way comparison idioms (e.g. the if/elif/else `-1/0/1`
// ladder a `tl.map_elementwise` callback expands to) into the `llvm.scmp` /
// `llvm.ucmp` intrinsics. The Metal AIR backend has no lowering for them, so
// the JIT fails with "Undefined symbols: llvm.scmp.*". Expand them inline:
// scmp(a,b) = zext(a > b) - zext(a < b) using signed/unsigned predicates per
// intrinsic, then sign-extended to the (possibly wider) result type.
static void lowerCmpIntrinsics(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 8> Calls;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Intrinsic::ID ID = CI->getIntrinsicID();
          if (ID == Intrinsic::scmp || ID == Intrinsic::ucmp)
            Calls.push_back(CI);
        }
    for (CallInst *CI : Calls) {
      bool Signed = CI->getIntrinsicID() == Intrinsic::scmp;
      IRBuilder<> B(CI);
      Value *A = CI->getArgOperand(0);
      Value *Bv = CI->getArgOperand(1);
      Value *Gt = Signed ? B.CreateICmpSGT(A, Bv) : B.CreateICmpUGT(A, Bv);
      Value *Lt = Signed ? B.CreateICmpSLT(A, Bv) : B.CreateICmpULT(A, Bv);
      Type *RetTy = CI->getType();
      Value *Res =
          B.CreateSub(B.CreateZExt(Gt, RetTy), B.CreateZExt(Lt, RetTy));
      CI->replaceAllUsesWith(Res);
      CI->eraseFromParent();
    }
  }
  // Drop the now-unused intrinsic declarations so no symbol is referenced.
  for (auto It = M.begin(); It != M.end();) {
    Function &F = *It++;
    Intrinsic::ID ID = F.getIntrinsicID();
    if ((ID == Intrinsic::scmp || ID == Intrinsic::ucmp) && F.use_empty())
      F.eraseFromParent();
  }
}

// O3's vectorizer can produce `<N x ptr addrspace(AS)>` values when it
// vectorizes a `tl.where`/select over pointer operands (e.g. the int-pointer
// payload of a masked gather). A vector-of-pointers POINTER element type has
// only one pointee slot per address space in the AIR type table, but the
// scalar pointers feeding the insertelement chain can carry conflicting
// pointees (an `i8` byte-array GEP base vs a `float`/`i64` typed null), so the
// emitted TYPE_CODE_VECTOR element disagrees with the scalar operands and the
// Metal reader rejects the module ("Invalid record"). These pointer vectors
// only ever exist to be `ptrtoint`'d to an integer vector and stored, so lower
// the whole `<N x ptr>` web to `<N x i64>` (pointer-width int): convert each
// scalar pointer operand with a scalar ptrtoint, keep the vector ops
// (insert/extract/phi/select) vectorized in integer space, and turn the
// trailing `ptrtoint <N x ptr>` into a plain truncation/passthrough. No
// pointee typing is needed for an integer vector, so the conflict disappears
// without scalarizing the bulk vector operations.
static void lowerVectorPointerToInt(Module &M) {
  auto isPtrVec = [](Type *T) -> FixedVectorType * {
    auto *VT = dyn_cast<FixedVectorType>(T);
    if (VT && VT->getElementType()->isPointerTy())
      return VT;
    return nullptr;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // Collect every instruction that produces a vector-of-pointers.
    SmallVector<Instruction *, 16> PtrVecDefs;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isPtrVec(I.getType()))
          PtrVecDefs.push_back(&I);
    if (PtrVecDefs.empty())
      continue;

    // Map each pointer-vector value to its integer-vector replacement.
    DenseMap<Value *, Value *> IntOf;
    auto intVecTy = [&](FixedVectorType *PVT) {
      return FixedVectorType::get(I64, PVT->getNumElements());
    };
    // Materialize the integer-vector form of an arbitrary pointer-vector
    // operand (constants/poison/undef and not-yet-rewritten defs).
    std::function<Value *(Value *, IRBuilder<> &)> asIntVec =
        [&](Value *V, IRBuilder<> &B) -> Value * {
      if (auto *It = IntOf.lookup(V))
        return It;
      auto *PVT = cast<FixedVectorType>(V->getType());
      if (isa<UndefValue>(V))
        return UndefValue::get(intVecTy(PVT));
      if (isa<ConstantAggregateZero>(V) ||
          (isa<Constant>(V) && cast<Constant>(V)->isNullValue()))
        return ConstantAggregateZero::get(intVecTy(PVT));
      // Fallback for any other constant/value: ptrtoint the whole vector.
      return B.CreatePtrToInt(V, intVecTy(PVT));
    };

    // First create placeholder integer phis so cycles resolve.
    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        IRBuilder<> B(PN);
        auto *NewPN =
            B.CreatePHI(intVecTy(cast<FixedVectorType>(PN->getType())),
                        PN->getNumIncomingValues());
        IntOf[PN] = NewPN;
      }

    // Rewrite the non-phi defs in program order.
    for (Instruction *I : PtrVecDefs) {
      if (isa<PHINode>(I))
        continue;
      IRBuilder<> B(I);
      Value *Repl = nullptr;
      if (auto *IE = dyn_cast<InsertElementInst>(I)) {
        Value *Vec = asIntVec(IE->getOperand(0), B);
        Value *Sc = B.CreatePtrToInt(IE->getOperand(1), I64);
        Repl = B.CreateInsertElement(Vec, Sc, IE->getOperand(2));
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(I)) {
        Value *A = asIntVec(SV->getOperand(0), B);
        Value *Bv = asIntVec(SV->getOperand(1), B);
        Repl = B.CreateShuffleVector(A, Bv, SV->getShuffleMask());
      } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
        Value *T = asIntVec(Sel->getTrueValue(), B);
        Value *Fv = asIntVec(Sel->getFalseValue(), B);
        Repl = B.CreateSelect(Sel->getCondition(), T, Fv);
      } else if (auto *BC = dyn_cast<BitCastInst>(I)) {
        // ptr-vec bitcast (e.g. addrspace-preserving): forward the int form.
        Repl = asIntVec(BC->getOperand(0), B);
      } else {
        // Unhandled producer: ptrtoint then back so users still see a ptr-vec.
        continue;
      }
      IntOf[I] = Repl;
    }

    // Fill phi incomings now that all defs have int forms.
    for (Instruction *I : PtrVecDefs)
      if (auto *PN = dyn_cast<PHINode>(I)) {
        auto *NewPN = cast<PHINode>(IntOf[PN]);
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          IRBuilder<> B(PN->getIncomingBlock(J)->getTerminator());
          NewPN->addIncoming(asIntVec(PN->getIncomingValue(J), B),
                             PN->getIncomingBlock(J));
        }
      }

    // Redirect users: consumers of the pointer-vector now read the int form.
    // ptrtoint <N x ptr>->ivec becomes the int form (with width fixups);
    // extractelement yields a scalar int turned back into a pointer;
    // everything else gets an inttoptr-rebuilt vector so it stays valid.
    for (Instruction *I : PtrVecDefs) {
      Value *Int = IntOf.lookup(I);
      if (!Int)
        continue;
      SmallVector<Use *, 8> Uses;
      for (Use &U : I->uses())
        Uses.push_back(&U);
      for (Use *U : Uses) {
        auto *User = cast<Instruction>(U->getUser());
        if (IntOf.count(User))
          continue; // already rewritten to consume the int form
        IRBuilder<> B(User);
        if (auto *P2I = dyn_cast<PtrToIntInst>(User)) {
          Value *V = Int;
          if (P2I->getType() != Int->getType())
            V = B.CreateZExtOrTrunc(Int, P2I->getType());
          P2I->replaceAllUsesWith(V);
          continue; // P2I now dead; cleaned up below
        }
        if (auto *EE = dyn_cast<ExtractElementInst>(User)) {
          Value *Sc = B.CreateExtractElement(Int, EE->getIndexOperand());
          EE->replaceAllUsesWith(B.CreateIntToPtr(Sc, EE->getType()));
          continue;
        }
        // Generic consumer still expecting a pointer vector: rebuild one.
        U->set(B.CreateIntToPtr(Int, I->getType()));
      }
    }

    // Erase the now-dead pointer-vector defs and orphaned ptrtoints.
    for (Instruction *I : reverse(PtrVecDefs)) {
      if (!IntOf.count(I))
        continue;
      if (!I->use_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
    }
    // Drop ptrtoint/extractelement consumers that were replaced.
    SmallVector<Instruction *, 8> Dead;
    for (auto &BB : F)
      for (auto &I : BB)
        if ((isa<PtrToIntInst>(I) || isa<ExtractElementInst>(I)) &&
            I.use_empty() && isPtrVec(I.getOperand(0)->getType()) &&
            IntOf.count(I.getOperand(0)))
          Dead.push_back(&I);
    for (Instruction *I : Dead)
      I->eraseFromParent();
    for (Instruction *I : reverse(PtrVecDefs))
      if (IntOf.count(I) && I->use_empty())
        I->eraseFromParent();
  }
}

// VSELECT form is rejected by the GPU JIT, see the SelectInst emission in
// FunctionWriter). The mid-end vectorizers produce vector-condition selects;
// scalarize them into per-lane extract/select/insert chains.
static void scalarizeVectorSelects(Module &M) {
  SmallVector<SelectInst *, 8> Sels;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Sel = dyn_cast<SelectInst>(&I))
          if (Sel->getCondition()->getType()->isVectorTy())
            Sels.push_back(Sel);
  for (auto *Sel : Sels) {
    IRBuilder<> B(Sel);
    auto *VT = cast<FixedVectorType>(Sel->getType());
    Value *Res = UndefValue::get(VT);
    for (unsigned L = 0; L < VT->getNumElements(); ++L) {
      Value *C = B.CreateExtractElement(Sel->getCondition(), B.getInt64(L));
      Value *T = B.CreateExtractElement(Sel->getTrueValue(), B.getInt64(L));
      Value *F = B.CreateExtractElement(Sel->getFalseValue(), B.getInt64(L));
      Res = B.CreateInsertElement(Res, B.CreateSelect(C, T, F), B.getInt64(L));
    }
    Sel->replaceAllUsesWith(Res);
    Sel->eraseFromParent();
  }
}

// The mid-end emits >64-bit integer arithmetic for overflow-free closed
// forms (e.g. SCEV's `trunc((zext(a) * zext(b)) >> 1)` triangular sums as
// i65). The AGX JIT cannot legalize any iN > 64; expand such chains into
// (lo, hi) i64 limb pairs. Unsupported wide ops fail loud.
static void expandWideIntegers(Module &M) {
  auto isWide = [](Type *T) {
    return T->isIntegerTy() && T->getIntegerBitWidth() > 64 &&
           T->getIntegerBitWidth() <= 128;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    DenseMap<Value *, std::pair<Value *, Value *>> Limbs; // wide -> (lo, hi)
    SmallVector<Instruction *, 8> Wide;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isWide(I.getType()) || llvm::any_of(I.operands(), [&](Value *Op) {
              return isWide(Op->getType());
            }))
          Wide.push_back(&I);
    if (Wide.empty())
      continue;
    auto umulh = [&](IRBuilder<> &B, Value *A, Value *Bv) -> Value * {
      Value *Mask = ConstantInt::get(I64, 0xffffffffull);
      Value *AL = B.CreateAnd(A, Mask), *AH = B.CreateLShr(A, 32);
      Value *BL = B.CreateAnd(Bv, Mask), *BH = B.CreateLShr(Bv, 32);
      Value *LL = B.CreateMul(AL, BL);
      Value *LH = B.CreateMul(AL, BH);
      Value *HL = B.CreateMul(AH, BL);
      Value *HH = B.CreateMul(AH, BH);
      Value *Mid =
          B.CreateAdd(B.CreateAdd(B.CreateLShr(LL, 32), B.CreateAnd(LH, Mask)),
                      B.CreateAnd(HL, Mask));
      return B.CreateAdd(B.CreateAdd(HH, B.CreateAdd(B.CreateLShr(LH, 32),
                                                     B.CreateLShr(HL, 32))),
                         B.CreateLShr(Mid, 32));
    };
    for (Instruction *I : Wide) {
      IRBuilder<> B(I);
      auto limbsOf = [&](Value *V) -> std::pair<Value *, Value *> {
        auto It = Limbs.find(V);
        if (It != Limbs.end())
          return It->second;
        if (auto *C = dyn_cast<ConstantInt>(V)) {
          APInt A = C->getValue();
          return {ConstantInt::get(I64, A.trunc(64)),
                  ConstantInt::get(I64, A.lshr(64).trunc(64))};
        }
        report_fatal_error("AIRWriter: unmapped wide integer operand");
      };
      if (auto *ZE = dyn_cast<ZExtInst>(I)) {
        Limbs[I] = {B.CreateZExtOrTrunc(ZE->getOperand(0), I64),
                    ConstantInt::get(I64, 0)};
      } else if (auto *SE2 = dyn_cast<SExtInst>(I)) {
        Value *Lo = B.CreateSExtOrTrunc(SE2->getOperand(0), I64);
        Limbs[I] = {Lo, B.CreateAShr(Lo, 63)};
      } else if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        auto [L1, H1] = limbsOf(BO->getOperand(0));
        switch (BO->getOpcode()) {
        case Instruction::Mul: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateMul(L1, L2);
          Value *Hi =
              B.CreateAdd(umulh(B, L1, L2), B.CreateAdd(B.CreateMul(L1, H2),
                                                        B.CreateMul(H1, L2)));
          Limbs[I] = {Lo, Hi};
          break;
        }
        case Instruction::Add: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateAdd(L1, L2);
          Value *Carry = B.CreateZExt(B.CreateICmpULT(Lo, L1), I64);
          Limbs[I] = {Lo, B.CreateAdd(B.CreateAdd(H1, H2), Carry)};
          break;
        }
        case Instruction::Sub: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateSub(L1, L2);
          Value *Borrow = B.CreateZExt(B.CreateICmpULT(L1, L2), I64);
          Limbs[I] = {Lo, B.CreateSub(B.CreateSub(H1, H2), Borrow)};
          break;
        }
        case Instruction::LShr: {
          auto *CI = dyn_cast<ConstantInt>(BO->getOperand(1));
          if (!CI)
            report_fatal_error("AIRWriter: wide lshr by non-constant");
          uint64_t Sh = CI->getZExtValue();
          if (Sh == 0) {
            Limbs[I] = {L1, H1};
          } else if (Sh < 64) {
            Limbs[I] = {
                B.CreateOr(B.CreateLShr(L1, Sh), B.CreateShl(H1, 64 - Sh)),
                B.CreateLShr(H1, Sh)};
          } else {
            Limbs[I] = {B.CreateLShr(H1, Sh - 64), ConstantInt::get(I64, 0)};
          }
          break;
        }
        default:
          report_fatal_error(Twine("AIRWriter: unhandled wide integer op '") +
                             BO->getOpcodeName() + "'");
        }
      } else if (auto *TR = dyn_cast<TruncInst>(I)) {
        auto [Lo, Hi] = limbsOf(TR->getOperand(0));
        (void)Hi;
        Value *R = B.CreateZExtOrTrunc(Lo, TR->getType());
        TR->replaceAllUsesWith(R);
      } else {
        report_fatal_error(Twine("AIRWriter: unhandled wide integer user '") +
                           I->getOpcodeName() + "'");
      }
    }
    for (auto It = Wide.rbegin(); It != Wide.rend(); ++It)
      (*It)->eraseFromParent();
  }
}

// The AGX JIT cannot legalize bool-vector <-> integer bitcasts (the
// optimizer's mask-packing idiom, `bitcast <N x i1> to iN`); expand into
// per-bit shifts.
static void scalarizeBoolVectorCasts(Module &M) {
  SmallVector<BitCastInst *, 8> Casts;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *BC = dyn_cast<BitCastInst>(&I)) {
          auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy());
          auto *DV = dyn_cast<FixedVectorType>(BC->getDestTy());
          if ((SV && SV->getElementType()->isIntegerTy(1) &&
               BC->getDestTy()->isIntegerTy(SV->getNumElements())) ||
              (DV && DV->getElementType()->isIntegerTy(1) &&
               BC->getSrcTy()->isIntegerTy(DV->getNumElements())))
            Casts.push_back(BC);
        }
  for (auto *BC : Casts) {
    IRBuilder<> B(BC);
    Value *R;
    if (auto *SV = dyn_cast<FixedVectorType>(BC->getSrcTy())) {
      R = ConstantInt::get(BC->getDestTy(), 0);
      for (unsigned L = 0; L < SV->getNumElements(); ++L) {
        Value *Bit = B.CreateZExt(
            B.CreateExtractElement(BC->getOperand(0), B.getInt64(L)),
            BC->getDestTy());
        R = B.CreateOr(R, B.CreateShl(Bit, L));
      }
    } else {
      auto *DV = cast<FixedVectorType>(BC->getDestTy());
      R = UndefValue::get(DV);
      for (unsigned L = 0; L < DV->getNumElements(); ++L) {
        Value *Bit =
            B.CreateTrunc(B.CreateLShr(BC->getOperand(0), L), B.getInt1Ty());
        R = B.CreateInsertElement(R, Bit, B.getInt64(L));
      }
    }
    BC->replaceAllUsesWith(R);
    BC->eraseFromParent();
  }
}

// `zext nneg` is definitionally equal to `sext`; emit the sext form. The
// AGX JIT widens zext-fed 64-bit multiplies into 65-bit operations it then
// fails to legalize (PSO abort "unable to legalize instruction ... s65"),
// while the sext form is its long-proven path.
static void canonicalizeNNegZExt(Module &M) {
  SmallVector<ZExtInst *, 16> Zexts;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *ZE = dyn_cast<ZExtInst>(&I))
          if (ZE->hasNonNeg())
            Zexts.push_back(ZE);
  for (auto *ZE : Zexts) {
    auto *SE = CastInst::Create(Instruction::SExt, ZE->getOperand(0),
                                ZE->getType(), "", ZE->getIterator());
    SE->takeName(ZE);
    ZE->replaceAllUsesWith(SE);
    ZE->eraseFromParent();
  }
}

// AIR v1 bitcode has no freeze opcode. Replacing freeze with its operand is
// a legal refinement (freeze only matters for poison/undef inputs, where any
// fixed value is a valid choice).
static void lowerFreezeInsts(Module &M) {
  SmallVector<FreezeInst *, 8> Frozen;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *FI = dyn_cast<FreezeInst>(&I))
          Frozen.push_back(FI);
  for (auto *FI : Frozen) {
    FI->replaceAllUsesWith(FI->getOperand(0));
    FI->eraseFromParent();
  }
}

// Give every simdgroup-matrix call a pointer operand whose typed pointee
// matches the intrinsic's element suffix. A float-typed TG/device pointer
// passed straight into a p3f16/p1f16 load (or any suffix mismatch) emits an
// invalid typed record (PSO "Failed to materializeAll"); the identity bitcast
// plus a PTM entry retypes it, same convention as fixGEPTypeMismatches.
static void scalarizeAggregateLoads(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<LoadInst *, 4> Aggs;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          if (isa<ArrayType>(LI->getType()))
            Aggs.push_back(LI);
    for (LoadInst *LI : Aggs) {
      auto *AT = cast<ArrayType>(LI->getType());
      Type *ElemTy = AT->getElementType();
      Value *Ptr = LI->getPointerOperand();
      Value *Agg = UndefValue::get(AT);
      IRBuilder<> B(LI);
      for (uint64_t E = 0; E < AT->getNumElements(); ++E) {
        Value *EP = B.CreateGEP(ElemTy, Ptr, B.getInt64(E));
        Value *EV = B.CreateLoad(ElemTy, EP);
        Agg = B.CreateInsertValue(Agg, EV, {unsigned(E)});
      }
      LI->replaceAllUsesWith(Agg);
      LI->eraseFromParent();
    }
  }
}

static void fixAccessTypeMismatch(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<Instruction *, 8> Fix;
    for (auto &BB : F)
      for (auto &I : BB) {
        Type *AccessTy = nullptr;
        Value *Ptr = nullptr;
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          AccessTy = LI->getType();
          Ptr = LI->getPointerOperand();
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          AccessTy = SI->getValueOperand()->getType();
          Ptr = SI->getPointerOperand();
        }
        if (!AccessTy || isa<BitCastInst>(Ptr))
          continue;
        // Vector accesses always get the retype (legacy behavior). Scalar
        // accesses only when the pointer's pointee provably disagrees — the
        // optimizer's folded byte-GEPs leave e.g. a float store through an
        // i8-typed pointer, which the reader rejects ("Explicit load/store
        // type does not match pointee type of pointer operand"). For a
        // GEP-defined pointer the reader derives the pointee structurally
        // from the GEP record, so compare against the GEP result type, not
        // the PTM entry (usage inference may already claim the access type).
        if (!AccessTy->isVectorTy()) {
          // inttoptr-derived pointers get their typed pointer from the
          // shared per-type default, which other values' inference can
          // claim first — always retype them to the access type.
          if (!isa<IntToPtrInst>(Ptr)) {
            Type *Pointee = nullptr;
            if (auto *G = dyn_cast<GetElementPtrInst>(Ptr))
              Pointee = G->getResultElementType();
            else
              Pointee = PTM.get(Ptr);
            if (!Pointee || Pointee == AccessTy)
              continue;
          }
        }
        Fix.push_back(&I);
      }
    for (Instruction *I : Fix) {
      Value *Ptr = isa<LoadInst>(I) ? cast<LoadInst>(I)->getPointerOperand()
                                    : cast<StoreInst>(I)->getPointerOperand();
      Type *AccessTy = isa<LoadInst>(I)
                           ? I->getType()
                           : cast<StoreInst>(I)->getValueOperand()->getType();
      auto *BC = CastInst::Create(Instruction::BitCast, Ptr, Ptr->getType(), "",
                                  I->getIterator());
      if (isa<LoadInst>(I))
        cast<LoadInst>(I)->setOperand(0, BC);
      else
        cast<StoreInst>(I)->setOperand(1, BC);
      PTM.set(BC, AccessTy);
    }
  }
}

static void fixMMAPointerSuffixMismatch(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 8> Calls;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() &&
              CI->getCalledFunction()->getName().starts_with(
                  "air.simdgroup_matrix_8x8_"))
            Calls.push_back(CI);
    for (auto *CI : Calls) {
      StringRef Name = CI->getCalledFunction()->getName();
      Type *Elem = nullptr;
      if (Name.contains("f16") && !Name.contains("bf16"))
        Elem = Type::getHalfTy(Ctx);
      else if (Name.contains("bf16"))
        Elem = Type::getBFloatTy(Ctx);
      else if (Name.contains("f32"))
        Elem = Type::getFloatTy(Ctx);
      if (!Elem)
        continue;
      for (unsigned J = 0; J < CI->arg_size(); J++) {
        Value *Op = CI->getArgOperand(J);
        if (!Op->getType()->isPointerTy())
          continue;
        if (Elem->isFloatTy() && !isa<Constant>(Op)) {
          // Float-suffix operands are normally float-typed already; wrap
          // only when the pointee provably disagrees (the optimizer's
          // byte-GEP chains leave i8-typed pointers feeding p1f32/p3f32).
          Type *Pointee = nullptr;
          if (auto *G = dyn_cast<GetElementPtrInst>(Op))
            Pointee = G->getResultElementType();
          else
            Pointee = PTM.get(Op);
          if (!Pointee || Pointee == Elem)
            continue;
        }
        if (isa<BitCastInst>(Op) || isa<AllocaInst>(Op))
          continue;
        auto *BC = CastInst::Create(Instruction::BitCast, Op, Op->getType(), "",
                                    CI->getIterator());
        CI->setArgOperand(J, BC);
        PTM.set(BC, Elem);
      }
    }
  }
}

// Fix air.arg_type_name / air.arg_type_size in kernel metadata to match
// actual parameter pointee types from PTM. The transform pipeline may set
// all buffer type names to "float" even when the actual type is bfloat/char.
// Metal GPU JIT validates these metadata entries against the bitcode types.
static void fixKernelArgMetadata(Module &M, const PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto *AirKernel = M.getNamedMetadata("air.kernel");
  if (!AirKernel)
    return;

  for (unsigned K = 0; K < AirKernel->getNumOperands(); K++) {
    auto *KernelMD = AirKernel->getOperand(K);
    if (KernelMD->getNumOperands() < 3)
      continue;
    // KernelMD: {fn, attrs, argDescs}
    auto *ArgDescs = dyn_cast_or_null<MDNode>(KernelMD->getOperand(2));
    if (!ArgDescs)
      continue;

    auto *FnVAM = dyn_cast_or_null<ValueAsMetadata>(KernelMD->getOperand(0));
    if (!FnVAM)
      continue;
    auto *Fn = dyn_cast<Function>(FnVAM->getValue());
    if (!Fn)
      continue;

    for (unsigned A = 0; A < ArgDescs->getNumOperands(); A++) {
      auto *ArgMD = dyn_cast_or_null<MDNode>(ArgDescs->getOperand(A));
      if (!ArgMD || ArgMD->getNumOperands() < 2)
        continue;

      // Check if this is a buffer arg (has "air.buffer" string)
      bool IsBuffer = false;
      for (unsigned I = 1; I < ArgMD->getNumOperands(); I++)
        if (auto *S = dyn_cast_or_null<MDString>(ArgMD->getOperand(I)))
          if (S->getString() == "air.buffer") {
            IsBuffer = true;
            break;
          }
      if (!IsBuffer)
        continue;

      // Get the arg index from the first operand
      auto *IdxVAM = dyn_cast_or_null<ValueAsMetadata>(ArgMD->getOperand(0));
      if (!IdxVAM)
        continue;
      auto *IdxCI = dyn_cast<ConstantInt>(IdxVAM->getValue());
      if (!IdxCI)
        continue;
      unsigned ArgIdx = IdxCI->getZExtValue();
      if (ArgIdx >= Fn->arg_size())
        continue;

      // Infer pointee type from PTM, following through bitcasts
      Argument *Arg = Fn->getArg(ArgIdx);
      if (!Arg->getType()->isPointerTy())
        continue;
      Type *Pointee = nullptr;
      if (auto *Ty = PTM.get(Arg))
        Pointee = Ty;
      if (!Pointee)
        Pointee = PointeeTypeMap::inferFromUsage(Arg);
      // Follow through bitcasts if inference failed on the arg directly
      if (!Pointee || Pointee->isFloatTy()) {
        for (auto *U : Arg->users()) {
          if (auto *BC = dyn_cast<BitCastInst>(U)) {
            if (auto *Ty = PTM.get(BC)) {
              if (!Ty->isFloatTy()) {
                Pointee = Ty;
                break;
              }
            }
            Type *BcTy = PointeeTypeMap::inferFromUsage(BC);
            if (BcTy && !BcTy->isFloatTy()) {
              Pointee = BcTy;
              break;
            }
          }
        }
      }
      if (!Pointee)
        continue;

      // Determine correct type name, size, and alignment
      StringRef TypeName;
      unsigned TypeSize = 0, TypeAlign = 0;
      if (Pointee->isBFloatTy()) {
        TypeName = "bfloat";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isFloatTy()) {
        TypeName = "float";
        TypeSize = 4;
        TypeAlign = 4;
      } else if (Pointee->isHalfTy()) {
        TypeName = "half";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(8)) {
        TypeName = "char";
        TypeSize = 1;
        TypeAlign = 1;
      } else if (Pointee->isIntegerTy(16)) {
        TypeName = "short";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(32)) {
        TypeName = "int";
        TypeSize = 4;
        TypeAlign = 4;
      } else {
        continue; // Unknown type, don't change
      }

      // Rebuild the metadata node with corrected values
      SmallVector<Metadata *, 16> NewOps;
      for (unsigned I = 0; I < ArgMD->getNumOperands(); I++) {
        Metadata *Op = ArgMD->getOperand(I);
        if (I + 1 < ArgMD->getNumOperands()) {
          if (auto *PrevS = dyn_cast_or_null<MDString>(ArgMD->getOperand(I))) {
            if (PrevS->getString() == "air.arg_type_name" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(MDString::get(Ctx, TypeName));
              I++; // skip original type name
              continue;
            }
            if (PrevS->getString() == "air.arg_type_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeSize)));
              I++; // skip original size
              continue;
            }
            if (PrevS->getString() == "air.arg_type_align_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeAlign)));
              I++; // skip original align
              continue;
            }
          }
        }
        NewOps.push_back(Op);
      }
      auto *NewArgMD = MDNode::get(Ctx, NewOps);
      ArgDescs->replaceOperandWith(A, NewArgMD);
    }
  }
}

// Map LLVM's in-memory AttrKind to the AIR-v1 bitcode attr-kind encoding.
// The two numbering spaces diverged long ago, and the mid-end optimizer
// attaches modern attributes (captures, noundef, memory, ...) that Apple's
// reader rejects as "Unknown attribute kind" — anything outside this
// whitelist is dropped (attributes are hints; dropping is always sound).
static std::optional<uint64_t> airEnumAttrKind(Attribute::AttrKind K) {
  switch (K) {
  case Attribute::NoAlias:
    return 9;
  case Attribute::NoUnwind:
    return 18;
  case Attribute::ReadNone:
    return 20;
  case Attribute::ReadOnly:
    return 21;
  case Attribute::NonNull:
    return 26;
  case Attribute::Convergent:
    return 43;
  case Attribute::WriteOnly:
    return 52;
  case Attribute::WillReturn:
    return 61;
  case Attribute::NoFree:
    return 62;
  case Attribute::NoSync:
    return 63;
  case Attribute::MustProgress:
    return 70;
  default:
    return std::nullopt;
  }
}

static std::optional<uint64_t> airIntAttrKind(Attribute::AttrKind K) {
  switch (K) {
  case Attribute::Alignment:
    return 1;
  case Attribute::Dereferenceable:
    return 41;
  case Attribute::DereferenceableOrNull:
    return 42;
  default:
    return std::nullopt;
  }
}

// Append the AIR-bitcode encoding of Attr to Grp (no-op for attributes
// outside the whitelist). Returns true if the attribute was encoded.
// captures(none) maps back to the legacy valueless nocapture kind.
static bool encodeAirAttr(const Attribute &Attr,
                          SmallVectorImpl<uint64_t> *Grp) {
  if (Attr.isEnumAttribute()) {
    if (auto BK = airEnumAttrKind(Attr.getKindAsEnum())) {
      if (Grp) {
        Grp->push_back(0);
        Grp->push_back(*BK);
      }
      return true;
    }
    return false;
  }
  if (Attr.isIntAttribute()) {
    if (Attr.getKindAsEnum() == Attribute::Captures) {
      if (!capturesNothing(Attr.getCaptureInfo()))
        return false;
      if (Grp) {
        Grp->push_back(0);
        Grp->push_back(11); // nocapture
      }
      return true;
    }
    if (auto BK = airIntAttrKind(Attr.getKindAsEnum())) {
      if (Grp) {
        Grp->push_back(1);
        Grp->push_back(*BK);
        Grp->push_back(Attr.getValueAsInt());
      }
      return true;
    }
    return false;
  }
  return false;
}

// True if at least one attribute in AS survives the AIR whitelist (string
// attributes always pass through).
static bool hasAirAttrs(const AttributeSet &AS) {
  for (Attribute Attr : AS) {
    if (Attr.isStringAttribute())
      return true;
    if (encodeAirAttr(Attr, nullptr))
      return true;
  }
  return false;
}

// Forward declarations (defined in separate .cpp files)
void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E);
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                        ArrayRef<const Constant *> Constants,
                        unsigned CodeSize);
void emitMetadataKindBlock(BitstreamWriter &W);
void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E,
                       MetadataEnumerator &MD);
void emitOperandBundleTagsBlock(BitstreamWriter &W);
void emitSinglethreadBlock(BitstreamWriter &W);
void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                       ValueEnumerator &E, const MetadataEnumerator &MD);

// Remove truly redundant ptr-to-ptr bitcasts where the PTM has the SAME
// pointee type on both sides. Bitcasts that serve as typed pointer
// transitions (where PTM records different types) must be kept.
// Only removes bitcasts where BOTH sides have the same PTM entry.
static void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<BitCastInst *, 16> ToRemove;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || BC->getSrcTy() != BC->getDestTy())
          continue;
        // Only remove if BOTH sides have the same PTM-recorded type.
        // If either side has no PTM entry, keep the bitcast (it may serve
        // as a type transition that the serializer needs).
        Type *SrcPT = PTM.get(BC->getOperand(0));
        Type *DstPT = PTM.get(BC);
        if (!SrcPT || !DstPT)
          continue; // Unknown - keep
        if (SrcPT != DstPT)
          continue; // Different - keep
        ToRemove.push_back(BC);
      }
    }
    for (auto *BC : ToRemove) {
      PTM.remove(BC);
      BC->replaceAllUsesWith(BC->getOperand(0));
      BC->eraseFromParent();
    }
  }
}

// Normalize single-index element GEPs on array globals into 2-index array
// GEPs. A global variable's typed-pointer pointee is its array value type
// (e.g. [2 x i32]); a GEP that indexes it as `gep i32, @GV, %idx` carries an
// explicit source element type (i32) that disagrees with that pointee, which
// the Metal AIR reader rejects with "Explicit gep type does not match pointee
// type of pointer operand". Rewriting to `gep [N x i32], @GV, 0, %idx` makes
// the GEP source element type equal the global pointee. Handles both the
// element-typed index (stride = element size) and the byte-typed constant
// index (i8 source with a constant byte offset that is a multiple of the
// element size). Runs after lowerConstantExprs so constant-expr GEPs that
// were materialized as instructions are covered too.
static void normalizeArrayGlobalGEPs(Module &M) {
  Type *I64Ty = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    // Loads/stores directly through an array global (no GEP, e.g. after a
    // single-incoming phi folds away) have element access type vs array
    // pointee; route them through an element-0 GEP.
    SmallVector<Instruction *, 4> DirectAccess;
    for (auto &BB : F)
      for (auto &I : BB) {
        Value *Ptr = nullptr;
        if (auto *LI = dyn_cast<LoadInst>(&I))
          Ptr = LI->getPointerOperand();
        else if (auto *SI = dyn_cast<StoreInst>(&I))
          Ptr = SI->getPointerOperand();
        if (!Ptr)
          continue;
        auto *GV = dyn_cast<GlobalVariable>(Ptr);
        if (!GV || !isa<ArrayType>(GV->getValueType()))
          continue;
        DirectAccess.push_back(&I);
      }
    for (Instruction *I : DirectAccess) {
      auto *GV = cast<GlobalVariable>(
          isa<LoadInst>(I) ? cast<LoadInst>(I)->getPointerOperand()
                           : cast<StoreInst>(I)->getPointerOperand());
      auto *Base = GetElementPtrInst::CreateInBounds(
          GV->getValueType(), GV,
          {ConstantInt::get(I64Ty, 0), ConstantInt::get(I64Ty, 0)});
      Base->insertBefore(I->getIterator());
      if (isa<LoadInst>(I))
        cast<LoadInst>(I)->setOperand(0, Base);
      else
        cast<StoreInst>(I)->setOperand(1, Base);
    }
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (GEP->getNumIndices() != 1)
            continue;
          auto *GV = dyn_cast<GlobalVariable>(GEP->getPointerOperand());
          if (!GV)
            continue;
          auto *AT = dyn_cast<ArrayType>(GV->getValueType());
          if (!AT)
            continue;
          if (GEP->getSourceElementType() == AT)
            continue;
          ToFix.push_back(GEP);
        }

    for (auto *GEP : ToFix) {
      auto *GV = cast<GlobalVariable>(GEP->getPointerOperand());
      auto *AT = cast<ArrayType>(GV->getValueType());
      Type *ElemTy = AT->getElementType();
      Type *SrcTy = GEP->getSourceElementType();
      Value *Idx = GEP->idx_begin()->get();

      Value *ElemIdx = nullptr;
      if (SrcTy == ElemTy) {
        ElemIdx = Idx;
      } else {
        uint64_t SrcSize = M.getDataLayout().getTypeAllocSize(SrcTy);
        uint64_t ElemSize = M.getDataLayout().getTypeAllocSize(ElemTy);
        auto *CI = dyn_cast<ConstantInt>(Idx);
        if (!CI || ElemSize == 0)
          continue;
        uint64_t ByteOff = CI->getZExtValue() * SrcSize;
        if (ByteOff % ElemSize != 0)
          continue;
        ElemIdx = ConstantInt::get(I64Ty, ByteOff / ElemSize);
      }

      auto *NewGEP = GetElementPtrInst::Create(
          AT, GV, {ConstantInt::get(I64Ty, 0), ElemIdx}, "",
          GEP->getIterator());
      NewGEP->setIsInBounds(GEP->isInBounds());
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
  }
}

std::vector<uint8_t> emitMetalBitcode(Module &M, PointeeTypeMap &PTM) {
  SmallVector<char, 0> Buf;
  // Scope the writer so its destructor runs FlushToWord() before Buf is read.
  // FlushToWord() pads the final partial 32-bit word; without it the bitstream
  // ends short of the length its own block-length fields declare, so readers
  // (llvm-dis, metal-objdump) over-read and report "truncated module".
  {
    BitstreamWriter W(Buf);

    // BC magic
    W.Emit('B', 8);
    W.Emit('C', 8);
    W.Emit(0xC0, 8);
    W.Emit(0xDE, 8);

    // IDENTIFICATION
    W.EnterSubblock(bitc::IDENTIFICATION_BLOCK_ID, 5);
    emitString(W, bitc::IDENTIFICATION_CODE_STRING, "MetalIR");
    {
      SmallVector<uint64_t, 1> V = {0};
      W.EmitRecord(bitc::IDENTIFICATION_CODE_EPOCH, V);
    }
    W.ExitBlock();

    // Pre-serialization IR fixups (these helpers refine the PTM in place).
    expandWideIntegers(M);
    lowerFreezeInsts(M);
    canonicalizeNNegZExt(M);
    scalarizeBoolVectorCasts(M);
    lowerCmpIntrinsics(M);
    lowerVectorPointerToInt(M);
    scalarizeVectorSelects(M);
    removeRedundantBitcasts(M, PTM);
    normalizeVectorGEPs(M);
    fixGEPTypeMismatches(M, PTM);
    normalizeByteArrayGEPs(M, PTM);
    fixPhiIncomingTypes(M, PTM);
    fixMMAPointerSuffixMismatch(M, PTM);
    scalarizeAggregateLoads(M);
    fixAccessTypeMismatch(M, PTM);

    // Lower ConstantExpr operands to real instructions before enumeration.
    lowerConstantExprs(M);

    // Normalize element-typed GEPs on array globals to 2-index array GEPs so
    // the GEP source type matches the global's typed-pointer pointee.
    normalizeArrayGlobalGEPs(M);

    // Fix kernel argument metadata to match actual pointee types.
    fixKernelArgMetadata(M, PTM);

    if (getenv("METAL_DUMP_PREWRITE"))
      M.print(llvm::errs(), nullptr);

    ValueEnumerator E(M, PTM);

    // MODULE_BLOCK (CodeSize=4)
    W.EnterSubblock(bitc::MODULE_BLOCK_ID, 4);

    {
      SmallVector<uint64_t, 1> V = {1};
      W.EmitRecord(bitc::MODULE_CODE_VERSION, V);
    }

    // PARAMATTR blocks BEFORE TYPE_BLOCK (Metal requires this order).
    //
    // E2c — general path: walk every Function, collect its parameter
    // AttributeSets and emit one PARAMATTR_GRP per unique (paramIdx, AS) tuple
    // plus one PARAMATTR list per function. Group ID 1 is reserved for the
    // legacy MMA-load nocapture+readonly entry, which lives on the call site
    // (the CallInst's first paramattr operand), not the function declaration.
    struct GroupKey {
      unsigned ListIdx;
      AttributeSet AS;
      bool operator==(const GroupKey &O) const {
        return ListIdx == O.ListIdx && AS == O.AS;
      }
    };
    struct GroupKeyInfo {
      static GroupKey getEmptyKey() {
        return {~0u, DenseMapInfo<AttributeSet>::getEmptyKey()};
      }
      static GroupKey getTombstoneKey() {
        return {~0u - 1, DenseMapInfo<AttributeSet>::getTombstoneKey()};
      }
      static unsigned getHashValue(const GroupKey &K) {
        return hash_combine(K.ListIdx,
                            DenseMapInfo<AttributeSet>::getHashValue(K.AS));
      }
      static bool isEqual(const GroupKey &A, const GroupKey &B) {
        return A.ListIdx == B.ListIdx && A.AS == B.AS;
      }
    };

    DenseMap<GroupKey, unsigned, GroupKeyInfo> GroupID;
    SmallVector<GroupKey, 8> GroupOrder;
    auto getGroupID = [&](unsigned ListIdx, AttributeSet AS) -> unsigned {
      GroupKey K{ListIdx, AS};
      auto It = GroupID.find(K);
      if (It != GroupID.end())
        return It->second;
      unsigned ID = GroupID.size() + 1;
      GroupID[K] = ID;
      GroupOrder.push_back(K);
      return ID;
    };

    DenseMap<const Function *, unsigned> FnAttrListID;
    SmallVector<SmallVector<unsigned, 4>, 8> AttrLists;

    // Synthesize attribute groups on the simdgroup-matrix intrinsic
    // declarations to match Apple's `xcrun metal` AIR — the macOS 13/14/15
    // Metal driver rejects the metallib otherwise. The groups must be encoded
    // with bitcode ATTR_KIND_* values, which do NOT match LLVM's in-memory
    // Attribute::AttrKind enum (e.g. convergent is 6 in-memory but 43 in
    // bitcode).
    enum : uint64_t {
      BK_NO_CAPTURE = 11,
      BK_NO_UNWIND = 18,
      BK_READ_ONLY = 21,
      BK_CONVERGENT = 43,
      BK_WRITEONLY = 52,
      BK_WILLRETURN = 61,
      BK_NOFREE = 62,
      BK_MUSTPROGRESS = 70,
    };
    // A synthesized group: target index (~0u=function, N=param N) + bitcode
    // enum attr-kinds.
    struct SynthGroup {
      uint64_t Index; // ~0u for function attrs, param index (1-based) otherwise
      SmallVector<uint64_t, 6> EnumKinds;
    };
    // Pending synthesized groups, in emission order, with their assigned IDs.
    SmallVector<std::pair<unsigned, SynthGroup>, 8> SynthGroups;
    // Functions that should be marked local_unnamed_addr in their record.
    DenseSet<const Function *> LocalUnnamedFns;
    auto addSynthGroup = [&](SynthGroup G) -> unsigned {
      unsigned ID = GroupID.size() + 1 + SynthGroups.size();
      SynthGroups.push_back({ID, std::move(G)});
      return ID;
    };

    bool HasMMALoad = false;
    for (auto &F : M) {
      if (F.getName().starts_with("air.simdgroup_matrix_8x8_load")) {
        HasMMALoad = true;
        break;
      }
    }
    if (HasMMALoad) {
      // Reserve group ID 1 + list ID 1 for the MMA call-site paramattr (the
      // FunctionWriter unconditionally threads `paramattr=1` into those calls).
      GroupKey K{/*ListIdx=*/1, AttributeSet()};
      GroupID[K] = 1;
      GroupOrder.push_back(K);
    }

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      AttributeList AL = F.getAttributes();
      SmallVector<unsigned, 4> GroupIDs;
      for (unsigned i = 0; i < F.arg_size(); i++) {
        AttributeSet AS = AL.getParamAttrs(i);
        if (!AS.hasAttributes() || !hasAirAttrs(AS))
          continue;
        GroupIDs.push_back(getGroupID(i + 1, AS));
      }
      AttributeSet RetAS = AL.getRetAttrs();
      if (RetAS.hasAttributes() && hasAirAttrs(RetAS))
        GroupIDs.push_back(getGroupID(0, RetAS));
      if (GroupIDs.empty())
        continue;
      unsigned ListID = AttrLists.size() + 1;
      AttrLists.push_back(std::move(GroupIDs));
      FnAttrListID[&F] = ListID;
    }

    // Now synthesize declaration attribute groups for the simdgroup intrinsics.
    for (auto &F : M) {
      if (!F.isDeclaration())
        continue;
      StringRef Name = F.getName();
      if (!Name.starts_with("air.simdgroup_matrix_8x8_"))
        continue;
      bool IsLoad = Name.contains("_load");
      bool IsStore = Name.contains("_store");

      // Function-attr group (matches Apple). Order matches Apple's textual AIR.
      SynthGroup FnG;
      FnG.Index = ~0u;
      FnG.EnumKinds.push_back(BK_CONVERGENT);
      FnG.EnumKinds.push_back(BK_MUSTPROGRESS);
      if (IsLoad)
        FnG.EnumKinds.push_back(BK_NOFREE);
      FnG.EnumKinds.push_back(BK_NO_UNWIND);
      if (IsLoad)
        FnG.EnumKinds.push_back(BK_READ_ONLY);
      FnG.EnumKinds.push_back(BK_WILLRETURN);
      if (IsStore)
        FnG.EnumKinds.push_back(BK_WRITEONLY);
      unsigned FnGID = addSynthGroup(std::move(FnG));

      SmallVector<unsigned, 4> GroupIDs;
      GroupIDs.push_back(FnGID);

      // Pointer-arg group: load ptr is arg 0, store ptr is arg 1
      // (`nocapture readonly` / `nocapture writeonly`).
      if (IsLoad || IsStore) {
        SynthGroup PG;
        PG.Index = IsStore ? 2u : 1u; // 1-based param index
        PG.EnumKinds.push_back(BK_NO_CAPTURE);
        PG.EnumKinds.push_back(IsStore ? BK_WRITEONLY : BK_READ_ONLY);
        GroupIDs.push_back(addSynthGroup(std::move(PG)));
      }

      unsigned ListID = AttrLists.size() + 1;
      AttrLists.push_back(std::move(GroupIDs));
      FnAttrListID[&F] = ListID;
      LocalUnnamedFns.insert(&F);
    }

    if (!GroupID.empty() || !SynthGroups.empty()) {
      W.EnterSubblock(bitc::PARAMATTR_GROUP_BLOCK_ID, 4);
      for (auto &K : GroupOrder) {
        unsigned ID = GroupID.lookup(K);
        SmallVector<uint64_t, 16> Grp;
        Grp.push_back(ID);
        Grp.push_back(K.ListIdx);
        if (HasMMALoad && ID == 1 && K.AS.getNumAttributes() == 0) {
          // Legacy MMA call-site group: param 1 nocapture + readonly.
          Grp.push_back(0);
          Grp.push_back(11);
          Grp.push_back(0);
          Grp.push_back(21);
        } else {
          for (Attribute Attr : K.AS) {
            if (Attr.isStringAttribute()) {
              StringRef Key = Attr.getKindAsString();
              StringRef Val = Attr.getValueAsString();
              // 3 = string-key-only, 4 = key + value (matches upstream
              // writeAttributeGroupTable encoding).
              Grp.push_back(Val.empty() ? 3 : 4);
              for (char C : Key)
                Grp.push_back((unsigned char)C);
              Grp.push_back(0);
              if (!Val.empty()) {
                for (char C : Val)
                  Grp.push_back((unsigned char)C);
                Grp.push_back(0);
              }
            } else {
              encodeAirAttr(Attr, &Grp);
            }
          }
        }
        W.EmitRecord(bitc::PARAMATTR_GRP_CODE_ENTRY, Grp);
      }
      // Synthesized declaration groups (simdgroup-matrix intrinsics). Each
      // entry is (ID, Index, [0, kind]...) — every attr here is a plain bitcode
      // enum (record-code 0), with the bitcode ATTR_KIND_* values set above.
      for (auto &IDG : SynthGroups) {
        SmallVector<uint64_t, 16> Grp;
        Grp.push_back(IDG.first);        // group ID
        Grp.push_back(IDG.second.Index); // ~0u = function, else param index
        for (uint64_t Kind : IDG.second.EnumKinds) {
          Grp.push_back(0); // 0 = well-known enum attribute (no value)
          Grp.push_back(Kind);
        }
        W.EmitRecord(bitc::PARAMATTR_GRP_CODE_ENTRY, Grp);
      }
      W.ExitBlock();

      W.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 4);
      if (HasMMALoad) {
        SmallVector<uint64_t, 2> List;
        List.push_back(1);
        W.EmitRecord(2, List);
      }
      for (auto &AL : AttrLists) {
        SmallVector<uint64_t, 8> List;
        for (unsigned ID : AL)
          List.push_back(ID);
        W.EmitRecord(2, List);
      }
      W.ExitBlock();
    }

    emitTypeBlock(W, E);

    // Emit target triple - Metal GPU JIT expects it for proper codegen.
    // Use module value if set, otherwise default Metal AIR triple.
    {
      std::string T = M.getTargetTriple().str();
      if (T.empty() || T == "air")
        // Derive the canonical AIR triple from whatever OS info is present
        // (falls back to macOS 16 / 26-era), rather than hardcoding it.
        T = MetalVersion::fromTriple(M.getTargetTriple().str()).tripleString();
      emitString(W, bitc::MODULE_CODE_TRIPLE, T);
    }
    // Emit data layout - Metal GPU JIT uses this for type size/alignment.
    {
      auto DLStr = M.getDataLayoutStr();
      if (!DLStr.empty()) {
        emitString(W, bitc::MODULE_CODE_DATALAYOUT, DLStr);
      } else {
        emitString(W, bitc::MODULE_CODE_DATALAYOUT,
                   "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                   "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32"
                   "-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
                   "-v192:256:256-v256:256:256-v512:512:512"
                   "-v1024:1024:1024-n8:16:32");
      }
    }

    if (!M.getSourceFileName().empty())
      emitString(W, bitc::MODULE_CODE_SOURCE_FILENAME, M.getSourceFileName());

    // GLOBALVAR and FUNCTION records - emit in globalValues order
    // (globals first, then functions, matching value ID assignment)
    for (auto *V : E.globalValues) {
      if (auto *G = dyn_cast<GlobalVariable>(V)) {
        SmallVector<uint64_t, 14> Ops;
        Ops.push_back(E.globalPtrTypeIdx(G)); // ptr-to-valueType
        Ops.push_back(G->isConstant() ? 1 : 0);
        Ops.push_back(G->hasInitializer()
                          ? E.moduleConstIdx(G->getInitializer()) + 1
                          : 0);
        Ops.push_back(encodeLinkage(G->getLinkage()));
        Ops.push_back(G->getAlign() ? Log2_32(G->getAlign()->value()) + 1 : 0);
        for (int J = 0; J < 3; J++)
          Ops.push_back(0);
        Ops.push_back(G->hasGlobalUnnamedAddr() ? 1 : 0);
        Ops.push_back(G->isExternallyInitialized() ? 1 : 0);
        Ops.push_back(0);
        Ops.push_back(0);
        Ops.push_back(G->getAddressSpace());
        Ops.push_back(0);
        W.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Ops);
      } else if (auto *Fn = dyn_cast<Function>(V)) {
        SmallVector<uint64_t, 17> Ops;
        Ops.push_back(E.typeIdx(Fn->getFunctionType()));
        Ops.push_back(Fn->getCallingConv());
        Ops.push_back(Fn->isDeclaration() ? 1 : 0);
        Ops.push_back(encodeLinkage(Fn->getLinkage()));
        // paramattr: attribute-list ID for this function (0 = none).
        // Per-function list IDs start at 2 when HasMMALoad reserves list 1
        // for the MMA call site, else at 1.
        unsigned ListID = 0;
        auto It = FnAttrListID.find(Fn);
        if (It != FnAttrListID.end())
          ListID = It->second + (HasMMALoad ? 1 : 0);
        Ops.push_back(ListID);
        Ops.push_back(0); // align
        // Function record fields 6..15: section, visibility, gc, unnamed_addr,
        // prologuedata, dllstorageclass, comdat, prefixdata, personalityfn, ...
        // Field 9 is unnamed_addr. Bitcode encoding (getEncodedUnnamedAddr):
        // None=0, Global=1, Local=2. Apple's simdgroup intrinsic decls are
        // `local_unnamed_addr` (=2); the macOS-14 driver expects this to match.
        Ops.push_back(0); // 6: section
        Ops.push_back(0); // 7: visibility
        Ops.push_back(0); // 8: gc
        Ops.push_back(LocalUnnamedFns.contains(Fn) ? 2u
                                                   : 0u); // 9: unnamed_addr
        for (int J = 10; J < 16; J++)
          Ops.push_back(0);
        Ops.push_back(Fn->getAddressSpace());
        W.EmitRecord(bitc::MODULE_CODE_FUNCTION, Ops);
      }
    }

    emitConstantsBlock(W, E, E.moduleConstants, 5);
    emitMetadataKindBlock(W);

    // Share one MetadataEnumerator between the module-level METADATA_BLOCK
    // (where the nodes are emitted) and per-function attachment blocks
    // (where they are referenced by ID).
    MetadataEnumerator MDEnum;
    MDEnum.collect(M, E);
    emitMetadataBlock(W, M, E, MDEnum);
    emitOperandBundleTagsBlock(W);
    emitSinglethreadBlock(W);

    for (auto *V : E.globalValues)
      if (auto *F = dyn_cast<Function>(V))
        if (!F->isDeclaration())
          emitFunctionBlock(W, *F, E, MDEnum);

    // VALUE_SYMTAB
    W.EnterSubblock(bitc::VALUE_SYMTAB_BLOCK_ID, 4);
    for (unsigned I = 0; I < E.globalValues.size(); I++) {
      if (!E.globalValues[I]->hasName())
        continue;
      SmallVector<uint64_t, 32> NV;
      NV.push_back(I);
      for (char C : E.globalValues[I]->getName())
        NV.push_back((uint64_t)(unsigned char)C);
      W.EmitRecord(bitc::VST_CODE_ENTRY, NV);
    }
    W.ExitBlock();

    W.ExitBlock(); // MODULE_BLOCK
  } // ~BitstreamWriter flushes the final word into Buf here.

  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metal
} // namespace llvm
