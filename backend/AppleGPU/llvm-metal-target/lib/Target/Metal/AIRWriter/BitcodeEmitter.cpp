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
#include "AggregateScalarize.h"
#include "BitcodeEncoding.h"
#include "ConstantExprLower.h"
#include "CoopTensorLowering.h"
#include "LowerPointerVectors.h"
#include "LowerVectorSelect.h"
#include "MetadataWriter.h"
#include "MetalConstraints.h"
#include "MetalVersion.h"
#include "NormalizeGEPs.h"
#include "PointerPointeeRepair.h"
#include "PointerRepairUtil.h"
#include "ValueEnumerator.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/IRSymtab.h"
#include "llvm/Support/Allocator.h"
#include <map>

using namespace llvm;

namespace llvm {
namespace metal {

// True if a value reachable from Arg (through casts/GEPs) is the address
// operand of an air.atomic.global.* call. Such args must carry metal::_atomic
// metadata or the Metal driver's TypeFinder null-derefs at PSO creation.
static bool isAtomicDeviceArg(Argument *Arg) {
  SmallVector<Value *, 8> Work{Arg};
  SmallPtrSet<Value *, 8> Seen;
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (User *U : V->users()) {
      if (auto *CI = dyn_cast<CallInst>(U)) {
        Function *Callee = CI->getCalledFunction();
        if (Callee && Callee->getName().starts_with("air.atomic.global.") &&
            CI->arg_size() > 0 && CI->getArgOperand(0) == V)
          return true;
        continue;
      }
      if (isa<BitCastInst>(U) || isa<GetElementPtrInst>(U) ||
          isa<AddrSpaceCastInst>(U))
        Work.push_back(U);
    }
  }
  return false;
}

// Fix air.arg_type_name/size in kernel metadata to match PTM pointee types
// (the pipeline may label all buffers "float"); the Metal GPU JIT validates
// these against the bitcode types.
static void fixKernelArgMetadata(Module &M, const PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto *AirKernel = M.getNamedMetadata("air.kernel");
  if (!AirKernel)
    return;

  // Canonical metal::_atomic arg metadata pairs with air.version >= (2,9,0) /
  // MSL 4.1 (macOS 26), the level the Metal compiler accepts the 8-arg
  // device-atomic form.
  bool AllowAtomicStruct =
      MetalVersion::fromTriple(M.getTargetTriple().str()).OSMajor >= 16;

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

      // Device-atomic arg: type as metal::_atomic { i32 } +
      // air.struct_type_info, matching Apple's front end. The 8-arg cmpxchg
      // PSO-crashes without it.
      bool IsAtomic = AllowAtomicStruct && Pointee->isIntegerTy(32) &&
                      isAtomicDeviceArg(Arg);
      Metadata *StructTypeInfo = nullptr;
      if (IsAtomic) {
        TypeName = "metal::_atomic";
        auto *I32 = Type::getInt32Ty(Ctx);
        StructTypeInfo = MDNode::get(
            Ctx, {ValueAsMetadata::get(ConstantInt::get(I32, 0)),
                  ValueAsMetadata::get(ConstantInt::get(I32, 4)),
                  ValueAsMetadata::get(ConstantInt::get(I32, 0)),
                  MDString::get(Ctx, "int"), MDString::get(Ctx, "__s")});
      }

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
            if (PrevS->getString() == "air.struct_type_info" &&
                I + 1 < ArgMD->getNumOperands()) {
              I++;
              continue;
            }
            if (PrevS->getString() == "air.arg_type_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              if (StructTypeInfo) {
                NewOps.push_back(MDString::get(Ctx, "air.struct_type_info"));
                NewOps.push_back(StructTypeInfo);
              }
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

// Map LLVM's in-memory AttrKind to the AIR-v1 bitcode attr-kind encoding (the
// two numbering spaces differ). Modern attrs Apple's reader rejects as "Unknown
// attribute kind" fall outside this whitelist and are dropped (sound: hints).
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
    return 39;
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

// Append the AIR-bitcode encoding of Attr to Grp; returns false for attrs
// outside the whitelist. captures(none) maps to the legacy nocapture kind.
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

// Defined in separate .cpp files.
void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E);
void emitConstantsBlock(
    BitstreamWriter &W, ValueEnumerator &E,
    ArrayRef<const Constant *> Constants, unsigned CodeSize,
    const DenseMap<const Constant *, unsigned> *PoisonPtrTypeIdx = nullptr);
void emitMetadataKindBlock(BitstreamWriter &W);
void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E,
                       MetadataEnumerator &MD);
void emitOperandBundleTagsBlock(BitstreamWriter &W);
void emitSinglethreadBlock(BitstreamWriter &W);
void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                       ValueEnumerator &E, const MetadataEnumerator &MD);

std::vector<uint8_t> emitMetalBitcode(Module &M, PointeeTypeMap &PTM) {
  SmallVector<char, 0> Buf;
  // Scope the writer so its destructor's FlushToWord() pads the final partial
  // 32-bit word before Buf is read; without it readers over-read and report
  // "truncated module".
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

    // Pre-serialization IR fixups: bring the O3 module into the subset Metal v1
    // bitcode + AGX JIT accept, refining the PTM in place. Four ordered stages;
    // stage A's lowering reshapes the IR the later type fixups inspect.
    // Stage A: legalize / lower constructs the AGX JIT can't take. (The pure
    // IR strips — lifetime/wide-int/freeze/nneg/disjoint/scmp — run earlier as
    // the MetalLegalizeUnsupportedIR codegen-prepare pass, asm-visible.)
    retagTensorOpsExternallyDefined(M);
    scalarizeBoolVectorCasts(M);
    lowerVectorPointerToInt(M);
    lowerVectorSelects(M);
    removeRedundantBitcasts(M, PTM);
    // Stage B: GEP source-type normalization (shapes 1-3, see normalizeGEPs).
    normalizeGEPs(M, PTM);
    // Stage C: pointer-pointee type agreement.
    fixPhiIncomingTypes(M, PTM);
    fixMMAPointerSuffixMismatch(M, PTM);
    fixSelectPointerArms(M, PTM);
    scalarizeAggregateLoads(M, PTM);
    scalarizeAggregateStores(M, PTM);
    fixTensorRuntimeArgTypes(M, PTM);
    fixAccessTypeMismatch(M, PTM);

    // Stage D: materialize constexprs, then post-constexpr fixups.
    lowerConstantExprs(M);
    normalizeArrayGlobalGEPs(M);
    fixKernelArgMetadata(M, PTM);

    if (const char *pw = getenv("METAL_DUMP_PREWRITE")) {
      // A path value writes the prewrite module there; otherwise stderr.
      if (pw[0] && strcmp(pw, "1") != 0) {
        std::error_code EC;
        llvm::raw_fd_ostream os(pw, EC);
        if (!EC)
          M.print(os, nullptr);
      } else {
        M.print(llvm::errs(), nullptr);
      }
    }

    ValueEnumerator E(M, PTM);

    // MODULE_BLOCK (CodeSize=4)
    W.EnterSubblock(bitc::MODULE_BLOCK_ID, 4);

    {
      SmallVector<uint64_t, 1> V = {1};
      W.EmitRecord(bitc::MODULE_CODE_VERSION, V);
    }

    // PARAMATTR blocks BEFORE TYPE_BLOCK (Metal requires this order). One
    // PARAMATTR_GRP per unique (paramIdx, AS) tuple, one list per function.
    // Group ID 1 is reserved for the legacy MMA-load nocapture+readonly entry,
    // which lives on the call site, not the function declaration.
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

    // Synthesize attr groups on the simdgroup-matrix intrinsic decls to match
    // Apple's `xcrun metal` AIR, else the macOS 13/14/15 driver rejects the
    // metallib. Bitcode ATTR_KIND_* values differ from LLVM's in-memory enum
    // (convergent is 6 in-memory, 43 in bitcode).
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
    struct SynthGroup {
      uint64_t Index; // ~0u for function attrs, param index (1-based) otherwise
      SmallVector<uint64_t, 6> EnumKinds;
    };
    SmallVector<std::pair<unsigned, SynthGroup>, 8> SynthGroups;
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

      // Function-attr group; order matches Apple's textual AIR.
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
      // Synthesized decl groups: (ID, Index, [0, kind]...); 0 = enum attr.
      for (auto &IDG : SynthGroups) {
        SmallVector<uint64_t, 16> Grp;
        Grp.push_back(IDG.first);        // group ID
        Grp.push_back(IDG.second.Index); // ~0u = function, else param index
        for (uint64_t Kind : IDG.second.EnumKinds) {
          Grp.push_back(0);
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

    // Target triple, required by the Metal GPU JIT; derive the canonical AIR
    // triple from present OS info rather than hardcoding it.
    {
      std::string T = M.getTargetTriple().str();
      if (T.empty() || T == "air")
        T = MetalVersion::fromTriple(M.getTargetTriple().str()).tripleString();
      emitString(W, bitc::MODULE_CODE_TRIPLE, T);
    }
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

    // A section is a 1-based index into these records (0 = none).
    std::map<std::string, unsigned> SectionMap;
    auto sectionIndex = [&](StringRef Sec) -> unsigned {
      if (Sec.empty())
        return 0;
      unsigned &Entry = SectionMap[std::string(Sec)];
      if (!Entry) {
        emitString(W, bitc::MODULE_CODE_SECTIONNAME, Sec);
        Entry = SectionMap.size();
      }
      return Entry;
    };
    for (auto *V : E.globalValues) {
      if (auto *G = dyn_cast<GlobalVariable>(V)) {
        if (G->hasSection())
          sectionIndex(G->getSection());
      } else if (auto *Fn = dyn_cast<Function>(V)) {
        if (Fn->hasSection())
          sectionIndex(Fn->getSection());
      }
    }

    // GLOBALVAR/FUNCTION records in globalValues order (globals first, matching
    // value ID assignment).
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
        Ops.push_back(G->hasSection() ? sectionIndex(G->getSection()) : 0);
        for (int J = 0; J < 2; J++)
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
        // paramattr list ID (0 = none); +1 when HasMMALoad reserves list 1.
        unsigned ListID = 0;
        auto It = FnAttrListID.find(Fn);
        if (It != FnAttrListID.end())
          ListID = It->second + (HasMMALoad ? 1 : 0);
        Ops.push_back(ListID);
        Ops.push_back(0); // align
        // Field 9 (unnamed_addr) encoding: None=0, Global=1, Local=2. Apple's
        // simdgroup intrinsic decls are local_unnamed_addr (=2); the macOS-14
        // driver expects this to match.
        Ops.push_back(Fn->hasSection() ? sectionIndex(Fn->getSection()) : 0);
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

    // Share one MetadataEnumerator between the module METADATA_BLOCK (nodes
    // emitted) and per-function attachment blocks (referenced by ID).
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

    // matmul2d/cooperative_tensor glue (see CoopTensorLowering); kept here as
    // it is entangled with the BitstreamWriter state below.
    // Only modules importing an `air.externally_defined` symbol need the
    // irsymtab (airdyld scans it to bind those imports); others stay byte-
    // identical to the legacy VALUE_SYMTAB-only form.
    bool HasExternallyDefined = false;
    for (auto &GO : M.global_objects())
      if (GO.getSection() == kExternallyDefinedSection) {
        HasExternallyDefined = true;
        break;
      }
    if (HasExternallyDefined) {
      SmallVector<char, 0> Symtab;
      StringTableBuilder StrtabBuilder(StringTableBuilder::RAW);
      BumpPtrAllocator Alloc;
      Module *Mods[] = {&M};
      if (!irsymtab::build(Mods, Symtab, StrtabBuilder, Alloc)) {
        StrtabBuilder.finalizeInOrder();
        std::vector<char> Strtab(StrtabBuilder.getSize());
        StrtabBuilder.write((uint8_t *)Strtab.data());

        auto emitBlob = [&](unsigned Block, unsigned Record, StringRef Blob) {
          W.EnterSubblock(Block, 3);
          auto Abbv = std::make_shared<BitCodeAbbrev>();
          Abbv->Add(BitCodeAbbrevOp(Record));
          Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
          unsigned AbbrevNo = W.EmitAbbrev(std::move(Abbv));
          W.EmitRecordWithBlob(AbbrevNo, ArrayRef<uint64_t>{Record}, Blob);
          W.ExitBlock();
        };
        emitBlob(bitc::SYMTAB_BLOCK_ID, bitc::SYMTAB_BLOB,
                 {Symtab.data(), Symtab.size()});
        emitBlob(bitc::STRTAB_BLOCK_ID, bitc::STRTAB_BLOB,
                 {Strtab.data(), Strtab.size()});
      }
    }
  } // ~BitstreamWriter flushes the final word into Buf here.

  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metal
} // namespace llvm
