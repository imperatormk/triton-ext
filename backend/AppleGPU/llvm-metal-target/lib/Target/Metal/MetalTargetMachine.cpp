//===- MetalTargetMachine.cpp - Metal Target Implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the Metal (Apple AIR) target initializer and pipeline.
///
//===----------------------------------------------------------------------===//

#include "MetalTargetMachine.h"
#include "AIRWriter/MetalEmbedderPass.h"
#include "AIRWriter/MetalWriterPass.h"
#include "Metal.h"
#include "MetalAIRSystemValues.h"
#include "MetalAliasAnnotate.h"
#include "MetalAsyncCopyToCooperative.h"
#include "MetalAsyncEventToAlloca.h"
#include "MetalBFloat16CastDecompose.h"
#include "MetalBarrierRename.h"
#include "MetalDemoteF64.h"
#include "MetalDeviceLoadsVolatile.h"
#include "MetalInlineNonKernel.h"
#include "MetalLLVMToAIRIntrinsics.h"
#include "MetalLowerAtomicRMW.h"
#include "MetalLowerFNeg.h"
#include "MetalNaNMinMax.h"
#include "MetalNormalizeAllocas.h"
#include "MetalPrepare.h"
#include "MetalScalarBufferPacking.h"
#include "MetalScalarStoreGuard.h"
#include "MetalSplitI64Shuffle.h"
#include "MetalSubtarget.h"
#include "MetalTGBarrierInsert.h"
#include "MetalTGGlobalCoalesce.h"
#include "MetalTargetTransformInfo.h"
#include "MetalTripleCompat.h"
#include "TargetInfo/MetalTargetInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMetalTarget() {
  RegisterTargetMachine<MetalTargetMachine> X(getTheMetalTarget());
  auto *PR = PassRegistry::getPassRegistry();
  initializeMetalInlineNonKernelLegacyPass(*PR);
  initializeMetalDemoteF64LegacyPass(*PR);
  initializeMetalLowerFNegLegacyPass(*PR);
  initializeMetalNaNMinMaxLegacyPass(*PR);
  initializeMetalLLVMToAIRIntrinsicsLegacyPass(*PR);
  initializeMetalBarrierRenameLegacyPass(*PR);
  initializeMetalLowerAtomicRMWLegacyPass(*PR);
  initializeMetalSplitI64ShuffleLegacyPass(*PR);
  initializeMetalScalarStoreGuardLegacyPass(*PR);
  initializeMetalTGGlobalCoalesceLegacyPass(*PR);
  initializeMetalTGBarrierInsertLegacyPass(*PR);
  initializeMetalDeviceLoadsVolatileLegacyPass(*PR);
  initializeMetalAsyncCopyToCooperativeLegacyPass(*PR);
  initializeMetalAsyncEventToAllocaLegacyPass(*PR);
  initializeMetalNormalizeAllocasLegacyPass(*PR);
  initializeMetalBFloat16CastDecomposeLegacyPass(*PR);
  initializeMetalScalarBufferPackingLegacyPass(*PR);
  initializeMetalAIRSystemValuesLegacyPass(*PR);
  initializeMetalAliasAnnotateLegacyPass(*PR);
  initializeMetalPrepareLegacyPass(*PR);
  // initializeMetalEmbedderLegacyPassPass disabled out-of-tree (no AsmPrinter).
}

namespace {
/// Object-file lowering stub: the Metal backend serializes to a .metallib
/// container rather than going through the normal section machinery.
class MetalTargetObjectFile : public TargetLoweringObjectFile {
public:
  MetalTargetObjectFile() = default;

  MCSection *getExplicitSectionGlobal(const GlobalObject *GO, SectionKind Kind,
                                      const TargetMachine &TM) const override {
    // Out-of-tree: the stock MCContext has no `getMetalLibSection` and the
    // .metallib payload is emitted directly via MetalWriterPass in
    // addPassesToEmitFile -- AsmPrinter never runs, so this hook is never
    // hit in practice. Return null to make the unreachable explicit.
    (void)GO;
    (void)Kind;
    (void)TM;
    llvm_unreachable("Metal out-of-tree: AsmPrinter path is disabled; "
                     "metallib emission goes through MetalWriterPass.");
  }

protected:
  MCSection *SelectSectionForGlobal(const GlobalObject *, SectionKind,
                                    const TargetMachine &) const override {
    llvm_unreachable("Not supported!");
  }
};

class MetalPassConfig : public TargetPassConfig {
public:
  MetalPassConfig(MetalTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  MetalTargetMachine &getMetalTargetMachine() const {
    return getTM<MetalTargetMachine>();
  }

  FunctionPass *createTargetRegisterAllocator(bool) override { return nullptr; }

  void addCodeGenPrepare() override {
    // Metal IR pipeline passes (LLVM IR -> AIR-conformant IR), in order.
    addPass(createMetalInlineNonKernelLegacyPass());
    // Apple GPU has no double type; demote all f64 to f32 before anything
    // else touches the IR (and before serialization, which would crash the
    // Metal shader compiler on any surviving double).
    addPass(createMetalDemoteF64LegacyPass());
    addPass(createMetalLowerFNegLegacyPass());
    addPass(createMetalNaNMinMaxLegacyPass());
    addPass(createMetalLLVMToAIRIntrinsicsLegacyPass());
    addPass(createMetalBarrierRenameLegacyPass());
    addPass(createMetalLowerAtomicRMWLegacyPass());
    addPass(createMetalSplitI64ShuffleLegacyPass());
    addPass(createMetalScalarStoreGuardLegacyPass());
    // AGX "materializeAll" trap: a kernel that reads a threadgroup arena with
    // air.simdgroup_matrix_8x8_load while that same arena is written by
    // air.simdgroup_async_copy_2d fails PSO creation with no diagnostic.  Lower
    // those async copies to a cooperative threadgroup copy (and drop the toxic
    // symbol) so the kernel is PSO-valid regardless of the producing frontend.
    // Runs before TGGlobalCoalesce so the merge sees the cooperative form.
    addPass(createMetalAsyncCopyToCooperativeLegacyPass());
    addPass(createMetalTGGlobalCoalesceLegacyPass());
    addPass(createMetalTGBarrierInsertLegacyPass());
    addPass(createMetalDeviceLoadsVolatileLegacyPass());
    addPass(createMetalAsyncEventToAllocaLegacyPass());
    addPass(createMetalNormalizeAllocasLegacyPass());
    addPass(createMetalBFloat16CastDecomposeLegacyPass());
    // Must run BEFORE AIRSystemValues so that !air.kernel metadata is
    // emitted against the post-packing signature.
    addPass(createMetalScalarBufferPackingLegacyPass());
    addPass(createMetalAIRSystemValuesLegacyPass());
    // Emit Apple-style alias-scope MD + "air-buffer-no-alias" param attrs
    // after the IR shape is final but before final normalisations.
    addPass(createMetalAliasAnnotateLegacyPass());
    // Final pre-serialization normalizations.
    addPass(createMetalPrepareLegacyPass());
    // MetalPrepare's mergeByteGlobals now emits the identity bitcast
    // inline on bfloat/half/float-through-bfloat typed-base GEPs, so the
    // post-Prepare NormalizeAllocas re-run is no longer needed.
    // See test_scan2d[cum{sum,prod}-bfloat16-*].
  }
};
} // namespace

MetalTargetMachine::MetalTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, metal_compat::kAirDataLayout, TT, CPU, FS,
                               Options, Reloc::Static, CodeModel::Small, OL),
      TLOF(std::make_unique<MetalTargetObjectFile>()),
      Subtarget(std::make_unique<MetalSubtarget>(TT, CPU, FS, *this)) {
  initAsmInfo();
}

MetalTargetMachine::~MetalTargetMachine() {}

void MetalTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "MetalPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

bool MetalTargetMachine::addPassesToEmitFile(
    PassManagerBase &PM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, bool DisableVerify,
    MachineModuleInfoWrapperPass *MMIWP) {
  TargetPassConfig *PassConfig = createPassConfig(PM);
  PassConfig->addCodeGenPrepare();

  switch (FileType) {
  case CodeGenFileType::AssemblyFile:
    // Phase 1: emit the transformed LLVM IR. The .metallib writer lands in
    // Phase 2 (MC ObjectWriter + embedder pass).
    PM.add(createPrintModulePass(Out, "", true));
    break;
  case CodeGenFileType::ObjectFile:
    // Out-of-tree: always use the direct MetalWriterPass path. The in-tree
    // build also wires an AsmPrinter + MC ObjectWriter route, but that
    // requires MCSectionMetalLib + MetalLibObjectWriter additions to core MC
    // which we cannot land into Triton's pinned LLVM. MetalWriterPass writes
    // the .metallib bytes directly to `Out`, producing byte-identical output.
    PM.add(createMetalWriterPass(Out));
    break;
  case CodeGenFileType::Null:
    break;
  }
  return false;
}

bool MetalTargetMachine::addPassesToEmitMC(PassManagerBase &, MCContext *&,
                                           raw_pwrite_stream &, bool) {
  return true;
}

TargetPassConfig *MetalTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MetalPassConfig(*this, PM);
}

const MetalSubtarget *
MetalTargetMachine::getSubtargetImpl(const Function &) const {
  return Subtarget.get();
}

TargetTransformInfo
MetalTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<MetalTTIImpl>(this, F));
}

MetalTargetLowering::MetalTargetLowering(const MetalTargetMachine &TM,
                                         const MetalSubtarget &STI)
    : TargetLowering(TM, STI) {}
