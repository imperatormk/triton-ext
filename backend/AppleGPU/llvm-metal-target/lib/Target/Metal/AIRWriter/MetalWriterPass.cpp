//===- MetalWriterPass.cpp - Emit a .metallib container -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalWriterPass.h"
#include "BitcodeEmitter.h"
#include "MetallibWriter.h"
#include "PointeeTypeMap.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void emitTGBytesRemark(Module &M) {
  Function *F = nullptr;
  for (Function &Fn : M)
    if (!Fn.isDeclaration()) {
      F = &Fn;
      break;
    }
  if (!F || F->empty())
    return;
  const DataLayout &DL = M.getDataLayout();
  uint64_t Total = 0;
  for (const GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != 3 || GV.isDeclaration())
      continue;
    uint64_t AlignB = GV.getAlign().value_or(Align(1)).value();
    if (Total % AlignB)
      Total += AlignB - (Total % AlignB);
    Total += DL.getTypeAllocSize(GV.getValueType());
  }
  OptimizationRemark R("metal-tg", "TGBytes", DebugLoc(), &F->getEntryBlock());
  R << "threadgroup memory total "
    << DiagnosticInfoOptimizationBase::Argument("TGBytes", Total);
  F->getContext().diagnose(R);
}

static void writeMetallibImpl(Module &M, raw_pwrite_stream &OS) {
  // Constexprs must become instructions before the PTM walk so the GEPs
  // they materialize get typed-pointer entries.
  metal::lowerConstantExprs(M);
  // Reconstruct typed-pointer info into a side table (AIR v1 bitcode needs
  // typed pointers; the module itself stays opaque).
  metal::PointeeTypeMap PTM = metal::buildPointeeTypeMap(M);
  emitTGBytesRemark(M);
  // The target triple drives the AIR version (VERS air_minor); derive it from
  // the module's triple, falling back to macOS 16 when none is present.
  metal::MetallibOptions Opts;
  Opts.Version = metal::MetalVersion::fromTriple(M.getTargetTriple().str());
  metal::writeMetallib(M, PTM, OS, Opts);
}

PreservedAnalyses MetalWriterPass::run(Module &M, ModuleAnalysisManager &AM) {
  writeMetallibImpl(M, OS);
  return PreservedAnalyses::all();
}

namespace {

class MetalWriterLegacyPass : public ModulePass {
  raw_pwrite_stream &OS;

public:
  static char ID;
  explicit MetalWriterLegacyPass(raw_pwrite_stream &Out)
      : ModulePass(ID), OS(Out) {}

  StringRef getPassName() const override { return "Metal metallib writer"; }

  bool runOnModule(Module &M) override {
    writeMetallibImpl(M, OS);
    return false;
  }
};

} // namespace

char MetalWriterLegacyPass::ID = 0;

ModulePass *llvm::createMetalWriterPass(raw_pwrite_stream &Out) {
  return new MetalWriterLegacyPass(Out);
}
