//===- metal-llc.cpp ----------------------------------------------------===//
//
// Minimal driver that mirrors the subset of `llc` the AppleGPU Triton
// backend invokes today:
//
//   metal-llc -mtriple=air -filetype=obj -o OUT IN
//
// Reads textual LLVM IR (or `-` for stdin), runs the Metal codegen pipeline,
// and emits a metallib byte stream. Designed to be byte-identical to the
// in-tree `llc -mtriple=air -filetype=obj` output so compiler.py can switch
// over by just flipping METAL_LLC_PATH.
//
// STAGE 1: this is a scaffold. It will not link until stage 2 resolves the
// Triple::air / MetalLib registration gap (see STAGE_2_NOTES.md).
//===---------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include <memory>
#include <string>

using namespace llvm;

// Provided by libLLVMMetalTarget (TargetInfo + MCTargetDesc + TargetMachine
// + AIRWriter registrations). Declared here so we don't need the in-tree
// llvm/Config/Targets.def hook.
extern "C" void LLVMInitializeMetalTargetInfo();
extern "C" void LLVMInitializeMetalTargetMC();
extern "C" void LLVMInitializeMetalTarget();
// Out-of-tree: AsmPrinter path is disabled (see MetalTargetMachine).

static cl::opt<std::string>
    InputFile(cl::Positional, cl::desc("<input IR file or ->"), cl::init("-"));

static cl::opt<std::string> OutputFile("o", cl::desc("Output metallib path"),
                                       cl::value_desc("filename"),
                                       cl::init("-"));

static cl::opt<std::string> MTriple("mtriple",
                                    cl::desc("Target triple (default: air)"),
                                    cl::init("air"));

static cl::opt<std::string>
    FileType("filetype",
             cl::desc("File type to emit: obj (=metallib) is the only "
                      "supported value"),
             cl::init("obj"));

static int reportError(const Twine &Msg) {
  errs() << "metal-llc: " << Msg << "\n";
  return 1;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "metal-llc: AIR/metallib backend\n");

  // Register only the Metal target (this binary is single-target).
  LLVMInitializeMetalTargetInfo();
  LLVMInitializeMetalTargetMC();
  LLVMInitializeMetalTarget();

  if (FileType != "obj")
    return reportError("only -filetype=obj is supported");

  // Parse input IR.
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFile, Err, Ctx);
  if (!M) {
    Err.print("metal-llc", errs());
    return 1;
  }

  // Resolve target.
  std::string ErrStr;
  const Target *TheTarget = TargetRegistry::lookupTarget(MTriple, ErrStr);
  if (!TheTarget)
    return reportError("lookupTarget failed: " + ErrStr);

  TargetOptions Options;
  Triple TT(MTriple);
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TT, /*CPU=*/"", /*Features=*/"", Options, std::nullopt));
  if (!TM)
    return reportError("createTargetMachine returned null");

  M->setDataLayout(TM->createDataLayout());
  if (M->getTargetTriple().empty())
    M->setTargetTriple(TT);

  // Open output (binary stream).
  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out;
  if (OutputFile == "-") {
    Out = std::make_unique<ToolOutputFile>("-", EC, sys::fs::OF_None);
  } else {
    Out = std::make_unique<ToolOutputFile>(OutputFile, EC, sys::fs::OF_None);
  }
  if (EC)
    return reportError("cannot open output: " + EC.message());

  // Run the codegen pipeline.
  legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  if (TM->addPassesToEmitFile(PM, Out->os(), /*DwoOut=*/nullptr,
                              CodeGenFileType::ObjectFile,
                              /*DisableVerify=*/false)) {
    return reportError("target does not support metallib emission");
  }

  PM.run(*M);
  Out->keep();
  return 0;
}
