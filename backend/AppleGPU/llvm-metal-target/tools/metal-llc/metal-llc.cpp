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
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
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
             cl::desc("File type to emit: 'obj' (metallib) or 'asm' (LLVM IR "
                      "text after the Metal pre-codegen passes)"),
             cl::init("obj"));

static int reportError(const Twine &Msg) {
  errs() << "metal-llc: " << Msg << "\n";
  return 1;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Register codegen / IR pass option parsers so -stop-after, -stop-before,
  // -print-after, -print-before etc. resolve before we parse the command line.
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);

  cl::ParseCommandLineOptions(argc, argv, "metal-llc: AIR/metallib backend\n");

  // Register only the Metal target (this binary is single-target).
  LLVMInitializeMetalTargetInfo();
  LLVMInitializeMetalTargetMC();
  LLVMInitializeMetalTarget();

  CodeGenFileType CGFT;
  if (FileType == "obj") {
    CGFT = CodeGenFileType::ObjectFile;
  } else if (FileType == "asm") {
    CGFT = CodeGenFileType::AssemblyFile;
  } else {
    return reportError("unsupported -filetype: " + FileType +
                       " (expected 'obj' or 'asm')");
  }

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
  Triple TT(MTriple);
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, ErrStr);
  if (!TheTarget)
    return reportError("lookupTarget failed: " + ErrStr);

  TargetOptions Options;
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TT, /*CPU=*/"", /*Features=*/"", Options, std::nullopt));
  if (!TM)
    return reportError("createTargetMachine returned null");

  M->setDataLayout(TM->createDataLayout());
  // The -mtriple flag carries the target macOS version (e.g.
  // air64_v26-apple-macosx14.0.0), which drives all per-OS AIR version fields
  // (air.version, MSL language version, VERS, subarch). The module's own triple
  // is typically a bare "air"/"" with NO OS component, so prefer the flag's
  // OS-bearing triple. Only keep the module triple if IT already carries a
  // macosx OS version and the flag does not.
  {
    StringRef ModTriple = M->getTargetTriple().str();
    bool ModHasOS = ModTriple.contains("macosx");
    bool FlagHasOS = StringRef(MTriple).contains("macosx");
    if (FlagHasOS || !ModHasOS)
      M->setTargetTriple(TT);
  }

  // Open output. Text for asm (IR dump), binary for obj (metallib bytes).
  std::error_code EC;
  sys::fs::OpenFlags Flags = (CGFT == CodeGenFileType::AssemblyFile)
                                 ? sys::fs::OF_Text
                                 : sys::fs::OF_None;
  std::unique_ptr<ToolOutputFile> Out;
  if (OutputFile == "-") {
    Out = std::make_unique<ToolOutputFile>("-", EC, Flags);
  } else {
    Out = std::make_unique<ToolOutputFile>(OutputFile, EC, Flags);
  }
  if (EC)
    return reportError("cannot open output: " + EC.message());

  // Run the codegen pipeline.
  legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  if (TM->addPassesToEmitFile(PM, Out->os(), /*DwoOut=*/nullptr, CGFT,
                              /*DisableVerify=*/false)) {
    return reportError("target does not support the requested output");
  }

  PM.run(*M);
  Out->keep();
  return 0;
}
