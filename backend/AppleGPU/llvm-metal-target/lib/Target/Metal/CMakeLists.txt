# Out-of-tree Metal target build.
#
# In-tree this uses add_llvm_target(MetalCodeGen ...) which implicitly hooks
# the target into the LLVM_ALL_TARGETS list. As an external project we instead
# produce a single shared library `libLLVMMetalTarget.dylib` that contains:
#
#   - LLVMMetalInfo      (TargetInfo/)        -- target registration
#   - LLVMMetalDesc      (MCTargetDesc/)      -- MC layer registration
#   - LLVMMetalAIRWriter (AIRWriter/)         -- bitcode/metallib emitter
#   - LLVMMetalCodeGen   (this directory)     -- passes + TargetMachine
#
# Stage 2 will decide whether to keep these as four object libraries glued
# into one .dylib (current plan) or to ship them as separate component
# libraries the way in-tree does. The single-dylib path makes runtime loading
# from compiler.py trivial (one dlopen, one LLVMInitializeMetalTarget call).

# ---------------------------------------------------------------------------
# TableGen
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS Metal.td)

tablegen(LLVM MetalGenSubtargetInfo.inc          -gen-subtarget)
tablegen(LLVM MetalGenInstrInfo.inc              -gen-instr-info)
tablegen(LLVM MetalGenRegisterInfo.inc           -gen-register-info)
tablegen(LLVM MetalGenAIRIntrinsicMappings.inc   -gen-searchable-tables)

add_public_tablegen_target(MetalCommonTableGen)

# ---------------------------------------------------------------------------
# Source lists (mirrors in-tree add_llvm_target(MetalCodeGen ...) body).
# ---------------------------------------------------------------------------
set(METAL_CODEGEN_SOURCES
  MetalAIRSystemValues.cpp
  MetalAliasAnnotate.cpp
  MetalAsyncEventToAlloca.cpp
  MetalBFloat16CastDecompose.cpp
  MetalBarrierRename.cpp
  MetalDeviceLoadsVolatile.cpp
  MetalInlineNonKernel.cpp
  MetalLLVMToAIRIntrinsics.cpp
  MetalLowerAtomicRMW.cpp
  MetalLowerFNeg.cpp
  MetalNaNMinMax.cpp
  MetalNormalizeAllocas.cpp
  MetalPrepare.cpp
  MetalScalarBufferPacking.cpp
  MetalScalarStoreGuard.cpp
  MetalSplitI64Shuffle.cpp
  MetalTGBarrierInsert.cpp
  MetalTGGlobalCoalesce.cpp
  # MetalAsmPrinter.cpp -- out-of-tree path bypasses AsmPrinter (see
  # MetalTargetMachine::addPassesToEmitFile). Kept on disk for reference.
  MetalInstrInfo.cpp
  MetalRegisterInfo.cpp
  MetalSubtarget.cpp
  MetalTargetMachine.cpp
)

set(METAL_INFO_SOURCES
  TargetInfo/MetalTargetInfo.cpp
)

set(METAL_DESC_SOURCES
  # MCTargetDesc/MetalContainerObjectWriter.cpp -- requires MCMetalLibObjectWriter
  # which lives only in the in-tree branch. The out-of-tree MetalAsmBackend
  # below is also disabled because the writer pipeline is bypassed entirely.
  MCTargetDesc/MetalMCTargetDesc.cpp
)

set(METAL_AIRWRITER_SOURCES
  AIRWriter/BitcodeEmitter.cpp
  AIRWriter/ConstantsWriter.cpp
  AIRWriter/FunctionWriter.cpp
  AIRWriter/MetadataWriter.cpp
  # MetalEmbedderPass is registered in MetalPassRegistry.def for `opt -passes=`
  # use; keep compiling it even though the AsmPrinter pipeline is bypassed.
  AIRWriter/MetalEmbedderPass.cpp
  AIRWriter/MetallibWriter.cpp
  AIRWriter/MetalWriterPass.cpp
  AIRWriter/PointeeTypeMap.cpp
  AIRWriter/TypeTableWriter.cpp
  AIRWriter/ValueEnumerator.cpp
)

# ---------------------------------------------------------------------------
# Single shared library bundling all four pieces.
# ---------------------------------------------------------------------------
add_library(LLVMMetalTarget SHARED
  ${METAL_INFO_SOURCES}
  ${METAL_DESC_SOURCES}
  ${METAL_AIRWRITER_SOURCES}
  ${METAL_CODEGEN_SOURCES}
)

add_dependencies(LLVMMetalTarget MetalCommonTableGen intrinsics_gen)

target_include_directories(LLVMMetalTarget PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Map in-tree LINK_COMPONENTS to actual prebuilt libraries. Resolved via
# llvm_map_components_to_libnames so we follow what the prebuilt config
# exports.
llvm_map_components_to_libnames(_metal_llvm_libs
  Analysis
  AsmPrinter
  BitWriter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
)
target_link_libraries(LLVMMetalTarget PUBLIC ${_metal_llvm_libs})

set_target_properties(LLVMMetalTarget PROPERTIES
  OUTPUT_NAME "LLVMMetalTarget"
  MACOSX_RPATH ON
)
