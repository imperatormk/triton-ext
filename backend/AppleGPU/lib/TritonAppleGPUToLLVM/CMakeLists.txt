add_triton_library(TritonAppleGPUToLLVM
  Dialect.cpp
  AppleMmaLayoutConversions.cpp
  DotOpToLLVM.cpp
  DotCommon.cpp
  DotOpBlockedConversion.cpp
  DotOpAppleMmaConversion.cpp
  LoadStoreToLLVM.cpp
  TargetInfo.cpp
  ConvertCommon.cpp
  ConvertLayoutOpConversion.cpp
  AtomicOpConversion.cpp
  MemoryOpConversion.cpp
  MiscOpConversion.cpp
  FragmentElementwiseConversion.cpp
  AsyncCopyConversion.cpp
  ConvertTritonAppleGPUToLLVM.cpp

  DEPENDS
  TritonAppleGPUDialectIncGen
  TritonAppleGPUAttrDefsIncGen
  TritonGPUIR
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRMathTransforms
  MLIRControlFlowToLLVM
  MLIRIndexToLLVM
  MLIRUBToLLVM
  MLIRReconcileUnrealizedCasts
)

target_include_directories(TritonAppleGPUToLLVM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAppleGPUToLLVM PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/../../include)
