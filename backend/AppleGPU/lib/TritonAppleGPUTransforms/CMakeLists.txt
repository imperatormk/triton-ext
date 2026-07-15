add_triton_library(TritonAppleGPUTransforms
  AccelerateAppleMatmul.cpp
  SimplifyGatherLayout.cpp
  StoreShuffleLayout.cpp

  DEPENDS
  TritonAppleGPUDialectIncGen
  TritonAppleGPUAttrDefsIncGen
  TritonAppleGPUTransformsIncGen
  TritonGPUIR
  MLIRLLVMDialect
)

target_include_directories(TritonAppleGPUTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAppleGPUTransforms PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/../../include)
