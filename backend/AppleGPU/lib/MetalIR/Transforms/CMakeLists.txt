llvm_map_components_to_libnames(llvm_libs
  Core Support IRReader BitWriter TransformUtils Analysis
)

add_library(MetalIRTransforms
  Pipeline.cpp
  InlineNonKernel.cpp
  DecomposeStructPhis.cpp
  PtrPhiToI64.cpp
  PtrSelectToI64.cpp
  LLVMToAIRIntrinsics.cpp
  DeviceLoadsVolatile.cpp
  InferTypedPointers.cpp
  ScalarBufferPacking.cpp
  AIRSystemValues.cpp
  NormalizeI1Pointers.cpp
  NormalizeAllocas.cpp
  PointeeTypeMap.cpp
  KernelProfile.cpp
)

target_link_libraries(MetalIRTransforms ${llvm_libs})
target_include_directories(MetalIRTransforms PUBLIC
  ${APPLE_GPU_INCLUDE_DIR}
)

set_target_properties(MetalIRTransforms PROPERTIES POSITION_INDEPENDENT_CODE ON)
