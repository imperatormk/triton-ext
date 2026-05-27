add_executable(metal-llc metal-llc.cpp)

llvm_map_components_to_libnames(_metal_llc_llvm_libs
  Analysis
  AsmPrinter
  BitReader
  CodeGen
  Core
  IRReader
  MC
  Support
  Target
  TargetParser
)

target_link_libraries(metal-llc PRIVATE LLVMMetalTarget ${_metal_llc_llvm_libs})

set_target_properties(metal-llc PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
