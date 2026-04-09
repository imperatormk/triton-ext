llvm_map_components_to_libnames(llvm_libs
  Core Support IRReader BitWriter TransformUtils Analysis Passes
)

add_library(MetalIRBridge SHARED
  metalir_bridge.cpp
)

target_link_libraries(MetalIRBridge
  MetalIRTransforms
  MetalIRSerialization
  ${llvm_libs}
)

target_include_directories(MetalIRBridge PUBLIC
  ${APPLE_GPU_INCLUDE_DIR}
)

set_target_properties(MetalIRBridge PROPERTIES
  OUTPUT_NAME "MetalIRBridge"
  PREFIX "lib"
  # Ship alongside the backend plugin so the Python driver finds it.
  LIBRARY_OUTPUT_DIRECTORY "${TRITON_EXT_LIBRARY_DIR}"
  CXX_VISIBILITY_PRESET hidden
  C_VISIBILITY_PRESET hidden
)

if(APPLE)
  target_link_options(MetalIRBridge PRIVATE
    -Wl,-exported_symbol,_metalir_compile
    -Wl,-exported_symbol,_metalir_free
    -Wl,-exported_symbol,_metalir_tg_memory_bytes
  )
else()
  target_link_options(MetalIRBridge PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
  )
endif()
