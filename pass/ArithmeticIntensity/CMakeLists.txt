triton_extension(ArithmeticIntensityPass)

# Add pass TableGen.
set(PASS_TABLEGEN_OUTPUT "TritonExt${TRITON_EXT_CLASS}IncGen")
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(${PASS_TABLEGEN_OUTPUT})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add compile definitions for generated code.
add_compile_definitions(
    TRITON_EXT_NAME=${PROJECT_NAME}
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
)

# Build the pass extension library (SHARED).
add_mlir_library(${PROJECT_NAME}
    ArithmeticIntensity.cpp

    SHARED

    DEPENDS
    ${PASS_TABLEGEN_OUTPUT}

    LINK_LIBS PUBLIC
    TritonExtensionSupport
)

# --no-as-needed/--as-needed force-keep libtriton on GNU ld; Apple ld lacks
# those flags and keeps it anyway, so link TRITON_LIB plainly on macOS.
if(APPLE)
  set(_triton_lib_link ${TRITON_LIB})
else()
  set(_triton_lib_link -Wl,--no-as-needed ${TRITON_LIB} -Wl,--as-needed)
endif()

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    TritonExtensionSupport
    ${_triton_lib_link}
)

# add_mlir_library links the static LLVM/MLIR archives (directly and via
# TritonExtensionSupport's PUBLIC interface), embedding a second copy of MLIR.
# Under macOS two-level namespacing those duplicate TypeIDs differ from the ones
# libtriton.so built the ModuleOp with, so cast<ModuleOp> aborts. Replace the
# link interface with just the two needed dylibs (by path, so their static-archive
# interface isn't inherited); dynamic_lookup + flat_namespace then bind every
# MLIR/LLVM/triton symbol to libtriton.so's single definition at load time.
if(APPLE)
  add_dependencies(${PROJECT_NAME} TritonExtensionSupport)
  set_target_properties(${PROJECT_NAME} PROPERTIES
      LINK_LIBRARIES
      "${TRITON_EXT_LIBRARY_DIR}/libTritonExtensionSupport.dylib;${TRITON_LIB}")
  target_link_options(${PROJECT_NAME} PRIVATE -undefined dynamic_lookup)
  target_link_options(${PROJECT_NAME} PRIVATE -Wl,-flat_namespace)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${TRITON_EXT_LIBRARY_DIR}")

if(APPLE)
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "$<TARGET_FILE_NAME:${PROJECT_NAME}>"
      "${TRITON_EXT_LIBRARY_DIR}/lib${PROJECT_NAME}.so"
    VERBATIM)
endif()
