set(APPLE_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(APPLE_GPU_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# Include support code for plugin export
include_directories(${TRITON_EXT_SUPPORT_DIR})
include_directories(${APPLE_GPU_INCLUDE_DIR})
# Generated inc files live under the binary dir, mirroring the source include tree
include_directories(${APPLE_GPU_GEN_DIR})

# TableGen needs to find .td files in subdirs
list(APPEND LLVM_TABLEGEN_FLAGS
    "-I${APPLE_GPU_INCLUDE_DIR}/Dialect/TritonAppleGPU/IR")

# Ensure output subdirectories exist
file(MAKE_DIRECTORY "${APPLE_GPU_GEN_DIR}/Dialect/TritonAppleGPU/IR")
file(MAKE_DIRECTORY "${APPLE_GPU_GEN_DIR}/TritonAppleGPUTransforms")

# TableGen for dialect — output to Dialect/TritonAppleGPU/IR/
set(LLVM_TARGET_DEFINITIONS
    ${APPLE_GPU_INCLUDE_DIR}/Dialect/TritonAppleGPU/IR/TritonAppleGPUDialect.td)
mlir_tablegen(Dialect/TritonAppleGPU/IR/Dialect.h.inc -gen-dialect-decls -dialect=applegpu)
mlir_tablegen(Dialect/TritonAppleGPU/IR/Dialect.cpp.inc -gen-dialect-defs -dialect=applegpu)
add_public_tablegen_target(TritonAppleGPUDialectIncGen)

set(LLVM_TARGET_DEFINITIONS
    ${APPLE_GPU_INCLUDE_DIR}/Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.td)
mlir_tablegen(Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=applegpu)
mlir_tablegen(Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=applegpu)
add_public_tablegen_target(TritonAppleGPUAttrDefsIncGen)

# TableGen for passes — output to TritonAppleGPUTransforms/
set(LLVM_TARGET_DEFINITIONS
    ${APPLE_GPU_INCLUDE_DIR}/TritonAppleGPUTransforms/Passes.td)
mlir_tablegen(TritonAppleGPUTransforms/Passes.h.inc -gen-pass-decls -name TritonAppleGPU)
add_public_tablegen_target(TritonAppleGPUTransformsPassesIncGen)

file(GLOB TOLLVM_SRCS lib/TritonAppleGPUToLLVM/*.cpp)

# The IR -> AIR -> metallib lowering pipeline now lives in the
# out-of-tree LLVM Metal target at <triton-ext>/llvm-metal-target/.
# The Python driver shells out to its `metal-llc` binary.

# Build as a single shared library
add_mlir_library(TritonAppleGPUBackend
    ExportAppleGPU.cpp
    ${TOLLVM_SRCS}

    SHARED

    DEPENDS
    TritonAppleGPUDialectIncGen
    TritonAppleGPUAttrDefsIncGen
    TritonAppleGPUTransformsPassesIncGen

    LINK_LIBS PUBLIC
    TritonExtensionSupport
    ${TRITON_LIB}
    MLIRMathTransforms
)

set_target_properties(TritonAppleGPUBackend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${TRITON_EXT_LIBRARY_DIR}"
    OUTPUT_NAME "applegpu_backend"
    CXX_STANDARD 17)

# On macOS with shared LLVM, use flat namespace so the plugin can resolve
# LLVM symbols from any loaded dylib (not just the one it was linked against).
if(APPLE)
    target_link_options(TritonAppleGPUBackend PRIVATE -Wl,-flat_namespace)
endif()
