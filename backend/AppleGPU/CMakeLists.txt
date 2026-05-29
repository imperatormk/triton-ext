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

# The convert-to-LLVM pass needs `populatePolynomialApproximateErfPattern`
# (math.erf lowering) which lives in MLIRMathTransforms. How we pull it in
# depends on how the pinned LLVM artifact ships that library:
#
#   * Linux artifact -> libMLIRMathTransforms.so (SHARED). Link it normally;
#     a shared lib's static initializers (LLVM cl::opt registrations) live in
#     the .so itself and run once, so dlopen-ing this plugin from a non-Python
#     host such as triton-opt is fine, and there is no size bloat.
#
#   * macOS artifact -> libMLIRMathTransforms.a (STATIC). Linking the whole
#     archive would drag in its transitive closure (MLIRX86Dialect ->
#     MLIRLLVMDialect -> LLVMCore) and bake those cl::opt initializers into
#     the plugin. They then re-register and crash when triton-opt dlopen's it
#     ("Option '...' registered more than once"), and bloat the .so ~15x. So
#     extract only the single object that defines the erf pattern (~0.5 MB,
#     no cl::opt initializers); its MLIR/Arith symbol references stay
#     undefined and resolve from libtriton.so at load time.
if(NOT TARGET MLIRMathTransforms)
    message(FATAL_ERROR "MLIRMathTransforms target not found; cannot provide "
                        "the math.erf polynomial-approximation pattern")
endif()
get_target_property(_MATH_TRANSFORMS_LIB MLIRMathTransforms IMPORTED_LOCATION)
if(NOT _MATH_TRANSFORMS_LIB)
    get_target_property(_MATH_TRANSFORMS_LIB MLIRMathTransforms
                        IMPORTED_LOCATION_RELEASE)
endif()
if(NOT _MATH_TRANSFORMS_LIB OR NOT EXISTS "${_MATH_TRANSFORMS_LIB}")
    message(FATAL_ERROR "Could not locate the MLIRMathTransforms library "
                        "(got '${_MATH_TRANSFORMS_LIB}')")
endif()

set(_ERF_OBJECT "")
if(_MATH_TRANSFORMS_LIB MATCHES "\\.a$")
    # Static archive: extract the single erf object at configure time so it is
    # never a build-graph input.
    set(_ERF_OBJECT_DIR "${CMAKE_CURRENT_BINARY_DIR}/erf-obj")
    set(_ERF_OBJECT "${_ERF_OBJECT_DIR}/PolynomialApproximation.cpp.o")
    file(MAKE_DIRECTORY "${_ERF_OBJECT_DIR}")
    execute_process(
        COMMAND "${CMAKE_AR}" x "${_MATH_TRANSFORMS_LIB}"
                PolynomialApproximation.cpp.o
        WORKING_DIRECTORY "${_ERF_OBJECT_DIR}"
        RESULT_VARIABLE _ERF_EXTRACT_RC)
    if(NOT _ERF_EXTRACT_RC EQUAL 0 OR NOT EXISTS "${_ERF_OBJECT}")
        message(FATAL_ERROR "Failed to extract PolynomialApproximation.cpp.o "
                            "from ${_MATH_TRANSFORMS_LIB} "
                            "(rc=${_ERF_EXTRACT_RC})")
    endif()
endif()

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
)

if(_ERF_OBJECT)
    # macOS: link the single extracted object (not the whole archive).
    target_link_options(TritonAppleGPUBackend PRIVATE "${_ERF_OBJECT}")
else()
    # Linux: MLIRMathTransforms is shared; link it directly.
    target_link_libraries(TritonAppleGPUBackend PUBLIC MLIRMathTransforms)
endif()

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

# Out-of-tree LLVM Metal/AIR target that produces metal-llc, the binary the
# AppleGPU plugin shells out to at runtime to emit .metallib from LLVM IR.
# This is an AppleGPU-private tool; it lives under backend/AppleGPU/ rather
# than at the repo root.
add_subdirectory(llvm-metal-target)
