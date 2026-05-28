# ---------------------------------------------------------------------------
# Out-of-tree LLVM Metal/AIR target.
#
# Mirrors llvm/lib/Target/Metal and builds it against the prebuilt LLVM that
# triton-ext already pulls (~/.triton/llvm/llvm-<sha>-...). Produces
# libLLVMMetalTarget + the metal-llc driver. The AppleGPU plugin shells out
# to metal-llc at runtime to lower LLVM IR to .metallib.
#
# Buildable two ways:
#  1. Nested (default): as a subdirectory of backend/AppleGPU when triton-ext
#     is built top-level. Inherits LLVM discovery from the parent project.
#  2. Standalone: cmake -S <this-dir> -B <build> for fast iteration on the
#     target without rebuilding the rest of triton-ext.
# ---------------------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_PROJECT_NAME)
  cmake_minimum_required(VERSION 3.20)
  project(LLVMMetalTarget LANGUAGES CXX C)
endif()

if(NOT LLVM_FOUND)
  # Standalone build: do our own LLVM discovery.
  if(NOT LLVM_DIR)
    set(LLVM_DIR
        "$ENV{HOME}/.triton/llvm/llvm-87717bf9-macos-arm64/lib/cmake/llvm"
        CACHE PATH "Path to LLVMConfig.cmake")
  endif()
  find_package(LLVM REQUIRED CONFIG)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
link_directories(${LLVM_LIBRARY_DIRS})

# Point TableGen at the prebuilt llvm-tblgen binary.
if(NOT LLVM_TABLEGEN_EXE)
  find_program(LLVM_TABLEGEN_EXE
    NAMES llvm-tblgen
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  if(NOT LLVM_TABLEGEN_EXE)
    message(FATAL_ERROR "llvm-tblgen not found in ${LLVM_TOOLS_BINARY_DIR}")
  endif()
  set(LLVM_TABLEGEN_EXE "${LLVM_TABLEGEN_EXE}" CACHE PATH "llvm-tblgen")
endif()
# `add_public_tablegen_target` and other AddLLVM helpers add deps on targets
# `intrinsics_gen`, `acc_gen`, `omp_gen` normally provided by an in-tree LLVM
# build. Create no-op stubs so out-of-tree projects don't fail.
foreach(_t intrinsics_gen acc_gen omp_gen)
  if(NOT TARGET ${_t})
    add_custom_target(${_t})
  endif()
endforeach()

# Vendor headers we add over the top of the prebuilt LLVM (e.g.
# MCSectionMetalLib.h). Must precede prebuilt include dir so our copy wins.
include_directories(BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Target/Metal/include)

# Match RTTI/EH so vtables and exception tables interop with the prebuilt libs.
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
if(NOT LLVM_ENABLE_EH)
  add_compile_options(-fno-exceptions)
endif()

# Sources in our mirror live under lib/Target/Metal/, so make headers there
# visible to all sub-projects (mirrors the in-tree -I llvm/lib/Target/Metal
# implicit include).
include_directories(BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Target/Metal
  ${CMAKE_CURRENT_BINARY_DIR}/lib/Target/Metal
)

add_subdirectory(lib)
add_subdirectory(tools)
