cmake_minimum_required(VERSION 3.20)
project(LLVMMetalTarget LANGUAGES CXX C)

# ---------------------------------------------------------------------------
# Out-of-tree LLVM Metal/AIR target plugin.
#
# This project mirrors llvm/lib/Target/Metal from the user's in-tree
# `metal-target-poc` branch and builds it against Triton-main's PINNED prebuilt
# LLVM (SHA 87717bf9). The result is a shared library + thin `metal-llc`
# driver that the AppleGPU Triton backend can load without requiring users to
# rebuild Triton's LLVM.
#
# STAGE 1: scaffold only. Do NOT expect this to configure cleanly yet --
# see STAGE_2_NOTES.md for the known gaps.
# ---------------------------------------------------------------------------

# Match the prebuilt LLVM's C++ ABI. Triton's LLVM is built with C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Triton's prebuilt LLVM. User can override with -DLLVM_DIR=...
if(NOT LLVM_DIR)
  set(LLVM_DIR "$ENV{HOME}/.triton/llvm/llvm-87717bf9-macos-arm64/lib/cmake/llvm"
      CACHE PATH "Path to LLVMConfig.cmake")
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "  LLVM_DIR          = ${LLVM_DIR}")
message(STATUS "  LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")
message(STATUS "  LLVM_LIBRARY_DIRS = ${LLVM_LIBRARY_DIRS}")
message(STATUS "  LLVM_ENABLE_RTTI  = ${LLVM_ENABLE_RTTI}")
message(STATUS "  LLVM_ENABLE_EH    = ${LLVM_ENABLE_EH}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
link_directories(${LLVM_LIBRARY_DIRS})

# LLVM TableGen + helper macros. The prebuilt ships AddLLVM.cmake +
# TableGen.cmake under LLVM_CMAKE_DIR.
include(AddLLVM)
include(TableGen)
include(HandleLLVMOptions)

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
