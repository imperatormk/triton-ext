# ---------------------------------------------------------------------------
# Out-of-tree LLVM Metal/AIR target.
#
# Mirrors llvm/lib/Target/Metal and builds it against the prebuilt LLVM that
# triton-ext already pulls. Produces libLLVMMetalTarget + the metal-llc driver.
# The AppleGPU plugin shells out to metal-llc at runtime to lower LLVM IR to
# .metallib.
#
# Built as a subdirectory of backend/AppleGPU when triton-ext is configured
# top-level, inheriting LLVM discovery (LLVM_FOUND, the CMake module path, and
# the AddLLVM/TableGen helpers) from the parent project.
# ---------------------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
