cmake_minimum_required(VERSION 3.20)
project(agpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(AGPU_WARNINGS -Wall -Wextra -Wpedantic)

add_library(agpu_core INTERFACE)
target_include_directories(agpu_core INTERFACE include)

add_library(agpu_msl STATIC src/Printer.cpp)
target_include_directories(agpu_msl PUBLIC include)
target_compile_options(agpu_msl PRIVATE ${AGPU_WARNINGS})

enable_testing()

file(GLOB AGPU_TEST_SRCS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.cpp")

foreach(src ${AGPU_TEST_SRCS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE agpu_msl agpu_core)
  target_include_directories(${name} PRIVATE test)
  target_compile_options(${name} PRIVATE ${AGPU_WARNINGS})
  add_test(NAME ${name} COMMAND ${name})
endforeach()
