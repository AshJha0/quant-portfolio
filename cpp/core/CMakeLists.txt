# qp::core: header-only library shared by the four C++ engines.
#
# Engines pull it in with
#   if(NOT TARGET qp::core)
#     add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/qp_core)
#   endif()
# and link qp::core. Built on its own (cmake --preset <p> in this
# directory) it also builds and registers its unit tests.
#
# Options:
#   QP_CXX26              compile with C++26 and enable the features behind
#                         feature-test macros (contracts, std::simd). Default OFF.
#   QP_SIMD_EXPERIMENTAL  route qp::matvec_rows through <experimental/simd>
#                         (GCC). Test builds only: it proves the vector path
#                         gives the scalar bits on compilers without std::simd.
#   QP_CORE_TESTS         build the core unit tests (default ON only when this
#                         directory is the top-level project).

cmake_minimum_required(VERSION 3.25)
project(qp_core VERSION 0.1.0 DESCRIPTION "Shared numerical core of the C++ engines" LANGUAGES CXX)

option(QP_CXX26 "Compile with C++26 and enable C++26 features where supported" OFF)
option(QP_SIMD_EXPERIMENTAL "Use <experimental/simd> for the matvec vector path (tests only)" OFF)
option(QP_CORE_TESTS "Build the qp::core unit tests" ${PROJECT_IS_TOP_LEVEL})

find_package(Threads REQUIRED)

add_library(qp_core INTERFACE)
add_library(qp::core ALIAS qp_core)
target_include_directories(qp_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qp_core INTERFACE Threads::Threads)
if(QP_CXX26)
  target_compile_features(qp_core INTERFACE cxx_std_26)
  target_compile_definitions(qp_core INTERFACE QP_ENABLE_STD_SIMD=1)
  # GCC 16 implements P2900 contracts behind -fcontracts. GCC 15 also accepts
  # the flag for an older, incompatible design, so only 16+ gets it.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
    target_compile_options(qp_core INTERFACE -fcontracts)
    # The default contract-violation handler (handle_contract_violation)
    # lives in libstdc++exp in GCC 16; without it every TU that evaluates a
    # contract fails to link.
    target_link_libraries(qp_core INTERFACE stdc++exp)
  endif()
else()
  target_compile_features(qp_core INTERFACE cxx_std_23)
endif()
if(QP_SIMD_EXPERIMENTAL)
  target_compile_definitions(qp_core INTERFACE QP_SIMD_EXPERIMENTAL=1)
endif()

if(QP_CORE_TESTS)
  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 23)
    if(QP_CXX26)
      set(CMAKE_CXX_STANDARD 26)
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
  endif()
  get_property(_qp_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(NOT _qp_multi AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  option(QP_WERROR "Treat compiler warnings as errors" OFF)
  enable_testing()
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    find_package(GTest QUIET)
  endif()
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  add_executable(qp_core_tests tests/test_core.cpp tests/test_perf.cpp)
  target_link_libraries(qp_core_tests PRIVATE qp::core GTest::gtest GTest::gtest_main)
  target_compile_options(qp_core_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    $<$<AND:$<BOOL:${QP_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>
    $<$<AND:$<BOOL:${QP_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>)
  include(GoogleTest)
  gtest_discover_tests(qp_core_tests)

  # Latency of the Phase 6 kernels next to the code they replace.
  add_executable(bench_core bench/bench_core.cpp)
  target_link_libraries(bench_core PRIVATE qp::core)
  target_compile_options(bench_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<BOOL:${QP_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _qp_bt)
  string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_qp_bt}}" _qp_flags)
  target_compile_definitions(bench_core PRIVATE QP_BENCH_FLAGS="${_qp_flags}")
endif()
