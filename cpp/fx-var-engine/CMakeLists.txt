cmake_minimum_required(VERSION 3.25)
project(fx_var_engine CXX)

# C++23 floor. QP_CXX26=ON compiles with C++26 and turns on the features
# qp::core guards with feature-test macros (contracts, std::simd); the
# results are the same either way.
option(QP_CXX26 "Compile with C++26 and enable C++26 features where supported" OFF)
if(QP_CXX26)
  set(CMAKE_CXX_STANDARD 26)
else()
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Respect the caller's build type; default to Release for single-config generators.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FXVAR_WERROR "Treat compiler warnings as errors" OFF)
option(FXVAR_BUILD_BENCH "Build the benchmark executable" ON)
# FXVAR_NATIVE adds -O3 -march=native to every target: local latency work
# only, never CI (the binary will not run on older CPUs and FMA contraction
# can move last bits). FXVAR_LTO turns on CMAKE_INTERPROCEDURAL_OPTIMIZATION.
# FXVAR_VEC_REPORT prints the vectorisation report for the hot sources.
# CMAKE_CXX_FLAGS is never modified.
option(FXVAR_NATIVE "Add -O3 -march=native (local benchmarking only)" OFF)
option(FXVAR_LTO "Enable link-time optimisation" OFF)
option(FXVAR_VEC_REPORT "Print the vectorisation report for the hot kernels" OFF)

if(FXVAR_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _fxvar_ipo OUTPUT _fxvar_ipo_msg)
  if(_fxvar_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "FXVAR_LTO requested but not supported: ${_fxvar_ipo_msg}")
  endif()
endif()

add_library(fxvar_tuning INTERFACE)
target_compile_options(fxvar_tuning INTERFACE
  $<$<AND:$<BOOL:${FXVAR_NATIVE}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3 -march=native>)

# Warnings: -Wall -Wextra -Wpedantic (GCC/Clang), /W4 /permissive- (MSVC).
set(FXVAR_WARNINGS
  $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive->
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall;-Wextra;-Wpedantic>
  $<$<AND:$<BOOL:${FXVAR_WERROR}>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-Werror>
  $<$<AND:$<BOOL:${FXVAR_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>
)

# ------------------------------------------------------------------ library
add_library(fxvar
  src/matrix.cpp
  src/stats.cpp
  src/returns.cpp
  src/book.cpp
  src/expected_shortfall.cpp
  src/historical.cpp
  src/parametric.cpp
  src/monte_carlo.cpp
  src/backtest.cpp
  src/stress.cpp
  src/kernels.cpp
  src/option_reval.cpp
)
target_include_directories(fxvar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fxvar PRIVATE ${FXVAR_WARNINGS})
target_link_libraries(fxvar PUBLIC fxvar_tuning)
# Shared numerical core (cpp/core, header-only target qp::core).
if(NOT TARGET qp::core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/qp_core)
endif()
target_link_libraries(fxvar PUBLIC qp::core)
if(FXVAR_VEC_REPORT)
  set_property(SOURCE src/kernels.cpp src/book.cpp src/matrix.cpp src/monte_carlo.cpp
    APPEND PROPERTY COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU>:-fopt-info-vec-all>"
    "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize>")
endif()

# -------------------------------------------------------------------- tests
include(CTest)  # defines BUILD_TESTING (default ON)
if(BUILD_TESTING)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)

  set(FXVAR_TESTS
    test_matrix_stats
    test_book
    test_expected_shortfall
    test_historical
    test_parametric
    test_monte_carlo
    test_backtest
    test_stress
    test_golden_python
    test_conventions
    test_fuzz
    test_latency_api
    test_option_reval
  )
  foreach(t IN LISTS FXVAR_TESTS)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE fxvar GTest::gtest GTest::gtest_main)
    target_compile_options(${t} PRIVATE ${FXVAR_WARNINGS}
      # The tests exercise the deprecated entry points on purpose.
      $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-unused-result;-Wno-deprecated-declarations>
      $<$<CXX_COMPILER_ID:MSVC>:/wd4834;/wd4996>)
    gtest_discover_tests(${t})
  endforeach()

  # Replaces global operator new to count allocations, so it is its own
  # binary. Sanitizer runtimes define operator new themselves (a static
  # runtime then clashes at link time), so sanitizer builds skip it.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_executable(test_no_alloc tests/test_no_alloc.cpp)
    target_link_libraries(test_no_alloc PRIVATE fxvar GTest::gtest GTest::gtest_main)
    target_compile_options(test_no_alloc PRIVATE ${FXVAR_WARNINGS})
    gtest_discover_tests(test_no_alloc)
  endif()
endif()

# ---------------------------------------------------------------- benchmark
if(FXVAR_BUILD_BENCH)
  add_executable(fxvar_bench bench/bench_main.cpp)
  target_link_libraries(fxvar_bench PRIVATE fxvar)
  target_compile_options(fxvar_bench PRIVATE ${FXVAR_WARNINGS})
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _fxvar_bt)
  set(_fxvar_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_fxvar_bt}}")
  if(FXVAR_NATIVE)
    string(APPEND _fxvar_flags " -O3 -march=native")
  endif()
  if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    string(APPEND _fxvar_flags " LTO")
  endif()
  string(STRIP "${_fxvar_flags}" _fxvar_flags)
  target_compile_definitions(fxvar_bench PRIVATE QP_BENCH_FLAGS="${_fxvar_flags}")
endif()
