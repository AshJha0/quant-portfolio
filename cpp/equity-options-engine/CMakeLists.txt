# Equity options pricing engine: C++20 port of the Python eq_options package.
#
# Build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#   ./build/eqopt_bench
#
# Options:
#   EQOPT_WERROR   treat compiler warnings as errors (default OFF; CI sets ON)
#   EQOPT_TESTS    build the GoogleTest suite (default ON)
# Sanitizers and other instrumentation are passed through CMAKE_CXX_FLAGS
# and CMAKE_EXE_LINKER_FLAGS; this file never overwrites them.

cmake_minimum_required(VERSION 3.20)
project(equity_options_engine
        VERSION 1.1.0
        DESCRIPTION "Equity options pricing engine"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EQOPT_WERROR "Treat compiler warnings as errors" OFF)
option(EQOPT_TESTS "Build the test suite" ON)

# Default to Release for single-config generators only; respect any choice.
get_property(_eqopt_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _eqopt_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Warning flags per compiler, attached to targets (not to CMAKE_CXX_FLAGS).
add_library(eqopt_warnings INTERFACE)
target_compile_options(eqopt_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
  $<$<AND:$<BOOL:${EQOPT_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>
  $<$<AND:$<BOOL:${EQOPT_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- library ----
add_library(eqopt
  src/black_scholes.cpp
  src/greeks.cpp
  src/implied_vol.cpp
  src/binomial.cpp
  src/black76.cpp
  src/monte_carlo.cpp
)
target_include_directories(eqopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(eqopt PUBLIC Threads::Threads PRIVATE eqopt_warnings)

# ------------------------------------------------------------- jsonl shim ----
# JSON-lines front end used by the cross-language differential test.
add_executable(eqopt_jsonl tools/jsonl_main.cpp)
target_link_libraries(eqopt_jsonl PRIVATE eqopt eqopt_warnings)

# -------------------------------------------------------------- benchmark ----
add_executable(eqopt_bench bench/bench_main.cpp)
target_link_libraries(eqopt_bench PRIVATE eqopt eqopt_warnings)

# ------------------------------------------------------------------ tests ----
if(EQOPT_TESTS)
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
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(eqopt_tests
    tests/test_golden.cpp
    tests/test_black_scholes.cpp
    tests/test_greeks.cpp
    tests/test_implied_vol.cpp
    tests/test_binomial.cpp
    tests/test_black76.cpp
    tests/test_monte_carlo.cpp
    tests/test_numerical_safety.cpp
  )
  target_include_directories(eqopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(eqopt_tests PRIVATE eqopt eqopt_warnings GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(eqopt_tests)

  # The shim answers one good and one bad request.
  add_test(NAME eqopt_jsonl_smoke
    COMMAND ${CMAKE_COMMAND}
      -DSHIM=$<TARGET_FILE:eqopt_jsonl>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke_input.txt
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke.cmake)
endif()
