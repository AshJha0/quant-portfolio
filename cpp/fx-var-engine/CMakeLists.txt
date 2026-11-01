cmake_minimum_required(VERSION 3.20)
project(fx_var_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Respect the caller's build type; default to Release for single-config generators.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FXVAR_WERROR "Treat compiler warnings as errors" OFF)
option(FXVAR_BUILD_BENCH "Build the benchmark executable" ON)

# Warnings for GCC/Clang only; MSVC keeps its defaults.
set(FXVAR_WARNINGS
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
)
target_include_directories(fxvar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fxvar PRIVATE ${FXVAR_WARNINGS})

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
  )
  foreach(t IN LISTS FXVAR_TESTS)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE fxvar GTest::gtest GTest::gtest_main)
    target_compile_options(${t} PRIVATE ${FXVAR_WARNINGS}
      # The tests exercise the deprecated entry points on purpose.
      $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-unused-result;-Wno-deprecated-declarations>)
    gtest_discover_tests(${t})
  endforeach()
endif()

# ---------------------------------------------------------------- benchmark
if(FXVAR_BUILD_BENCH)
  add_executable(fxvar_bench bench/bench_main.cpp)
  target_link_libraries(fxvar_bench PRIVATE fxvar)
  target_compile_options(fxvar_bench PRIVATE ${FXVAR_WARNINGS})
endif()
