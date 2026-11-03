# qp::rates_curves: header-only dated OIS, multi-curve and cross-currency
# bootstrap (W13b), the C++ port of fi_rates.bootstrap / multicurve and
# fx_rates.xccy. Built on its own it also builds its tests, which read the
# QuantLib fixture shared with the Python and Rust tests:
#
#   cmake -S cpp/rates-curves -B build/rates-curves -DQP_WERROR=ON
#   cmake --build build/rates-curves -j && ctest --test-dir build/rates-curves

cmake_minimum_required(VERSION 3.25)
project(qp_rates_curves VERSION 0.1.0 DESCRIPTION "Dated curve bootstrap (W13b)" LANGUAGES CXX)

option(QP_RATES_TESTS "Build the rates-curves tests" ${PROJECT_IS_TOP_LEVEL})
option(QP_WERROR "Treat compiler warnings as errors" OFF)

add_library(qp_rates_curves INTERFACE)
add_library(qp::rates_curves ALIAS qp_rates_curves)
target_include_directories(qp_rates_curves INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_features(qp_rates_curves INTERFACE cxx_std_23)

if(QP_RATES_TESTS)
  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
  endif()
  get_property(_qp_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(NOT _qp_multi AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
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
  add_executable(qp_rates_curves_tests tests/test_rates_curves.cpp)
  target_link_libraries(qp_rates_curves_tests PRIVATE qp::rates_curves GTest::gtest GTest::gtest_main)
  target_compile_definitions(qp_rates_curves_tests PRIVATE
    QP_RATES_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/../../python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt")
  target_compile_options(qp_rates_curves_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    $<$<AND:$<BOOL:${QP_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>
    $<$<AND:$<BOOL:${QP_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>)
  include(GoogleTest)
  gtest_discover_tests(qp_rates_curves_tests)
endif()
