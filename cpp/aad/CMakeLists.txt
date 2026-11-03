# qp::aad: header-only tape-based reverse-mode AAD (W19), the C++ port of
# qp_core.aad and rust/aad. Depends on qp::core (normal cdf) and
# qp::rates_curves (quote and period types, bootstrap for the oracle).
# Built on its own it also builds its test and the cost-ratio bench:
#
#   cmake -S cpp/aad -B build/aad -DQP_WERROR=ON
#   cmake --build build/aad -j && ctest --test-dir build/aad
#   build/aad/qp_aad_cost_ratio
#
# The test is a plain executable (exit code 0 = pass), so no test
# framework is downloaded.

cmake_minimum_required(VERSION 3.25)
project(qp_aad VERSION 0.1.0 DESCRIPTION "Adjoint algorithmic differentiation (W19)" LANGUAGES CXX)

option(QP_AAD_TESTS "Build the AAD test and bench" ${PROJECT_IS_TOP_LEVEL})
option(QP_WERROR "Treat compiler warnings as errors" OFF)

add_library(qp_aad INTERFACE)
add_library(qp::aad ALIAS qp_aad)
target_include_directories(qp_aad INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../core/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../rates-curves/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(qp_aad INTERFACE cxx_std_23)

if(QP_AAD_TESTS)
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
  set(_qp_warn
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    $<$<AND:$<BOOL:${QP_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>
    $<$<AND:$<BOOL:${QP_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>)
  add_executable(qp_aad_tests tests/test_aad.cpp)
  target_link_libraries(qp_aad_tests PRIVATE qp::aad)
  target_compile_definitions(qp_aad_tests PRIVATE
    QP_AAD_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/../../python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt")
  target_compile_options(qp_aad_tests PRIVATE ${_qp_warn})
  add_test(NAME qp_aad_tests COMMAND qp_aad_tests)
  add_executable(qp_aad_cost_ratio bench/cost_ratio.cpp)
  target_link_libraries(qp_aad_cost_ratio PRIVATE qp::aad)
  target_compile_options(qp_aad_cost_ratio PRIVATE ${_qp_warn})
endif()
