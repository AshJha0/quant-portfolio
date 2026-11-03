# cpp/ull/gateway: order gateway with W28d pre-trade checks and the shared
# kill contract (docs/KILL_CONTRACT.md). Header-only library plus tests.
# Builds on its own:  cmake -S cpp/ull/gateway -B build/ull-gateway && cmake --build build/ull-gateway && ctest --test-dir build/ull-gateway
cmake_minimum_required(VERSION 3.25)
project(qp_ull_gateway LANGUAGES CXX)

if(NOT TARGET qp_ull_feed)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../feed ${CMAKE_BINARY_DIR}/ull_feed)
endif()

add_library(qp_ull_gateway INTERFACE)
target_include_directories(qp_ull_gateway INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qp_ull_gateway INTERFACE qp_ull_feed)
target_compile_features(qp_ull_gateway INTERFACE cxx_std_20)

option(QP_ULL_GATEWAY_TESTS "Build cpp/ull/gateway tests" ON)
if(QP_ULL_GATEWAY_TESTS)
  enable_testing()
  get_filename_component(_qp_repo ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
  add_executable(test_ull_gateway tests/test_gateway.cpp)
  target_link_libraries(test_ull_gateway PRIVATE qp_ull_gateway)
  target_compile_definitions(test_ull_gateway PRIVATE QP_REPO_ROOT="${_qp_repo}")
  if(NOT MSVC)
    target_compile_options(test_ull_gateway PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endif()
  add_test(NAME ull_gateway COMMAND test_ull_gateway)
endif()
