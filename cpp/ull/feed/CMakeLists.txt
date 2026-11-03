# cpp/ull/feed: crypto L2/L3 feed handler (Coinbase full/level2, Binance
# diff-depth) with A/B arbitration, snapshot buffering, stale-book flag and a
# raw-byte recorder. Header-only library plus tests on recorded fixtures; the
# tests never open a socket.
# Builds on its own:  cmake -S cpp/ull/feed -B build/ull-feed && cmake --build build/ull-feed && ctest --test-dir build/ull-feed
cmake_minimum_required(VERSION 3.25)
project(qp_ull_feed LANGUAGES CXX)

add_library(qp_ull_feed INTERFACE)
target_include_directories(qp_ull_feed INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qp_ull_feed INTERFACE cxx_std_20)

option(QP_ULL_FEED_TESTS "Build cpp/ull/feed tests" ON)
if(QP_ULL_FEED_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_feed.cpp)
  enable_testing()
  add_executable(test_ull_feed tests/test_feed.cpp)
  target_link_libraries(test_ull_feed PRIVATE qp_ull_feed)
  target_compile_definitions(test_ull_feed PRIVATE QP_FEED_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
  if(NOT MSVC)
    target_compile_options(test_ull_feed PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME ull_feed COMMAND test_ull_feed)
endif()
