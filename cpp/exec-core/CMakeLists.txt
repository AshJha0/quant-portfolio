# qp::exec: the fast limit order book (W25). Semantics:
# python/execution/src/qp_exec/lob/SPEC.md.
#
#   cmake -S cpp/exec-core -B build/exec-core -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/exec-core -j && ctest --test-dir build/exec-core
#
# exec_core_replay is the differential-test driver used by
# python/execution/tests/test_differential_cpp.py.
cmake_minimum_required(VERSION 3.25)
project(qp_exec_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(EXEC_CORE_TESTS "Build exec-core tests" ${PROJECT_IS_TOP_LEVEL})

add_library(qp_exec STATIC src/book.cpp)
add_library(qp::exec ALIAS qp_exec)
target_include_directories(qp_exec PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
if(MSVC)
  target_compile_options(qp_exec PRIVATE /W4 /WX)
else()
  target_compile_options(qp_exec PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)
endif()

add_executable(exec_core_replay tools/replay.cpp)
target_link_libraries(exec_core_replay PRIVATE qp::exec)

if(EXEC_CORE_TESTS)
  enable_testing()
  add_executable(exec_core_test_book tests/test_book.cpp)
  target_link_libraries(exec_core_test_book PRIVATE qp::exec)
  add_test(NAME exec_core_test_book COMMAND exec_core_test_book)
endif()
