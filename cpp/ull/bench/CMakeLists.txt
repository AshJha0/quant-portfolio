# Build for the low-latency (ull) components: ring, qpbf, book, bench, mcast.
#
#   cmake -S cpp/ull/bench -B build/ull -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/ull -j && ctest --test-dir build/ull --output-on-failure
#
# Linux only (rdtscp, /proc readback, sockets). Options:
#   ULL_SANITIZERS=ON  also build ASan/UBSan and TSan variants of the tests
#   ULL_FUZZ=ON        build the libFuzzer target (needs clang)
cmake_minimum_required(VERSION 3.20)
project(qp_ull LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(ULL_SANITIZERS "Build sanitizer variants of the tests" ON)
option(ULL_FUZZ "Build the libFuzzer target (clang only)" OFF)

set(ULL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
find_package(Threads REQUIRED)
enable_testing()

function(ull_test name src)
  add_executable(${name} ${src})
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-pedantic)
  target_link_libraries(${name} PRIVATE Threads::Threads)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

function(ull_sanitized name src flags)
  add_executable(${name} ${src})
  target_compile_options(${name} PRIVATE -O1 -g -fno-omit-frame-pointer ${flags})
  target_link_options(${name} PRIVATE ${flags})
  target_link_libraries(${name} PRIVATE Threads::Threads)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

ull_test(ull_ring_test ${ULL_ROOT}/ring/tests/test_ring.cpp)
ull_test(ull_qpbf_test ${ULL_ROOT}/qpbf/tests/test_qpbf.cpp)
ull_test(ull_book_test ${ULL_ROOT}/book/tests/test_book.cpp)
ull_test(ull_bench_test ${ULL_ROOT}/bench/tests/test_bench.cpp)
# Timing test: run alone so parallel tests do not preempt it.
set_tests_properties(ull_bench_test PROPERTIES RUN_SERIAL TRUE)
ull_test(ull_noalloc_test ${ULL_ROOT}/bench/tests/test_noalloc.cpp)
ull_test(ull_recovery_test ${ULL_ROOT}/mcast/tests/test_recovery.cpp)
ull_test(ull_loopback_test ${ULL_ROOT}/mcast/tests/test_loopback.cpp)

# Acceptance run: 1e8 messages round-tripped bit-exactly. Labelled "long";
# run with  ctest -L long  (about a minute on a laptop core).
add_test(NAME ull_qpbf_roundtrip_1e8 COMMAND ull_qpbf_test 100000000)
set_tests_properties(ull_qpbf_roundtrip_1e8 PROPERTIES LABELS long)

if(ULL_SANITIZERS)
  ull_sanitized(ull_ring_tsan ${ULL_ROOT}/ring/tests/test_ring.cpp "-fsanitize=thread")
  ull_sanitized(ull_qpbf_asan ${ULL_ROOT}/qpbf/tests/test_qpbf.cpp "-fsanitize=address,undefined;-fno-sanitize-recover=all")
  ull_sanitized(ull_book_asan ${ULL_ROOT}/book/tests/test_book.cpp "-fsanitize=address,undefined;-fno-sanitize-recover=all")
  ull_sanitized(ull_recovery_asan ${ULL_ROOT}/mcast/tests/test_recovery.cpp "-fsanitize=address,undefined;-fno-sanitize-recover=all")
endif()

if(ULL_FUZZ)
  add_executable(ull_fuzz_decode ${ULL_ROOT}/qpbf/fuzz/fuzz_decode.cpp)
  target_compile_options(ull_fuzz_decode PRIVATE -g -O1 -fsanitize=fuzzer,address,undefined)
  target_link_options(ull_fuzz_decode PRIVATE -fsanitize=fuzzer,address,undefined)
  add_test(NAME ull_fuzz_seeds COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/fuzz_corpus)
  add_test(NAME ull_fuzz_seeds_write COMMAND ull_qpbf_test --write-seeds ${CMAKE_BINARY_DIR}/fuzz_corpus)
  add_test(NAME ull_fuzz_decode_60s COMMAND ull_fuzz_decode -max_total_time=60 -max_len=1600 ${CMAKE_BINARY_DIR}/fuzz_corpus)
  set_tests_properties(ull_fuzz_seeds_write PROPERTIES DEPENDS ull_fuzz_seeds LABELS long)
  set_tests_properties(ull_fuzz_seeds PROPERTIES LABELS long)
  set_tests_properties(ull_fuzz_decode_60s PROPERTIES DEPENDS ull_fuzz_seeds_write)
  set_tests_properties(ull_fuzz_decode_60s PROPERTIES LABELS long)
endif()

add_executable(ull_latency_bench ${ULL_ROOT}/bench/latency_bench.cpp)
target_link_libraries(ull_latency_bench PRIVATE Threads::Threads)

add_executable(ull_replay_publisher ${ULL_ROOT}/mcast/replay_publisher.cpp)
target_link_libraries(ull_replay_publisher PRIVATE Threads::Threads)
