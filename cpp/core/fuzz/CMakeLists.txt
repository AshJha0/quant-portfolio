# Fuzz targets for qp-core (W12). Standalone project: configure this
# directory on its own, it is not part of the default build.
#
#   Clang (coverage-guided, libFuzzer):
#     cmake -S cpp/core/fuzz -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DQP_FUZZ_ENGINE=libfuzzer
#     cmake --build build-fuzz && build-fuzz/fuzz_jsonl_parse -max_total_time=60 cpp/core/fuzz/corpus/jsonl_parse
#   GCC (corpus replay only):
#     cmake -S cpp/core/fuzz -B build-replay -DCMAKE_CXX_COMPILER=g++ -DQP_FUZZ_ENGINE=replay
#     cmake --build build-replay && build-replay/fuzz_jsonl_parse cpp/core/fuzz/corpus/jsonl_parse
cmake_minimum_required(VERSION 3.25)
project(qp_core_fuzz CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(QP_FUZZ_ENGINE "libfuzzer" CACHE STRING "libfuzzer or replay")

set(QP_FUZZ_TARGETS fuzz_implied_vol fuzz_cholesky_repair fuzz_jsonl_parse)
foreach(t IN LISTS QP_FUZZ_TARGETS)
  if(QP_FUZZ_ENGINE STREQUAL "libfuzzer")
    add_executable(${t} ${t}.cpp)
    target_compile_options(${t} PRIVATE -g -O1 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined)
    target_link_options(${t} PRIVATE -fsanitize=fuzzer,address,undefined)
  else()
    add_executable(${t} ${t}.cpp replay_main.cpp)
    target_compile_options(${t} PRIVATE -g -O1 -fsanitize=address,undefined -fno-sanitize-recover=undefined)
    target_link_options(${t} PRIVATE -fsanitize=address,undefined)
  endif()
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
