# Out-of-tree consumer of the installed QuantPortfolio package. It knows
# nothing about the source tree: only find_package and the exported targets.
#
#   cmake -S cpp/tests/find_package_consumer -B build/consumer \
#         -DCMAKE_PREFIX_PATH=<install prefix>
#   cmake --build build/consumer && ctest --test-dir build/consumer

cmake_minimum_required(VERSION 3.25)
project(qp_consumer LANGUAGES CXX)

find_package(QuantPortfolio 1.0 CONFIG REQUIRED)

add_executable(qp_consumer main.cpp)
target_link_libraries(qp_consumer PRIVATE
  qp::core qp::eqopt qp::eqvar qp::fxopt qp::fxvar)

enable_testing()
add_test(NAME qp_consumer COMMAND qp_consumer)
