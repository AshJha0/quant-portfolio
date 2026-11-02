# Install and export the four C++ engines and qp::core as one CMake package.
#
# Each engine still builds on its own from its directory (its presets, tests
# and benches are unchanged). This file exists only to install them:
#
#   cmake -S cpp -B build/cpp-install -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cpp-install -j
#   cmake --install build/cpp-install --prefix <prefix>
#
# A consumer then writes
#
#   find_package(QuantPortfolio 1.0 CONFIG REQUIRED)
#   target_link_libraries(app PRIVATE qp::eqopt qp::fxopt qp::eqvar qp::fxvar)
#
# and passes -DCMAKE_PREFIX_PATH=<prefix>. tests/find_package_consumer is
# that consumer; CI builds it against a fresh install prefix.
#
# Tests and benches of the engines are off here: this is a packaging build.
# The installed libraries are compiled without -march=native (the *_NATIVE
# options default OFF), so they run on any CPU of the target architecture.

cmake_minimum_required(VERSION 3.25)
project(QuantPortfolio VERSION 1.0.0
        DESCRIPTION "quant-portfolio C++ option and VaR engines"
        LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_property(_qp_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _qp_multi AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EQOPT_TESTS OFF CACHE BOOL "" FORCE)
set(FXOPT_TESTS OFF CACHE BOOL "" FORCE)
set(EQVAR_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(FXVAR_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(QP_CORE_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(core ${CMAKE_BINARY_DIR}/qp_core)
add_subdirectory(equity-options-engine)
add_subdirectory(equity-var-engine)
add_subdirectory(fx-options-engine)
add_subdirectory(fx-var-engine)

# Exported names: qp::core, qp::eqopt, qp::eqvar, qp::fxopt, qp::fxvar.
set_target_properties(qp_core PROPERTIES EXPORT_NAME core)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# eqopt_warnings and fxopt_warnings are private to the static libraries; they
# appear in the export only as link-only entries and add no consumer flags.
install(TARGETS qp_core eqopt eqopt_tuning eqopt_warnings eqvar eqvar_tuning
                fxopt fxopt_tuning fxopt_warnings fxvar fxvar_tuning
        EXPORT QuantPortfolioTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Headers: one directory per library, so <eqopt/...>, <qp/...> and so on.
foreach(_dir core/include/qp equity-options-engine/include/eqopt
             equity-var-engine/include/eqvar fx-options-engine/include/fxopt
             fx-var-engine/include/fxvar)
  install(DIRECTORY ${_dir} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          FILES_MATCHING PATTERN "*.hpp")
endforeach()

set(_qp_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/QuantPortfolio)
install(EXPORT QuantPortfolioTargets
        NAMESPACE qp::
        DESTINATION ${_qp_cmake_dir})
configure_package_config_file(cmake/QuantPortfolioConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/QuantPortfolioConfig.cmake
  INSTALL_DESTINATION ${_qp_cmake_dir})
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/QuantPortfolioConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/QuantPortfolioConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/QuantPortfolioConfigVersion.cmake
        DESTINATION ${_qp_cmake_dir})
