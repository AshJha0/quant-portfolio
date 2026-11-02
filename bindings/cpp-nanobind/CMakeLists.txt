# qp_cpp: nanobind extension over the four C++ engines in ../../cpp.
# The engines are pulled in with add_subdirectory(... EXCLUDE_FROM_ALL), so
# only their libraries are built (no tests, benches or JSON-lines tools).
cmake_minimum_required(VERSION 3.26)
project(qp_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The engine libraries are static and end up inside a shared module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Engine switches: libraries only.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EQOPT_TESTS OFF CACHE BOOL "" FORCE)
set(FXOPT_TESTS OFF CACHE BOOL "" FORCE)
set(EQVAR_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(FXVAR_BUILD_BENCH OFF CACHE BOOL "" FORCE)

set(QP_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)
if(NOT EXISTS ${QP_CPP_ROOT}/equity-options-engine/CMakeLists.txt)
  message(FATAL_ERROR "qp_cpp builds from the repository tree: ${QP_CPP_ROOT} not found. "
                      "Install with `pip install ./bindings/cpp-nanobind` from the repo root.")
endif()
foreach(engine equity-options-engine fx-options-engine equity-var-engine fx-var-engine)
  add_subdirectory(${QP_CPP_ROOT}/${engine} ${CMAKE_BINARY_DIR}/engines/${engine} EXCLUDE_FROM_ALL)
endforeach()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC
  src/module.cpp
  src/bind_eqopt.cpp
  src/bind_fxopt.cpp
  src/bind_eqvar.cpp
  src/bind_fxvar.cpp)
target_link_libraries(_core PRIVATE eqopt fxopt eqvar fxvar)
install(TARGETS _core LIBRARY DESTINATION qp_cpp)
