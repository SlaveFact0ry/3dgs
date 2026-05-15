cmake_minimum_required(VERSION 3.18)
project(render_2dgs_cpu_omp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Apple Clang does not ship libomp; provide Homebrew hint before find_package
if(APPLE)
    set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES   "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    find_library(OpenMP_omp_LIBRARY omp
        HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    set(OpenMP_omp_LIBRARY ${OpenMP_omp_LIBRARY})
endif()

find_package(OpenMP REQUIRED)

pybind11_add_module(render_2dgs_cpu_omp
    render.cpp
    bindings.cpp
)

target_compile_options(render_2dgs_cpu_omp PRIVATE
    -O3
)

target_link_libraries(render_2dgs_cpu_omp PRIVATE
    OpenMP::OpenMP_CXX
)
