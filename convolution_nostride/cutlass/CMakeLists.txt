cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(main LANGUAGES CXX CUDA)

#
# CUTLASS 3.x requires C++17
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# cutlass
include_directories(../../cutlass/include)
include_directories(../../cutlass/tools/util/include)
include_directories(../../cutlass/examples/common)

add_executable(cutlass_fprop cutlass_fprop.cu)

set_property(TARGET cutlass_fprop PROPERTY CUDA_ARCHITECTURES "80")

message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}") 
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}") 