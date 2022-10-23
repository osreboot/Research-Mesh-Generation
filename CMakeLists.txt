cmake_minimum_required(VERSION 3.23)
project(TriangulationDeWall CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(SOURCE_FILES
        "main.cu"
        "primitive.cuh"
        "math.cuh"
        "dewall.cuh"
        "blelloch.cuh"
        "profiler.cuh"
        "profiler.cu")

add_executable(TriangulationDeWall ${SOURCE_FILES})

set_target_properties(TriangulationDeWall PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
