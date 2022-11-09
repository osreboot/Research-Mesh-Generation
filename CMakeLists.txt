cmake_minimum_required(VERSION 3.23)
project(MeshGeneration CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(SOURCE_FILES
        "main.cu")

add_subdirectory(src)

add_executable(MeshGeneration ${SOURCE_FILES})

set_target_properties(MeshGeneration PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
