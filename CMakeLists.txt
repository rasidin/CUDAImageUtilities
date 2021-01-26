cmake_minimum_required(VERSION 3.1)
project(CUDAImageUtilities)

option(USE_TEST "Use test" OFF)
if (USE_TEST)
    add_subdirectory(test)
endif()

file(GLOB_RECURSE FILES_CUDA
    "cuda/*.cu"
)
source_group("CUDASource" FILES ${FILES_CUDA})
file(GLOB_RECURSE FILES_HEADER
    "include/*.h"
)
source_group("Include" FILES ${FILES_HEADER})

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA Version : " ${CUDA_VERSION})
endif()