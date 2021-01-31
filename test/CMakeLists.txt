cmake_minimum_required(VERSION 3.1)
enable_testing()

project(CUDAImageUtilitiesTest)

add_executable(CUDAImageUtilitiesTest test.cpp "Test_GenerateIrradianceMap.h" "TestImage.h")

target_link_libraries(CUDAImageUtilitiesTest CUDAImageUtilities)

add_test(NAME test COMMAND CUDAImageUtilitiesTest)