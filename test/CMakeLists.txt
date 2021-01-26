cmake_minimum_required(VERSION 3.1)
enable_testing()

project(CUDAImageUtilitiesTest)

add_executable(CUDAImageUtilitiesTest test.cpp)

add_test(NAME test COMMAND test)