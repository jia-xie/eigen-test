cmake_minimum_required(VERSION 3.27)
project(dare_eigen_decomposition)

set(CMAKE_CXX_STANDARD 17)


add_executable(dare_eigen_decomposition src/main.cpp)

target_include_directories(dare_eigen_decomposition PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
