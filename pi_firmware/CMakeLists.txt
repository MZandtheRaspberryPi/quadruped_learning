cmake_minimum_required(VERSION 3.16)

project(pi_firmware VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pi_i2c)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC pi_i2c_lib)