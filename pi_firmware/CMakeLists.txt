cmake_minimum_required(VERSION 3.16)

project(pi_bittle_firmware)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pi_i2c)

add_library(${PROJECT_NAME}_lib STATIC
              src/bittle_config.cpp
              )
target_include_directories(${PROJECT_NAME}_lib PUBLIC include)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC pi_i2c_lib)

add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC pi_i2c_lib ${PROJECT_NAME}_lib)