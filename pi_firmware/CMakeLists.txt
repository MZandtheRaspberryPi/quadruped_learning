cmake_minimum_required(VERSION 3.16)

project(pi_bittle_firmware)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pi_i2c)

add_library(${PROJECT_NAME}_lib SHARED
              src/bittle_config.cpp
              src/bittle.cpp
              )
target_include_directories(${PROJECT_NAME}_lib PUBLIC include)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC pi_i2c_lib)

add_executable(bittle-main src/main.cpp)
target_link_libraries(bittle-main PUBLIC pi_i2c_lib ${PROJECT_NAME}_lib)

add_executable(bittle-set-angle src/set_angle.cpp)
target_link_libraries(bittle-set-angle PUBLIC pi_i2c_lib ${PROJECT_NAME}_lib)