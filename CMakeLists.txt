cmake_minimum_required(VERSION 3.16)

project(tsparter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_subdirectory(deps)
add_subdirectory(tsparter)
add_subdirectory(apps)
