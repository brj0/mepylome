cmake_minimum_required(VERSION 3.10)

project(mepylome
    VERSION 0.0.0
    LANGUAGES CXX
    DESCRIPTION "A toolkit to analyse DNA methylation arrays")

set(CMAKE_CXX_STANDARD 11)


# Enable the subset mode option for faster compile time
option(DEBUG_MODE "Enable subset compile mode" OFF)

# Standard mode includes all distance functions (slower)
set(STANDARD_CXX_FLAGS "-Wall -g -pg -O3")

# Subset mode (only subset of metrics)
set(DEBUG_CXX_FLAGS "-Wall -g -pg -O0 -Wextra")

# Select the appropriate compiler flags
if(DEBUG_MODE)
    set(CMAKE_CXX_FLAGS "${DEBUG_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "${STANDARD_CXX_FLAGS}")
endif()


# Collect source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Library
add_library(mepylome STATIC ${SRC_FILES})

# Source code
target_include_directories(mepylome PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)


# Tests
add_executable(parser_test tests/parser_test.cpp)
target_link_libraries(parser_test PRIVATE mepylome)
