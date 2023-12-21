# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(AI-SOLUTIONS)

include(FindPkgConfig)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
pkg_check_modules(JSON    REQUIRED json-glib-1.0)
pkg_check_modules(GFLAGS  REQUIRED gflags)

set(PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_ROOT}/CMake)
set(CMAKE_CXX_STANDARD 17)

find_package(GStreamer REQUIRED)
find_package(OpenCV REQUIRED )

add_subdirectory("./src")

link_directories(
    ${JSONCPP_LIBRARY_DIRS} 
    ${JSON_LIBRARY_DIRS}
    ${GFLAGS_LIBRARY_DIRS}
)
