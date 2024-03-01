cmake_minimum_required(VERSION 3.5.1)


# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

message(STATUS "source file path" ${PROJECT_SRC_ROOT})

file(GLOB_RECURSE SRC_FILE
    ../src/*.cpp
)

set(SOURCE_FILE
    ${SRC_FILE}
)

add_executable(ai-solutions ${SRC_FILE})
target_compile_options(ai-solutions PUBLIC -fPIC  -O0 -g -Wall -Wnon-virtual-dtor)

# Header path
include_directories(
    "../inc"
    "/usr/include/glib-2.0"
    "/usr/lib/aarch64-linux-gnu/glib-2.0/include"
    "/usr/include/gstreamer-1.0"
    "/usr/local/include/opencv4"
    ${OpenCV_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)

message(STATUS "JSON file path" ${JSON_INCLUDE_DIRS})
message(STATUS "JSONCPP file path" ${JSONCPP_INCLUDE_DIRS})

target_link_libraries(ai-solutions PUBLIC pthread dl ${OpenCV_LIBS} ${GST_APP_LIBRARIES} ${JSON_LIBRARIES} jsoncpp SNPE jsoncpp)
