
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("ImageEnhancement")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

###OPENCV
#find_package(OpenCV REQUIRED)  ##FAILED, cannot find libcpufeatures.so
#set(OpenCV_STATIC on)
#set(OpenCV_DIR C:/Users/shubgoya/Desktop/SNPEworkspace/github_workspace/HRNET_posenet/opencv45/native/jni)
find_package(OpenCV REQUIRED)
#INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


###INCLUDE_DIRECTORIES
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc/zdl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc/hpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


add_library( # Sets the name of the library.
        ImageEnhancement

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
              inference.cpp inference_helper.cpp Model.h ImageEnhancement.cpp
              MBLLEN.h MBLLEN.cpp RUAS.h RUAS.cpp SCI.cpp SCI.h StableLLVE.h StableLLVE.cpp ZeroDCE.h ZeroDCE.cpp
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        ImageEnhancement

                       # Links the target library to the log library
                       # included in the NDK.
                ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a/libSNPE.so

                       ${log-lib} ${OpenCV_LIBS})
