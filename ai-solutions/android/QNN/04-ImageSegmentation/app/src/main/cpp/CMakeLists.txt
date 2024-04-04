
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("imageSegmentation")

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Log/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/PAL/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Utils/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/WrapperUtils/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/QNN/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)



add_library( # Sets the name of the library.
        imageSegmentation

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
              src/inference.cpp src/imageSegmentation.cpp src/QnnSampleApp.cpp include/QnnTypeMacros.hpp src/QnnWrapperUtils.cpp Utils/DataUtil.cpp Log/Logger.cpp Log/LogUtils.cpp PAL/src/common/GetOpt.cpp PAL/src/common/StringOp.cpp PAL/src/linux/Directory.cpp PAL/src/linux/DynamicLoading.cpp PAL/src/linux/FileOp.cpp PAL/src/linux/Path.cpp Utils/DynamicLoadUtil.cpp Utils/IOTensor.cpp Utils/QnnSampleAppUtils.cpp
              Utils/DynamicLoadUtil.cpp
              Log/Logger.cpp
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
        imageSegmentation

                       # Links the target library to the log library
                       # included in the NDK.
                ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a/libQnnSystem.so

                       ${log-lib} ${OpenCV_LIBS})
