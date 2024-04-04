# -*- mode: shell script -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

#RESOLVING DEPENDENCIES

# steps to copy opencv
wget https://sourceforge.net/projects/opencvlibrary/files/4.5.5/opencv-4.5.5-android-sdk.zip/download
unzip download
rm download
mkdir sdk
mv OpenCV-android-sdk/sdk/* sdk
rm -r OpenCV-android-sdk

#Steps to paste files in JNI
##copying snpe-release.aar file
## Change $SNPE_ROOT/lib/android/snpe-release.aar to $SNPE_ROOT/android/snpe-release.aar  for SNPE<=2.10
mkdir snpe-release
cp $SNPE_ROOT/lib/android/snpe-release.aar snpe-release
unzip -o snpe-release/snpe-release.aar -d snpe-release/snpe-release

mkdir -p app/src/main/jniLibs/arm64-v8a

##writing jniLibs
cp snpe-release/snpe-release/jni/arm64-v8a/libc++_shared.so app/src/main/jniLibs/arm64-v8a/
cp snpe-release/snpe-release/jni/arm64-v8a/libSNPE.so app/src/main/jniLibs/arm64-v8a/
cp snpe-release/snpe-release/jni/arm64-v8a/libsnpe-android.so app/src/main/jniLibs/arm64-v8a/
cp snpe-release/snpe-release/jni/arm64-v8a/libSnpeHtpPrepare.so app/src/main/jniLibs/arm64-v8a/
cp snpe-release/snpe-release/jni/arm64-v8a/libSnpeHtpV73Skel.so app/src/main/jniLibs/arm64-v8a/
cp snpe-release/snpe-release/jni/arm64-v8a/libSnpeHtpV73Stub.so app/src/main/jniLibs/arm64-v8a/