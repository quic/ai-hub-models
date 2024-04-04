#RESOLVING DEPENDENCIES

# steps to copy opencv
wget https://sourceforge.net/projects/opencvlibrary/files/4.5.5/opencv-4.5.5-android-sdk.zip/download
unzip download
rm download
mkdir sdk
mv OpenCV-android-sdk/sdk/* sdk
rm -r OpenCV-android-sdk

mkdir -p app/src/main/jniLibs/arm64-v8a
mkdir -p app/src/main/assets

##writing jniLibs
cp $QNN_SDK_ROOT/lib/hexagon-v73/unsigned/*.so app/src/main/jniLibs/arm64-v8a/
cp $QNN_SDK_ROOT/lib/aarch64-android/*.so app/src/main/jniLibs/arm64-v8a/