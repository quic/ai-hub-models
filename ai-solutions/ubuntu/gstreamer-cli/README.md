## Prepare Executable - Ubuntu

### Install dependencies
```bash
apt update
apt install build-essential cmake unzip git pkg-config
apt install libjpeg-dev libpng-dev
apt-get install libjsoncpp-dev libjson-glib-dev libgflags-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt install libxvidcore-dev libx264-dev
apt install libgtk-3-dev
apt install libatlas-base-dev gfortran
```

### Installing OpenCV 4.5.5
```bash
adb shell
wget https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.5.5 -O opencv-4.5.5.tar.gz
tar  -zxvf opencv-4.5.5.tar.gz
cd ./opencv-4.5.5
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local/opencv4.5 \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_GENERATE_PKGCONFIG=YES \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      ..
make -j8
make install 
```

### Setup environment
```bash
cp -r <SNPE_ROOT>/include/SNPE/Dl* <APP_PATH>/include/
cp -r <SNPE_ROOT>/include/SNPE/DiagLog <APP_PATH>/include/
cp -r <SNPE_ROOT>/include/SNPE/Wrapper.hpp <APP_PATH>/include/
cp -r <SNPE_ROOT>/include/SNPE/SNPE/ <APP_PATH>/include/
```

```bash
adb shell
cd <SNPE_ROOT>
cp -r lib/aarch64-ubuntu-gcc7.5/* /usr/lib/
cp bin/aarch64-ubuntu-gcc7.5/snpe-net-run /usr/bin/
cp -r lib/hexagon-v66/unsigned/lib* /usr/lib/rfsa/adsp/
chmod +x /usr/bin/snpe-net-run
snpe-net-run --version
```
Expected output: SNPE v2.12.0.230626174329_59328

### Building the application
```bash
adb shell
cd <APP_PATH>
mkdir build 
cd build
cmake ..
make -j8
```

### Running the application
```bash
export XDG_RUNTIME_DIR=/run/user/root
cd build
```
#### To run inference on input image
NOTE: Make sure "input-config-name":"image" in data/config.json
```bash
./out/ai-solutions -c ../data/config.json -i Sample1.jpg -o output.jpg
```
#### To run inference on camera stream
NOTE: Make sure "input-config-name":"camera" in data/config.json
```bash
./out/ai-solutions -c ../data/config.json
```

#### Details on Input arguments:

##### Sample config.json
model-config:
```json
"model-configs":[

    "model-name":"QSrnet-medium",                               -> model name which is used while enabling solution
    "model-type":"superresolution",                             -> To specify the use case such superresolution or detection or segmentation etc..
    "model-path":"../models/quicksrnet_medium_quantized.dlc",   -> Path at which model is located on target
    "runtime":"DSP",                                            -> Select Runtime either CPU or DSP
    "input-layers":[                                            -> Input layer of the model 
        "t.1"
    ],
    "output-layers":[
        "depth_to_space#1"                                      -> Output layer of the model
    ],
    "output-tensors":[
            "65"                                                -> Output node for post processing
        ]
]
```

solution-config:
```json
"solution-configs":[
    {
        "solution-name":"AI-Solutions",                         -> To identify Solution
        "model-name":"mobilenet-ssd",                           -> Specify model name to be executed
        "input-config-name":"camera",                           -> To read input from camera stream
        "Enable":1,                                             -> Enable specific solution
        "output-type":"wayland"                                 -> To display output on monitor                        
    },
    {
        "solution-name":"AI-Solutions",
        "model-name":"mobilenet-ssd",
        "input-config-name":"image",                            -> To read input from image
        "Enable":0,
        "output-type":"wayland"
    }
]


```