## Object Detection with YoloNAS / YoloX
The project is designed to utilize the [Qualcomm® AI Engine Direct](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html), a deep learning software for Object Detection in Android. The Android application can be designed to use any built-in/connected camera to capture the objects and use Machine Learning model to get the prediction/inference and location of the respective objects.

# Pre-requisites

* Qualcomm® AI Engine Direct setup should be completed by following the guide [here](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/setup.html)
* Install onnx and onnxruntime using `pip install onnx onnxruntime`
* Android device 6.0 and above can be used to test the application
* Download CocoDataset 2014.


## List of Supported Devices

- Snapdragon® SM8550

The above targets supports the application with CPU and DSP.

# Source Overview

## Source Organization

demo : Contains demo GIF

app : Contains source files in standard Android app format

app\src\main\assets, app\src\main\jniLibs\arm64-v8a : Contains Model library file / cached binary

app\src\main\java\com\qc\objectdetectionYoloNas : Application java source code 

app\src\main\cpp : native source code 
  
sdk: Contains openCV sdk

## Models

### Model Overview

Please refer to [Models](https://github.qualcomm.com/qualcomm-model-zoo-public-mirror/models-for-solutions/tree/main/03-object-detection) repository for model overview.

### Model Conversion

YoloNAS model is trained on COCO dataset for 80 classes of everyday objects.
List of the classes can be found in dataset at : https://cocodataset.org/#explore 

For the required model `<model_name>` implemented in the `models` folder

- Execute the corresponding `.ipynb` file to obtain the model library and serialized binary files. In case of error in execution, execute the model generation/conversions commands in the notebook on terminal.
- After execution, the model's .so file can be obtained from `models\<model_name>\models\model_libs2\aarch64-android` and serialized binary file can be obtained from `models\<model_name>\output` folder.
- Copy both the model files to `app\src\main\assets` and `app\src\main\jniLibs\arm64-v8a` folders, for utilizing in application.
	
## Code Implementation

This application opens a camera preview, collects all the frames and converts them to bitmap. The network is built via Neural Network builder by passing model name and runtime as the input. The bitmap is then given to the model for inference, which returns object prediction and localization of the respective object.


### Prerequisite for Camera Preview.

Permission to obtain camera preview frames is granted in the following file:
```python
/app/src/main/AndroidManifest.xml
<uses-permission android:name="android.permission.CAMERA" />
 ```
In order to use camera2 APIs, add the below feature
```python
<uses-feature android:name="android.hardware.camera2" />
```
### Loading Model
Function for neural network connection and loading model, in `inference.cpp`:
```cpp
std::string build_network(const char * modelPath_cstr, const char* backEndPath_cstr, char* buffer, long bufferSize)
```
### Preprocessing
The bitmap image is passed as openCV Mat to native and then converted to BGR Mat. Models can work with specific image sizes.
Therefore, we need to resize the input image to the size accepted by the corresponding selected model before passing image.
Below code reference for YoloNAS preprocessing. Similarly for other models based on model requirements, the preprocessing may change.
```cpp
    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,img,cv::Size(dims[1],dims[0]), 0, 0, cv::INTER_LINEAR); //Resizing based on input
        LOGI("inputimage SIZE width::%d height::%d channels::%d",img.cols, img.rows, img.channels());

        float inputScale = 0.00392156862745f;    //normalization value, this is 1/255

        //opencv read in BGRA by default
        cvtColor(img, img, CV_BGRA2BGR);
        img.convertTo(img,CV_32FC3,inputScale);
 ```
 
 ## PostProcessing
 This included getting the class with highest confidence for each boxes and applying Non-Max Suppression to remove overlapping boxes.
 Below code reference for YoloNAS postprocessing. Similarly for other models based on model requirements, the postprocessing may change.
 
 ```python
    for(int i =0;i<(2100);i++)
    {
        int start = i*80;
        int end = (i+1)*80;

        auto it = max_element (BBout_class.begin()+start, BBout_class.begin()+end);
        int index = distance(BBout_class.begin()+start, it);

        std::string classname = classnamemapping[index];
        if(*it>=0.5 )
        {
            int x1 = BBout_boxcoords[i * 4 + 0];
            int y1 = BBout_boxcoords[i * 4 + 1];
            int x2 = BBout_boxcoords[i * 4 + 2];
            int y2 = BBout_boxcoords[i * 4 + 3];
            Boxlist.push_back(BoxCornerEncoding(x1, y1, x2, y2,*it,classname));
        }
    }

    std::vector<BoxCornerEncoding> reslist = NonMaxSuppression(Boxlist,0.20);
```
then we just scale the coords for original image

```python
        float top,bottom,left,right;
        left = reslist[k].y1 * ratio_1;   //y1
        right = reslist[k].y2 * ratio_1;  //y2

        bottom = reslist[k].x1 * ratio_2;  //x1
        top = reslist[k].x2 * ratio_2;   //x2
```

## Drawing bounding boxes

```python
 RectangleBox rbox = boxlist.get(j);
            float y = rbox.left;
            float y1 = rbox.right;
            float x =  rbox.top;
            float x1 = rbox.bottom;

            String fps_textLabel = "FPS: "+String.valueOf(rbox.fps);
            canvas.drawText(fps_textLabel,10,70,mTextColor);

            String processingTimeTextLabel= rbox.processing_time+"ms";

            canvas.drawRect(x1, y, x, y1, mBorderColor);
            canvas.drawText(rbox.label,x1+10, y+40, mTextColor);
            canvas.drawText(processingTimeTextLabel,x1+10, y+90, mTextColor);
```
	    
# Build and run with Android Studio

## Build APK file with Android Studio 

1. Clone this repo.
2. Set the environment variable `QNN_SDK_ROOT` by running the command `export QNN_SDK_ROOT=<QNN_SDK_PATH_HERE>`, suitably replacing the placeholder. After that, execute `bash resolveDependencies.sh`.
    * This script will download opencv and paste to sdk directory, to enable OpenCv for android Java.
    * This script will copy all necessary library files to `app\src\main\jniLibs\arm64-v8a` for execution.
3. Generate model files using the steps mentioned.
4. Import folder 03-ObjectDetection as a project in Android Studio
5. Do gradle sync
6. Compile the project. 
7. Output APK file should get generated : app-debug.apk
8. Prepare the Qualcomm Innovators development kit to install the application (Do not run APK on emulator)

9. If Unsigned or Signed DSP runtime is not getting detected, then please check the logcat logs for the FastRPC error. DSP runtime may not get detected due to SE Linux security policy. Please try out following commands to set permissive SE Linux policy.

It is recommended to run below commands.
```java
adb disable-verity
adb reboot
adb root
adb remount
adb shell setenforce 0
```

9. Install and test application : app-debug.apk
```java
adb install -r -t app-debug.apk
```

10. launch the application

Following is the basic "Pose Detection" Android App 

1. On launch of application, from home screen user can select the model and runtime and then press start camera button.
2. On first launch of camera, user needs to provide camera permissions.
3. After camera launched, the selected model with runtime starts loading in the background. User will see a dialogue box till model is being loaded.
4. Once the model is loaded, it will start detecting objects and box will be seen around the object if respective object is detected on the screen 
5. User can go back to home screen by pressing back button and select appropriate model and run-time and observe performance difference.

Same results for the application are : 

## Demo of the application
![Demo video.](.//demo/ObjectDetectYoloNAS.gif)

# References
1. SSD - Single shot Multi box detector - https://arxiv.org/pdf/1512.02325.pdf
2. https://github.com/Deci-AI/super-gradients
3. https://zenodo.org/record/7789328

	
###### *Snapdragon and Qualcomm Neural Processing SDK are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*