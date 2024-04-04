# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
    + [About "Image Segmentation"](#about-image-segmentation)
    + [Pre-Requisites](#pre-requisites)
- [Model Selection and DLC conversion](#model-selection-and-dlc-conversion)
    + [Model Overview](#model-overview)
    + [Steps to convert model to DLC](#steps-to-convert-model-to-dlc)
- [Source Overview](#source-overview)
    + [Source Organization](#source-organization)
    + [Code Implementation](#code-implementation)
- [Build APK file with Android Studio](#build-apk-file-with-android-studio)
- [Results](#results)

# Introduction

### About "Image Segmentation" 

- Current project is an sample Android application for AI-based Image Segmentation using [Qualcomm® Neural Processing SDK for AI](https://developer.qualcomm.com/sites/default/files/docs/snpe/index.html) framework. 
- We have used 5 Models in this Solution
- This sample segments objects in the Image.
- DLC models take only fixed input size.
- If users intend to use a different model in this demo framework, **image pre/post processing will be needed**. 
- Current pre/post processing is specific to the models used. 

### Pre-Requisites 

- Qualcomm® Neural Processing SDK for AI setup should be completed by following the guide here : https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html
- Android Studio to import sample project
- Android NDK to build native code
- Install opencv using ```pip install opencv-python```

# Model Selection and DLC conversion

### Model Overview

Please refer to Models repository for model overview
<TODO> Add public link

### Steps to convert model to DLC
Please refer to Models repository for model overview
<TODO> Add public link

# Source Overview

### Source Organization

- <DIR> demo: Contains demo video, GIF 
- <DIR> app: Contains source files in standard Android app format.
- app\src\main\assets : Contains Model binary DLC
- app\src\main\java\com\qcom\imageSegmentation : Application java source code
- app\src\main\cpp : Application C++(native) source code
- sdk : Contains openCV sdk (Will be generated using _ResolveDependencies.sh_ )
   
### Code Implementation

- Model Initialization
   
   `public boolean loadingMODELS(char runtime_var, String dlc_name)`
  - runtime_var: Possible options are D, G, C. 
  - dlc_name: Name of the DLC.
  
- Running Model
 
  - Following is the Java Function, that handles model execution. This function iternally calls sub functions to handle pre-processing and post-processing
     
      `inferSNPE(inputMat.getNativeObjAddr(), outputMat.getNativeObjAddr())`
       - inputMat is opencv Matrix that contains input image.
       - outputMat is the destination for the output image

   - C++ function that handles preprocessing for the input image.
   
       `void preprocess(std::vector<float32_t> &dest_buffer, cv::Mat &img)`
  
   - Model gives a segmented map that we can see in executeDLC function
      
       `void executeDLC(cv::Mat &img, int orig_width, int orig_height, float &inferenceTime, cv::Mat &destmat)`
   
   - SNPE API function that runs the network and give result 

       `snpe->execute(inputMap, outputMap);`


# Build APK file with Android Studio  
   
1. Clone this repo.
2. Generate DLC using the steps mentioned.
3. Run below script, from the directory where it is present, to resolve dependencies of this project.
   
	`bash resolveDependencies.sh`
 
   * This script will download opencv and paste to sdk directory, to enable OpenCv for android Java.
   * This script will copy snpe-release.aar file from $SNPE_ROOT to "snpe-release" directory in Android project.

	**NOTE - If you are using SNPE version 2.11 or greater, please change following line in resolveDependencies.sh.**
   ```
	From: cp $SNPE_ROOT/android/snpe-release.aar snpe-release
	To : cp $SNPE_ROOT/lib/android/snpe-release.aar snpe-release
	```

4. Import folder ImageSegmentation as a project in Android Studio 
5. Do gradle sync
6. Compile the project. 
7. Output APK file should get generated : app-debug.apk
8. Prepare the Qualcomm Innovators development kit(QIDK) to install the application (Do not run APK on emulator)
9. Install and test application : app-debug.apk

```java
adb install -r -t app-debug.apk
```

10. launch the application

Following is the basic "Image Segmentation" Android App 

1. Select one of the models
3. Select the run-time to run the model (CPU, GPU or DSP)
4. Observe the result of model on screen
5. Also note the performance indicator for the particular run-time in mSec

Same results for the application are shown below 

# Results

- Demo video, and performance details as seen below:
	
![Demo video.](demo/VisionSolution2-ImageSuperResolution.gif)

###### *Qualcomm Neural Processing SDK and Snapdragon are products of Qualcomm Technologies, Inc. and/or its subsidiaries. AIMET Model Zoo is a product of Qualcomm Innovation Center, Inc.*
