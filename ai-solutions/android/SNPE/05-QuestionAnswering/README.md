# OnDevice Question-Answering with Transformers

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Model Selection and DLC conversion](#1-model-preparation)
  1. Model Overview
  2. Steps to convert model to DLC
 
- [Build and Run with Android Studio](#4-build-and-run-with-android-studio)
  1. [Source Organization](###Source Organization)
  2. [Code implementataion](### Code Implementation)
- [Qualcomm® Neural Processing SDK C++ APIs JNI Integration](#qualcomm-neural-processing-sdk-c-apis-jni-integration)
- [Build APK file with Android Studio](## Build APK file With Android Stduio)
- [Reults](## Results)
- [Credits](#credits)
- [References](#references)

# Introduction

Question Answering (QA) is one of the common and challenging Natural Language Processing tasks. <br>
- Current project is an sample Android application for OnDevice Question Ansering using [Qualcomm® Neural Processing SDK for AI](https://developer.qualcomm.com/sites/default/files/docs/snpe/index.html) framework. 
-  We have used 3 Models in this Solution
  1. [Albert](https://github.com/google-research/ALBERT)
  2. [Mobilebert](https://github.com/gemde001/MobileBERT)
  3. [Electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)
- We Need to Give 3 Inputs, Input_ids, Attention_Mask, Token_Type_Ids of fixed Size(1,384)
-  All 3 Models are small, efficient and mobile friendly Transformer model fine-tuned on [SQUAD v2.0 dataset](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/) for **Q&A** downstream task
- In this project, we'll show how to efficiently convert, deploy and acclerate of these model on Snapdragon® platforms to perform Ondevice Question Answering.

<p align="center">
<img src="readme_assets/QA.gif" width=35% height=35%>
</p>

## Prerequisites
* Android Studio to import and build the project
* Android NDK "r19c" or "r21e" to build native code in Android Studio
* Python 3.6, PyTorch 1.10.1, Tensorflow 2.6.2, Transformers 4.18.0, Datasets 2.4.0 to prepare and validate the model<br>
  ###### <i>(above mentioned Python packages version and Android Studio version is just a recommendation and is not a hard requirement. Please install SDK dependencies in Python 3.6 virtual environment) </i>
* [Qualcomm® Neural Processing Engine for AI SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) v2.x.x and its [dependencies](https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html) to integrate and accelerate the network on Snapdragon<br>
  ###### <i>(During developement of this tutorial, the AI SDK recommends Python 3.6 version and is subject to change with future SDK releases. Please refer SDK Release Notes.)</i>
  

# Quick Start

## 1. Model Preparation

### 1.1 Generate Electra-small model as Tensorflow Frozen Graph:

```
python scripts/qa_model_gen.py
```
Model will get generated at `./frozen_models` directory with name `electra_small_squad2.pb` having input Sequence_Length = `384` <br>

### 1.2 Generating Mobilebert and Albert Model:
Please go to scripts/generating_model.ipynb file to generate these 2 models.
<br>
#### 1.2 Setup the Qualcomm® Neural Processing SDK Environment:
```
source <snpe-sdk-location>/bin/envsetup.sh -t $TENSORFLOW_DIR
```

### 1.3 Convert generated frozen graph into DLC (Deep Learning Container):
```
snpe-tensorflow-to-dlc -i frozen_models/electra_small_squad2.pb -d input_ids 1,384 -d attention_mask 1,384 -d token_type_ids 1,384 --out_node Identity --out_node Identity_1 -o frozen_models/electra_small_squad2.dlc
```
```
snpe-onnx-to-dlc -i alberta-onnx/model.onnx -d input_ids 1,384 -d attention_mask 1,384 -d token_type_ids 1,384 -o alberta.dlc
```
```
snpe-onnx-to-dlc -i mobilebert-onnx/model.onnx -d input_ids 1,384 -d attention_mask 1,384 -d token_type_ids 1,384 -o mobile_bert.dlc
```
where "input_ids, attention_mask, token_type_ids" are input of the model.<br><br>
This command converts Tensorflow/onnx-model  into DLC format, which DSP, GPU And CPU accelerators can understand for running inference.<br>

###### <i>(If you are using a different Tensorflow version to generate PB file, it may be a case that Output Layer names gets changed. Please check once by visualizing graph using Netron viewer or any other visualization tools )</i> <br>


### 1.4 Offline Preparation (caching) of DLC (for optimizing model loading time on DSP accelerator)
If You're using it to lanai device please build it for sm8650.
```
snpe-dlc-graph-prepare --input_dlc frozen_models/electra_small_squad2.dlc --use_float_io --htp_archs v73 
```
```
snpe-dlc-graph-prepare --input_dlc alberta.dlc --input_list tf_raw_list.txt  --output_dlc alberta_float.dlc --set_output_tensors end_logits,start_logits --use_float_io --htp_socs sm8550
```
```
snpe-dlc-graph-prepare --input_dlc mobile_bert.dlc --input_list tf_raw_list.txt  --output_dlc mobile_bert_float.dlc --use_float_io --set_output_tensors end_logits,start_logits --htp_socs sm8550 
```
 <br>




## 2. Build and run with Android Studio
1. Clone QIDK repo. 
2. Generate DLC using the steps mentioned above
3. Copy "snpe-release.aar" file from android folder in "Qualcomm Neural Processing SDK for AI" release from Qualcomm Developer Network into this folder: NLPSolution1-QuestionAnswering\snpe-release\
4. Copy DLC generated in step-2 at : NLPSolution1-QuestionAnswering\QuestionAnswering\bert\src\main\assets\
5. Copy from SNPE_ROOT\lib\android\snpe-release\jni\arm64-v8a at :NLPSolution1-QuestionAnswering\QuestionAnswering\bert\src\main\jniLibs\

**Note- If you're using sm8650 then please take all the files from SNPE-2.12.1 otherwise take it from SNPE-2.12.0**

#### Open the `QuestionAnswering` directory in Android Studio and build the project
On opening the project, the Android Studio may ask you to download Android NDK which is needed for building the AI SDK C++ Native APIs.
On sucessfull completion of project sync and build process, press the play icon to install and run the app on connected device.

* If build process fails with `libSNPE.so` duplication error, then please change its path from "jniLibs" to "cmakeLibs" as follows : `${CMAKE_CURRENT_SOURCE_DIR}/../cmakeLibs/arm64-v8a/libSNPE.so` in `QuestionAnswering/bert/src/main/cpp/CMakeList.txt` under `target_link_libraries` section and delete `libSnpe.so` from "jniLibs" directory.

#### Manual APK Installation
If Android Studio is not able to detect the device or if device is in remote location and copy the APK to current directory:
```
cp ./QuestionAnswering/app/build/outputs/apk/debug/app-debug.apk ./qa-app.apk
```
```
adb install -r -t qa-app.apk
```

#### Debug Tips
* After installing the application, if it is crashing, try to collect the logs from QIDK device.
* To collect logs run the below commands.
	*	adb logcat -c
	* 	adb logcat > log.txt
	*	Now, run the app. Once, the app has crashed do Ctrl+C to terminate log collection.
	*	log.txt will be generated in current folder.
	*	Search for the keyword "crash" to analyze the error.

* On opening the app, if Unsigned or Signed DSP runtime is not getting detected, then please search the logcat logs with keywork `dsp` for the FastRPC errors.
* DSP runtime may not get detected due to SE Linux security policy in some Android builds. Please try out following commands to set `permissive` SE Linux policy.
```
adb disable-verity
adb reboot
adb root
adb remount
adb shell setenforce 0
// launch the application
```		

#### QA App Workflow 
Following is the basic Question Answering Android App.
* Select any Article from list of Articles on App Home screen
*	On Article selection instantiate SDK Network
* Select a Model(Alberta,Mobilebert,ElectraSmall)
*	Select desired runtime from drop down (for example, DSP,CPU)
*	Ask a Question and prepare input data for the model (input_ids, attention_mask, token_type_ids)
*	Execute the SDK Network
*	Post-process inference output and highlight Top1 answer in the Article

<p align="center">
<img src="readme_assets/1.png" width=20% height=20%>
<img src="readme_assets/2.png" width=20% height=20%>
<img src="readme_assets/3.png" width=20% height=20%>
<img src="readme_assets/4.png" width=20% height=20%>
</p>


## Qualcomm® Neural Processing SDK C++ APIs JNI Integration

Please refer to SDK Native application tutorial : https://developer.qualcomm.com/sites/default/files/docs/snpe/cplus_plus_tutorial.html

## Credits

The pre-trained model is from HuggingFace Repository by MRM8488 (https://huggingface.co/mrm8488/electra-small-finetuned-squadv2)

The app is forked from https://github.com/huggingface/tflite-android-transformers repository and uses the same
tokenizer with Electra-small model.

## References

- https://github.com/tensorflow/examples
- https://openreview.net/pdf?id=r1xMH1BtvB
- https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/
- https://github.com/huggingface/tflite-android-transformers
- https://huggingface.co/google/electra-small-discriminator
- https://huggingface.co/mrm8488/electra-small-finetuned-squadv2
- https://developer.qualcomm.com/sites/default/files/docs/snpe/index.html
- https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html
- https://developer.qualcomm.com/sites/default/files/docs/snpe/cplus_plus_tutorial.html
- https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/learning-resources/vision-based-ai-use-cases/performance-analysis-using-benchmarking-tools



###### *Qualcomm Neural Processing SDK is a product of Qualcomm Technologies, Inc. and/or its subsidiaries.*
