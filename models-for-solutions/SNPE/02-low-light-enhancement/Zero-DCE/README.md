# Low Light Image Enhancement using Zero-DCE

| Field | Description |
| --- | --- |
| Model Name | Zero-DCE |
| DNN Framwork | ONNX |
| Public Repo  |  https://github.com/Li-Chongyi/Zero-DCE.git  |
| Paper        | NA |
| Accuracy Metric | PSNR |
| Pre-Process | cv2.resize, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.clip |
| post-Process| np.reshape, np.clip, transpose |

## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk"> Qualcomm® Neural Processing SDK (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 


## Changes to open source Repository

Changes made to open-source repository to generate pre-trained models is given as a patch file - Zero_DCE.patch

## Pre-Trained Model

Please refer to python file 'generate_model.py' for detailed steps to prepare Pre-Trained Model

## Convert Model to DLC

Please refer to notebook for detailed steps to converting pre-trained model to DLC

## Quantization of DLC

Please refer to notebook for detailed steps to converting pre-trained model to DLC

## Make Inference, Verify output. 

Please refer to notebook for detailed steps to making inference, verifying model output

###### *Snapdragon and Qualcomm Neural Processing SDK are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*