# Low Light Image Enhancement using RUAS

| Field | Description |
| --- | --- |
| Model Name | RUAS |
| DNN Framwork | ONNX |
| Public Repo  |  https://github.com/dut-media-lab/RUAS.git |
| Paper        | NA |
| Accuracy Metric | PSNR |
| Pre-Process | cv2.resize, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.clip |
| post-Process| np.reshape, np.clip, transpose |

## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct"> Qualcomm® AI Engine Direct (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 


## Changes to open source Repository

Changes made to open-source repository to generate pre-trained models is given as a patch file - RUAS.patch

## Pre-Trained Model

Please refer to python file 'generate_model.py' for detailed steps to prepare Pre-Trained Model

## Convert Model to library file / serialized binary

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary

## Quantization of model

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary


## Make Inference, Verify output. 

Please refer to notebook for detailed steps to making inference, verifying model output

###### *Snapdragon and Qualcomm AI Engine Direct are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*
