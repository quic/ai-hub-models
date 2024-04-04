# Image Super Resolution - quicksrnet_medium

| Field | Description |
| --- | --- |
| Model Name | quicksrnet_medium |
| DNN Framwork | ONNX |
| Public Repo  | https://github.com/quic/aimet-model-zoo/ |
| Paper        | https://arxiv.org/abs/2303.04336 |
| Accuracy Metric | PSNR |
| Input Resolution | 128 x 128 |
| Output Resolution | 512 x 512 |
| Pre-Processing | Resize, Normalize on tensor input; unsqueeze and transpose |
| Post-Processing | reshape, transpose, max prediction value, decoding depending on dataset, image from array |


## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk"> Qualcomm® Neural Processing SDK (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 

-  Please follow the instructions for setting up Qualcomm Neural Processing SDK using the [link] (https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html) provided. 
- Install onnx v1.6.0. Installation instruction can be found [here](https://qdn-drekartst.qualcomm.com/hardware/qualcomm-innovators-development-kit/frameworks-qualcomm-neural-processing-sdk-for-ai)

- Install onnxsim ```pip install onnxsim``` and onnxruntime ```pip install onnxruntime```.

- Install OpenCV ```pip install cv2```

- Install mxnet ```pip install mxnet```

## Pre-Trained Model

Please refer to notebook for detailed steps to prepare Pre-Trained Model

## Convert Model to DLC

Please refer to notebook for detailed steps to converting pre-trained model to DLC

## Quantization of DLC

Please refer to notebook for detailed steps to converting pre-trained model to DLC

## Make Inference, Verify output. 

Please refer to notebook for detailed steps to making inference, verifying model output

###### *Snapdragon and Qualcomm Neural Processing SDK are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*
