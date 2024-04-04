# Image Super Resolution - quicksrnet_small

| Field | Description |
| --- | --- |
| Model Name | quicksrnet_small |
| DNN Framwork | ONNX |
| Public Repo  | https://github.com/quic/aimet-model-zoo/ |
| Paper        | https://arxiv.org/abs/2303.04336 |
| Accuracy Metric | PSNR |
| Input Resolution | 128 x 128 |
| Output Resolution | 512 x 512 |
| Pre-Processing | Resize, Normalize on tensor input; unsqueeze and transpose |
| Post-Processing | reshape, transpose, max prediction value, decoding depending on dataset, image from array |



## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct"> Qualcomm® AI Engine Direct (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 

-  Please follow the instructions for setting up  Qualcomm® AI Engine Direct using the [link] (https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/setup.html) provided. 

- Install onnx v1.6.0 ```pip install onnx==1.6.0```.

- Install onnxsim ```pip install onnxsim``` and onnxruntime ```pip install onnxruntime```.

- Install OpenCV ```pip install cv2```

- Install mxnet ```pip install mxnet```

## Pre-Trained Model

Please refer to notebook for detailed steps to prepare Pre-Trained Model

## Convert Model to library file / serialized binary

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary

## Quantization of model

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary


## Make Inference, Verify output. 

Please refer to notebook for detailed steps to making inference, verifying model output

###### *Snapdragon and Qualcomm AI Engine Direct are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*