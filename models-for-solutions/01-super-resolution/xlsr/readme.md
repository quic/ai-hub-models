# Image Super Resolution - XLSR

| Field | Description |
| --- | --- |
| Model Name | XLSR |
| DNN Framwork | ONNX |
| Public Repo  | https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/xlsr/XLSR.md |
| Paper        | https://arxiv.org/abs/2105.10288 |
| Accuracy Metric | PSNR |
| Input Resolution | 128 x 128 |
| Output Resolution | 512 x 512 |
| Pre-Process | Resize, Normalize on tensor input; unsqueeze and transpose |
| post-Process| reshape, transpose, max prediction value, decoding depending on dataset, image from array |

## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk"> Qualcomm® Neural Processing SDK (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 

## Pre-Trained Model

Please refer to notebook for detailed steps to prepare Pre-Trained Model

## Convert Model to DLC

Please refer to notebook for detailed steps to converting pre-trained model to DLC

## Quantization of DLC

Please refer to notebook for detailed steps to converting pre-trained model to DLC

## Make Inference, Verify output. 

Please refer to notebook for detailed steps to making inference, verifying model output

###### *Snapdragon and Qualcomm Neural Processing SDK are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*