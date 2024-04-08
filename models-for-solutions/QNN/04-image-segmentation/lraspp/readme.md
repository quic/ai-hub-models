# Semantic segmentation LRASPP_Mobilenetv3_large


## Pre-requisites

* Please follow the instructions for setting up  Qualcomm® AI Engine Direct using the [link](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/setup.html) provided. 
- Install onnx v1.6.0 ```pip install onnx==1.6.0```.
- Install onnxsim ```pip install onnxsim``` and onnxruntime ```pip install onnxruntime```.
- Install OpenCV ```pip install cv2```
- Install mxnet ```pip install mxnet```


## How to get the model ? 

for ONNX model follow  attached notebook  
- You Need to change 2 layer Hardsigmoid and hardswish Layer(It'll Take 10 Minutes)
- https://github.com/quic/qidk/tree/master/Model-Enablement/Model-Conversion-Layer-Replacement
- Follow The above Link to add CustomHardSigmoid and CustomHardswish Layer

## Convert Model to library file / serialized binary

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary

## Quantization of model

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary


## Accuracy Analysis

- To check results please run the jupyter notebook
- To run any jupyter notebook, run below command. It will generate few links on the screen, pick the link with your machine name on it (host-name) and paste it in any browser.
- Navigate to the notebook ".ipynb" file and simply click that file.
```python
jupyter notebook --no-browser --port=8080 --ip 0.0.0.0 --allow-root
```


###### *Snapdragon and Qualcomm AI Engine Direct are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*
