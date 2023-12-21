# Object Detection with YOLOX Model

| Field | Description |
| --- | --- |
| Model Name | YoloX |
| DNN Framwork | ONNX |
| Public Repo  | https://github.com/Megvii-BaseDetection/YOLOX/ |
| Paper        | https://arxiv.org/abs/2107.08430 |
| Accuracy Metric | mAP |

## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk"> Qualcomm® Neural Processing SDK (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 

## How to get the onnx model from opensource ? 

```python
!wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.onnx
```

# Accuracy analysis

- To check accuracy please run "Accuracy_analyzer.ipynb" jupyter notebook.
- To run any jupyter notebook, run below command. It will generate few links on the screen, pick the link with your machine name on it (host-name) and paste it in any browser.
- Navigate to the notebook ".ipynb" file and simply click that file.
```python
jupyter notebook --no-browser --port=8080 --ip 0.0.0.0 --allow-root
```

# References

1. YOLOX Model paper: https://arxiv.org/abs/2107.08430 
2. https://github.com/Megvii-BaseDetection/YOLOX/
