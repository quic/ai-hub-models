# Object Detection with DETR Model

| Field | Description |
| --- | --- |
| Model Name | DETR |
| DNN Framwork | ONNX |
| Public Repo  | https://github.com/facebookresearch/detr |
| Paper        | NA |
| Accuracy Metric | box ap |

## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk"> Qualcomm® Neural Processing SDK (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 

## How to get the model ? 

For this demo, a ONNX version of DETR was used, execute this Python Script (generateModels.py). Once executed it will create models folder containing ONNX Model,
Quantized and Non-Quantized DLC.


# Accuracy analysis

- To check accuracy please run "detr_resnet101-accuracy-analysis.ipynb" jupyter notebook.
- To run any jupyter notebook, run below command. It will generate few links on the screen, pick the link with your machine name on it (host-name) and paste it in any browser.
- Navigate to the notebook ".ipynb" file and simply click that file.
```python
jupyter notebook --no-browser --port=8080 --ip 0.0.0.0 --allow-root
```

# References

1. DETR Model paper: https://arxiv.org/pdf/2104.01318.pdf
2. https://huggingface.co/docs/transformers/model_doc/detr
3. 2017 Train Val dataset:  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
