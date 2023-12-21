# Object Detection with SSD-MobileNetV2 Model

| Field | Description |
| --- | --- |
| Model Name | SSD MobilenetV2 |
| DNN Framwork | ONNX |
| Public Repo  | https://github.com/lufficc/SSD.git |
| Paper        | NA |
| Accuracy Metric | mAP |

## Pre-requisites

* Please follow the instructions for setting up Qualcomm Neural Processing SDK using the [link](https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html) provided. 
* Tested this on SNPE-2.14.0
* To install caffe follow the instructions from this [link](https://qdn-drekartst.qualcomm.com/hardware/qualcomm-innovators-development-kit/frameworks-qualcomm-neural-processing-sdk-for-ai)
* Please make torchvision version as 0.9.1


## How to get the onnx model from opensource ? 


```python
git clone https://github.com/lufficc/SSD.git
cd SSD/
git reset --hard 68dc0a20efaf3997e58b616afaaaa21bf8ca3c05
wget https://github.com/lufficc/SSD/releases/download/1.2/mobilenet_v2_ssd320_voc0712_v2.pth
patch -i  ../changes_on_top_without_ABP-NMS.patch
python demo.py --config-file configs/mobilenet_v2_ssd320_voc0712.yaml --images
```



# Accuracy analysis

- To check accuracy please run "SSD MobileNetV2 Accuracy Analysis.ipynb" jupyter notebook.
- To run any jupyter notebook, run below command. It will generate few links on the screen, pick the link with your machine name on it (host-name) and paste it in any browser.
- Navigate to the notebook ".ipynb" file and simply click that file.
```python
jupyter notebook --no-browser --port=8080 --ip 0.0.0.0 --allow-root
```

# References

1. SSD MobileNetV2 Model paper: https://arxiv.org/abs/1801.04381
2. https://github.com/lufficc/SSD
3. 2017 Train Val dataset:  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
