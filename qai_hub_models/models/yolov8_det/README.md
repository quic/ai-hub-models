[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Yolo-v8-Detection: Real-time object detection optimized for mobile and edge](https://aihub.qualcomm.com/models/yolov8_det)

YoloV8 is a machine learning model that predicts bounding boxes and classes of objects in an image.

This is based on the implementation of Yolo-v8-Detection found
[here](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/yolo/detect). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/yolov8_det).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[yolov8_det]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.yolov8_det.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.yolov8_det.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of Yolo-v8-Detection can be found
  [here](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).


## References
* [Real-Time Flying Object Detection with YOLOv8](https://arxiv.org/abs/2305.09972)
* [Source Model Implementation](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/yolo/detect)
