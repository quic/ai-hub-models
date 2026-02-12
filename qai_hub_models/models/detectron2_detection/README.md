# [Detectron2-Detection: A next-generation library for object detection](https://aihub.qualcomm.com/models/detectron2_detection)

Detectron2-Detection is a machine learning model that can detect objects (trained on COCO dataset).

This is based on the implementation of Detectron2-Detection found [here](https://github.com/facebookresearch/detectron2/).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/detectron2_detection).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install wheel==0.45.1 "torch>=2.1,<2.9.0" "setuptools>=77.0.3"
pip install "qai-hub-models[detectron2-detection]" git+https://github.com/facebookresearch/detectron2.git@d38d716 --no-build-isolation
```

### 2. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.detectron2_detection.demo { --quantize w8a8, w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

By default, the demo will run locally in PyTorch. Pass `--eval-mode on-device` to the demo script to run the model on a cloud-hosted target device.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.detectron2_detection.export { --quantize w8a8, w8a16 }
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Detectron2-Detection can be found
  [here](https://github.com/facebookresearch/detectron2/blob/main/LICENSE).

## References
* [Source Model Implementation](https://github.com/facebookresearch/detectron2/)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
