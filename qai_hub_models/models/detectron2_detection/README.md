# [Detectron2-Detection: A next-generation library for object detection](https://aihub.qualcomm.com/models/detectron2_detection)

Detectron2-Detection is a machine learning model that can detect objects (trained on COCO dataset).

This is based on the implementation of Detectron2-Detection found [here](https://github.com/facebookresearch/detectron2/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/detectron2_detection).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install wheel==0.45.1 "torch>=2.1,<2.9.0" "setuptools>=77.0.3"
pip install "qai-hub-models[detectron2-detection]" git+https://github.com/facebookresearch/detectron2.git@d38d716 --no-build-isolation
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.detectron2_detection.demo { --quantize w8a8, w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

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
