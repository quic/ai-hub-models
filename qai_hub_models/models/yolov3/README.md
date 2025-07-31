# [Yolo-v3: Real-time object detection optimized for mobile and edge](https://aihub.qualcomm.com/models/yolov3)

YoloV3 is a machine learning model that predicts bounding boxes and classes of objects in an image.

This is based on the implementation of Yolo-v3 found [here](https://github.com/ultralytics/yolov3/tree/v8). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/yolov3).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[yolov3]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.yolov3.demo { --quantize w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.yolov3.export { --quantize w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Yolo-v3 can be found
  [here](https://github.com/ultralytics/yolov3/blob/v8/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/ultralytics/yolov3/blob/v8/LICENSE)


## References
* [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
* [Source Model Implementation](https://github.com/ultralytics/yolov3/tree/v8)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
