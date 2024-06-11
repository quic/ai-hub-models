[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Yolo-v6: Real-time object detection optimized for mobile and edge](https://aihub.qualcomm.com/models/yolov6)

YoloV6 is a machine learning model that predicts bounding boxes and classes of objects in an image.

This is based on the implementation of Yolo-v6 found
[here](https://github.com/meituan/YOLOv6/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/yolov6).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.




## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.yolov6.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.yolov6.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Yolo-v6 can be found
  [here](https://github.com/meituan/YOLOv6/blob/47625514e7480706a46ff3c0cd0252907ac12f22/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://github.com/meituan/YOLOv6/blob/47625514e7480706a46ff3c0cd0252907ac12f22/LICENSE)

## References
* [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976)
* [Source Model Implementation](https://github.com/meituan/YOLOv6/)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


