# [Yolo-v7: Real-time object detection optimized for mobile and edge](https://aihub.qualcomm.com/models/yolov7)

YoloV7 is a machine learning model that predicts bounding boxes and classes of objects in an image.

This is based on the implementation of Yolo-v7 found [here](https://github.com/WongKinYiu/yolov7/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/yolov7).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[yolov7]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.yolov7.demo { --quantize w8a8, w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.yolov7.export { --quantize w8a8, w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Yolo-v7 can be found
  [here](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md)


## References
* [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
* [Source Model Implementation](https://github.com/WongKinYiu/yolov7/)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
