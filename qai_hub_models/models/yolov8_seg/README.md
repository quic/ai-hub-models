# [YOLOv8-Segmentation: Real-time object segmentation optimized for mobile and edge by Ultralytics](https://aihub.qualcomm.com/models/yolov8_seg)

Ultralytics YOLOv8 is a machine learning model that predicts bounding boxes, segmentation masks and classes of objects in an image.

This is based on the implementation of YOLOv8-Segmentation found [here](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/yolo/segment). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/yolov8_seg).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[yolov8-seg]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.yolov8_seg.demo { --quantize w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.yolov8_seg.export { --quantize w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of YOLOv8-Segmentation can be found
  [here](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)


## References
* [Ultralytics YOLOv8 Docs: Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
* [Source Model Implementation](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/yolo/segment)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
