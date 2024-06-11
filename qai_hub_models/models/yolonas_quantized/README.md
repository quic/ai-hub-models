[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Yolo-NAS-Quantized: Quantized real-time object detection optimized for mobile and edge](https://aihub.qualcomm.com/models/yolonas_quantized)

YoloNAS is a machine learning model that predicts bounding boxes and classes of objects in an image. This model is post-training quantized to int8 using samples from the COCO dataset.

This is based on the implementation of Yolo-NAS-Quantized found
[here](https://github.com/Deci-AI/super-gradients). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/yolonas_quantized).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[yolonas_quantized]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.yolonas_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.yolonas_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Yolo-NAS-Quantized can be found
  [here](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)

## References
* [YOLO-NAS by Deci Achieves SOTA Performance on Object Detection Using Neural Architecture Search](https://deci.ai/blog/yolo-nas-object-detection-foundation-model/)
* [Source Model Implementation](https://github.com/Deci-AI/super-gradients)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


