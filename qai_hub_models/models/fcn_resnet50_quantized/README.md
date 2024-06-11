[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [FCN-ResNet50-Quantized: Quantized fully-convolutional network model for image segmentation](https://aihub.qualcomm.com/models/fcn_resnet50_quantized)

FCN_ResNet50 is a quantized machine learning model that can segment images from the COCO dataset. It uses ResNet50 as a backbone.

This is based on the implementation of FCN-ResNet50-Quantized found
[here](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/fcn_resnet50_quantized).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[fcn_resnet50_quantized]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.fcn_resnet50_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.fcn_resnet50_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of FCN-ResNet50-Quantized can be found
  [here](https://github.com/pytorch/vision/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)

## References
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
* [Source Model Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


