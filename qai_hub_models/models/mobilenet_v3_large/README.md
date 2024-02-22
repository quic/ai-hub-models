[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [MobileNet-v3-Large: Imagenet classifier and general purpose backbone](https://aihub.qualcomm.com/models/mobilenet_v3_large)

MobileNetV3Large is a machine learning model that can classify images from the Imagenet dataset. It can also be used as a backbone in building more complex models for specific use cases.

This is based on the implementation of MobileNet-v3-Large found
[here](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/mobilenet_v3_large).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.mobilenet_v3_large.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.mobilenet_v3_large.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of MobileNet-v3-Large can be found
  [here](https://github.com/pytorch/vision/blob/main/LICENSE).


## References
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
* [Source Model Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)
