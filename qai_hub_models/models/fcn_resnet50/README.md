[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [FCN_ResNet50: Fully-convolutional network model for image segmentation](https://aihub.qualcomm.com/models/fcn_resnet50)

FCN_ResNet50 is a machine learning model that can segment images from the COCO dataset. It uses ResNet50 as a backbone.

This is based on the implementation of FCN_ResNet50 found
[here](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/fcn_resnet50).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.fcn_resnet50.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.fcn_resnet50.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of FCN_ResNet50 can be found
  [here](https://github.com/pytorch/vision/blob/main/LICENSE).


## References
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
* [Source Model Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py)
