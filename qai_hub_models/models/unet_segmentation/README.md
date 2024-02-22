[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Unet-Segmentation: Real-time segmentation optimized for mobile and edge](https://aihub.qualcomm.com/models/unet_segmentation)

UNet is a machine learning model that produces a segmentation mask for an image. The most basic use case will label each pixel in the image as being in the foreground or the background. More advanced usage will assign a class label to each pixel.

This is based on the implementation of Unet-Segmentation found
[here](https://github.com/milesial/Pytorch-UNet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/unet_segmentation).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.unet_segmentation.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.unet_segmentation.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of Unet-Segmentation can be found
  [here](https://github.com/milesial/Pytorch-UNet/blob/master/LICENSE).


## References
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [Source Model Implementation](https://github.com/milesial/Pytorch-UNet)
