# [Unet-Segmentation: Real-time segmentation optimized for mobile and edge](https://aihub.qualcomm.com/models/unet_segmentation)

UNet is a machine learning model that produces a segmentation mask for an image. The most basic use case will label each pixel in the image as being in the foreground or the background. More advanced usage will assign a class label to each pixel. This version of the model was trained on the data from Kaggle's Carvana Image Masking Challenge (see https://www.kaggle.com/c/carvana-image-masking-challenge) and is used for vehicle segmentation.

This is based on the implementation of Unet-Segmentation found [here](https://github.com/milesial/Pytorch-UNet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/unet_segmentation).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install qai-hub-models
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.unet_segmentation.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.unet_segmentation.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Unet-Segmentation can be found
  [here](https://github.com/milesial/Pytorch-UNet/blob/master/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/milesial/Pytorch-UNet/blob/master/LICENSE)


## References
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [Source Model Implementation](https://github.com/milesial/Pytorch-UNet)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
