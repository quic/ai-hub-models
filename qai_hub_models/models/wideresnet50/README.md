# [WideResNet50: Imagenet classifier and general purpose backbone](https://aihub.qualcomm.com/models/wideresnet50)

WideResNet50 is a machine learning model that can classify images from the Imagenet dataset. It can also be used as a backbone in building more complex models for specific use cases.

This is based on the implementation of WideResNet50 found [here](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/wideresnet50).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install qai-hub-models
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.wideresnet50.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.wideresnet50.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of WideResNet50 can be found
  [here](https://github.com/pytorch/vision/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
* [Source Model Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
