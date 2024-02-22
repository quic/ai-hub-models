[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [DDRNet23-Slim: Segment images or video by class in real-time on device](https://aihub.qualcomm.com/models/ddrnet23_slim)

DDRNet23Slim is a machine learning model that segments an image into semantic classes, specifically designed for road-based scenes. It is designed for the application of self-driving cars.

This is based on the implementation of DDRNet23-Slim found
[here](https://github.com/chenjun2hao/DDRNet.pytorch). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/ddrnet23_slim).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.ddrnet23_slim.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.ddrnet23_slim.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of DDRNet23-Slim can be found
  [here](https://github.com/chenjun2hao/DDRNet.pytorch/blob/main/LICENSE).


## References
* [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](https://arxiv.org/abs/2101.06085)
* [Source Model Implementation](https://github.com/chenjun2hao/DDRNet.pytorch)
