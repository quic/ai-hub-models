[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [FFNet-40S-Quantized: Semantic segmentation for automotive street scenes](https://aihub.qualcomm.com/models/ffnet_40s_quantized)

FFNet-40S-Quantized is a "fuss-free network" that segments street scene images with per-pixel classes like road, sidewalk, and pedestrian. Trained on the Cityscapes dataset.

This is based on the implementation of FFNet-40S-Quantized found
[here](https://github.com/Qualcomm-AI-research/FFNet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/ffnet_40s_quantized).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.ffnet_40s_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.ffnet_40s_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of FFNet-40S-Quantized can be found
  [here](https://github.com/Qualcomm-AI-research/FFNet/blob/master/LICENSE).


## References
* [Simple and Efficient Architectures for Semantic Segmentation](https://arxiv.org/abs/2206.08236)
* [Source Model Implementation](https://github.com/Qualcomm-AI-research/FFNet)
