[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [ControlNet: Generating visual arts from text prompt and input guiding image](https://aihub.qualcomm.com/models/controlnet_quantized)

On-device, high-resolution image synthesis from text and image prompts. ControlNet guides Stable-diffusion with provided input image to generate accurate images from given input prompt.

This is based on the implementation of ControlNet found
[here](https://github.com/lllyasviel/ControlNet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/controlnet_quantized).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[controlnet_quantized]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.controlnet_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.controlnet_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of ControlNet can be found
  [here](https://github.com/lllyasviel/ControlNet/blob/main/LICENSE).


## References
* [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
* [Source Model Implementation](https://github.com/lllyasviel/ControlNet)
