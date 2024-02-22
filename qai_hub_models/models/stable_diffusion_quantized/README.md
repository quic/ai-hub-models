[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Stable-Diffusion: State-of-the-art generative AI model used to generate detailed images conditioned on text descriptions](https://aihub.qualcomm.com/models/stable_diffusion_quantized)

Generates high resolution images from text prompts using a latent diffusion model. This model uses CLIP ViT-L/14 as text encoder, U-Net based latent denoising, and VAE based decoder to generate the final image.

This is based on the implementation of Stable-Diffusion found
[here](https://github.com/CompVis/stable-diffusion/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/stable_diffusion_quantized).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[stable_diffusion_quantized]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.stable_diffusion_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.stable_diffusion_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of Stable-Diffusion can be found
  [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).


## References
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
* [Source Model Implementation](https://github.com/CompVis/stable-diffusion/tree/main)
