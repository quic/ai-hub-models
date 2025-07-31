# [Stable-Diffusion-v1.5: State-of-the-art generative AI model used to generate detailed images conditioned on text descriptions](https://aihub.qualcomm.com/models/stable_diffusion_v1_5)

Generates high resolution images from text prompts using a latent diffusion model. This model uses CLIP ViT-L/14 as text encoder, U-Net based latent denoising, and VAE based decoder to generate the final image.

This is based on the implementation of Stable-Diffusion-v1.5 found [here](https://github.com/CompVis/stable-diffusion/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/stable_diffusion_v1_5).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploy to Snapdragon X Elite NPU
Please follow the [Stable Diffusion Windows App](https://github.com/quic/ai-hub-apps/tree/main/apps/windows/python/StableDiffusion) tutorial to quantize model with custom weights.

## Quantize and Deploy Your Own Fine-Tuned Stable Diffusion

Please follow the [Quantize Stable Diffusion]({REPOSITORY_URL}/tutorials/stable_diffusion/quantize_stable_diffusion.md) tutorial to quantize model with custom weights.



## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[stable-diffusion-v1-5]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.stable_diffusion_v1_5.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.stable_diffusion_v1_5.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Stable-Diffusion-v1.5 can be found
  [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)


## References
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
* [Source Model Implementation](https://github.com/CompVis/stable-diffusion/tree/main)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation
