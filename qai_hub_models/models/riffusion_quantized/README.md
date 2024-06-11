[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Riffusion: State-of-the-art generative AI model used to generate spectrogram images given any text input. These spectrograms can be converted into audio clips](https://aihub.qualcomm.com/models/riffusion_quantized)

Generates high resolution spectrograms images from text prompts using a latent diffusion model. This model uses CLIP ViT-L/14 as text encoder, U-Net based latent denoising, and VAE based decoder to generate the final image.

This is based on the implementation of Riffusion found
[here](https://github.com/CompVis/stable-diffusion/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/riffusion_quantized).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[riffusion_quantized]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.riffusion_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.riffusion_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Riffusion can be found
  [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)

## References
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
* [Source Model Implementation](https://github.com/CompVis/stable-diffusion/tree/main)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
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


