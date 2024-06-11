[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Real-ESRGAN-x4plus: Upscale images and remove image noise](https://aihub.qualcomm.com/models/real_esrgan_x4plus)

Real-ESRGAN is a machine learning model that upscales an image with minimal loss in quality. The implementation is a derivative of the Real-ESRGAN-x4plus architecture, a larger and more powerful version compared to the Real-ESRGAN-general-x4v3 architecture.

This is based on the implementation of Real-ESRGAN-x4plus found
[here](https://github.com/xinntao/Real-ESRGAN). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/real_esrgan_x4plus).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[real_esrgan_x4plus]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.real_esrgan_x4plus.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.real_esrgan_x4plus.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Real-ESRGAN-x4plus can be found
  [here](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)

## References
* [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
* [Source Model Implementation](https://github.com/xinntao/Real-ESRGAN)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


