# [Facial-Landmark-Detection: Real-time 3D facial landmark detection optimized for mobile and edge](https://aihub.qualcomm.com/models/facemap_3dmm)

Detects facial landmarks (eg, nose, mouth, etc.). This model's architecture was developed by Qualcomm. The model was trained by Qualcomm on a proprietary dataset of faces, but can be used on any image.

This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/facemap_3dmm).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[facemap-3dmm]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.facemap_3dmm.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.facemap_3dmm.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Facial-Landmark-Detection can be found
  [here](https://github.com/quic/ai-hub-models/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
