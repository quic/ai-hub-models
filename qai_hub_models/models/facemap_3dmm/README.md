[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Facial-Landmark-Detection: Facial landmark predictor with 3DMM](https://aihub.qualcomm.com/models/facemap_3dmm)

Real-time 3D facial landmark detection optimized for mobile and edge.

This is based on the implementation of Facial-Landmark-Detection found [here](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/facemap_3dmm/model.py). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/facemap_3dmm).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Before installing, make sure you are running Python 3.10
```bash
python --version
```

Install the requirements via pip:

```bash
pip install scikit-image
```

Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.facemap_3dmm.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.facemap_3dmm.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.


## License
* The license for the original implementation of Facial-Landmark-Detection can be found
  [here](https://github.com/qcom-ai-hub/ai-hub-models-internal/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [None](None)
* [Source Model Implementation](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/facemap_3dmm/model.py)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


