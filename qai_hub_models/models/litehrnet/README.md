# [LiteHRNet: Human pose estimation](https://aihub.qualcomm.com/models/litehrnet)

LiteHRNet is a machine learning model that detects human pose and returns a location and confidence for each of 17 joints.

This is based on the implementation of LiteHRNet found [here](https://github.com/HRNet/Lite-HRNet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/litehrnet).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[litehrnet]" torch==2.4.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.4/index.html -f https://qaihub-public-python-wheels.s3.us-west-2.amazonaws.com/index.html
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.litehrnet.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.litehrnet.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of LiteHRNet can be found
  [here](https://github.com/HRNet/Lite-HRNet/blob/hrnet/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403)
* [Source Model Implementation](https://github.com/HRNet/Lite-HRNet)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
