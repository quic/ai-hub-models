[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [HRNetPose: Perform accurate human pose estimation](https://aihub.qualcomm.com/models/hrnet_pose)

HRNet performs pose estimation in high-resolution representations.

This is based on the implementation of HRNetPose found
[here]({source_repo}). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/hrnet_pose).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[hrnet_pose]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.hrnet_pose.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.hrnet_pose.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.


## License
* The license for the original implementation of HRNetPose can be found
  [here](https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)
* [Source Model Implementation](https://github.com/quic/aimet-model-zoo/tree/develop/aimet_zoo_torch/hrnet_posenet)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


