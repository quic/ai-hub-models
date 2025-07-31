# [RTMPose-Body2d: Human pose estimation](https://aihub.qualcomm.com/models/rtmpose_body2d)

RTMPose is a machine learning model that detects human pose and returns a location and confidence for each of 133 joints.

This is based on the implementation of RTMPose-Body2d found [here](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/rtmpose_body2d).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[rtmpose-body2d]" torch==2.4.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.4/index.html -f https://qaihub-public-python-wheels.s3.us-west-2.amazonaws.com/index.html
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.rtmpose_body2d.demo { --quantize w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.rtmpose_body2d.export { --quantize w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of RTMPose-Body2d can be found
  [here](https://github.com/open-mmlab/mmpose/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose](https://arxiv.org/abs/2303.07399)
* [Source Model Implementation](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
