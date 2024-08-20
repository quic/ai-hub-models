[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [MediaPipe-Pose-Estimation: Detect and track human body poses in real-time images and video streams](https://aihub.qualcomm.com/models/mediapipe_pose)

The MediaPipe Pose Landmark Detector is a machine learning pipeline that predicts bounding boxes and pose skeletons of poses in an image.

This is based on the implementation of MediaPipe-Pose-Estimation found
[here](https://github.com/zmurez/MediaPipePyTorch/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/mediapipe_pose).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.mediapipe_pose.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.mediapipe_pose.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of MediaPipe-Pose-Estimation can be found
  [here](https://github.com/zmurez/MediaPipePyTorch/blob/master/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)

## References
* [BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/abs/2006.10204)
* [Source Model Implementation](https://github.com/zmurez/MediaPipePyTorch/)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


