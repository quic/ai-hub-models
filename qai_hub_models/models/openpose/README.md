[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [OpenPose: Human pose estimation](https://aihub.qualcomm.com/models/openpose)

OpenPose is a machine learning model that estimates body and hand pose in an image and returns location and confidence for each of 19 joints.

This is based on the implementation of OpenPose found [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/openpose).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[openpose]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.openpose.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.openpose.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.


## License
* The license for the original implementation of OpenPose can be found
  [here](https://cmu.flintbox.com/technologies/b820c21d-8443-4aa2-a49f-8919d93a8740).
* The license for the compiled assets for on-device deployment can be found [here](https://cmu.flintbox.com/technologies/b820c21d-8443-4aa2-a49f-8919d93a8740)


## References
* [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008)
* [Source Model Implementation](https://github.com/CMU-Perceptual-Computing-Lab/openpose)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
