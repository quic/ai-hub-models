[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [RTMDet: Real-time object detection optimized for mobile and edge](https://aihub.qualcomm.com/models/rtmdet)

RTMDet is a highly efficient model for real-time object detection,capable of predicting both the bounding boxes and classes of objects within an image.It is highly optimized for real-time applications, making it reliable for industrial and commercial use

This is based on the implementation of RTMDet found [here](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/rtmdet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/rtmdet).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[rtmdet]" torch==2.4.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.4/index.html -f https://qaihub-public-python-wheels.s3.us-west-2.amazonaws.com/index.html
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.rtmdet.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.rtmdet.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of RTMDet can be found
  [here](https://github.com/open-mmlab/mmdetection/blob/3.x/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://github.com/open-mmlab/mmdetection/blob/3.x/README.md)
* [Source Model Implementation](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/rtmdet)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
