# [Segment-Anything-Model-2: High-quality segmentation in images and videos with real-time performance and minimal user interaction](https://aihub.qualcomm.com/models/sam2)

SAM 2, the successor to Meta's Segment Anything Model (SAM), is a cutting-edge tool designed for comprehensive object segmentation in both images and videos. It excels in handling complex visual data through a unified, promptable model architecture that supports real-time processing and zero-shot generalization.

This is based on the implementation of Segment-Anything-Model-2 found [here](https://github.com/facebookresearch/sam2). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/sam2).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[sam2]" git+https://github.com/facebookresearch/sam2.git@2b90b9f
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.sam2.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.sam2.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Segment-Anything-Model-2 can be found
  [here](https://github.com/facebookresearch/sam2/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [SAM 2 Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
* [Source Model Implementation](https://github.com/facebookresearch/sam2)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
