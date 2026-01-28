# [InternImage: InternImage Large-Scale Vision Foundation Model](https://aihub.qualcomm.com/models/internimage)

InternImage employs DCNv3 as its core operator to equips the model with dynamic and effective receptive fields required for downstream tasks like object detection and segmentation, while enabling adaptive spatial aggregation.

This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/internimage).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.internimage.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.internimage.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of InternImage can be found
  [here](https://github.com/OpenGVLab/InternImage/tree/master?tab=MIT-1-ov-file).



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
