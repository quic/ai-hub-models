# [Depth-Anything: Deep Convolutional Neural Network model for depth estimation](https://aihub.qualcomm.com/models/depth_anything)

Depth Anything is designed for estimating depth at each point in an image.

This is based on the implementation of Depth-Anything found [here](https://github.com/huggingface/transformers/tree/main/src/transformers/models/depth_anything). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/depth_anything).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[depth-anything]"
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.depth_anything.demo { --quantize w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.depth_anything.export { --quantize w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Depth-Anything can be found
  [here](https://github.com/huggingface/transformers/blob/main/LICENSE).


## References
* [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891)
* [Source Model Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/depth_anything)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
