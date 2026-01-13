# [Segformer-Base: Real-time object segmentation](https://aihub.qualcomm.com/models/segformer_base)

Segformer Base is a machine learning model that predicts masks and classes of objects in an image.

This is based on the implementation of Segformer-Base found [here](https://github.com/huggingface/transformers/tree/main/src/transformers/models/segformer). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/segformer_base).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.segformer_base.demo { --quantize w8a16, w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.segformer_base.export { --quantize w8a16, w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Segformer-Base can be found
  [here](https://github.com/huggingface/transformers/blob/main/LICENSE).


## References
* [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
* [Source Model Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/segformer)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
