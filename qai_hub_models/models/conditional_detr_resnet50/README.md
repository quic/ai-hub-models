[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Conditional-DETR-ResNet50: Transformer based object detector with ResNet50 backbone](https://aihub.qualcomm.com/models/conditional_detr_resnet50)

DETR is a machine learning model that can detect objects (trained on COCO dataset).

This is based on the implementation of Conditional-DETR-ResNet50 found [here](https://github.com/huggingface/transformers/tree/main/src/transformers/models/conditional_detr). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/conditional_detr_resnet50).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[conditional-detr-resnet50]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.conditional_detr_resnet50.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.conditional_detr_resnet50.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Conditional-DETR-ResNet50 can be found
  [here](https://github.com/huggingface/transformers/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Conditional {DETR} for Fast Training Convergence](https://arxiv.org/abs/2108.06152)
* [Source Model Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/conditional_detr)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
