[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Segformer_Base: Real-time object segmentation](https://aihub.qualcomm.com/models/segformer_base)

Segformer Base is a machine learning model that predicts masks and classes of objects in an image.

This is based on the implementation of Segformer_Base found [here](https://github.com/huggingface/transformers/tree/main/src/transformers/models/segformer). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/segformer_base).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.segformer_base.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.segformer_base.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.


## License
* The license for the original implementation of Segformer_Base can be found
  [here](https://github.com/huggingface/transformers/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
* [Source Model Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/segformer)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


