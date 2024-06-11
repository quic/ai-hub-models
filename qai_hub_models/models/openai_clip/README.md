[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [OpenAI-Clip: Multi-modal foundational model for vision and language tasks like image/text similarity and for zero-shot image classification](https://aihub.qualcomm.com/models/openai_clip)

Contrastive Language-Image Pre-Training (CLIP) uses a ViT like transformer to get visual features and a causal language model to get the text features. Both the text and visual features can then be used for a variety of zero-shot learning tasks.

This is based on the implementation of OpenAI-Clip found
[here](https://github.com/openai/CLIP/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/openai_clip).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[openai_clip]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.openai_clip.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.openai_clip.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of OpenAI-Clip can be found
  [here](https://github.com/openai/CLIP/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)

## References
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [Source Model Implementation](https://github.com/openai/CLIP/)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


