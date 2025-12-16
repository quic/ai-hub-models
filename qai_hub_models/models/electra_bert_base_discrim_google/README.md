# [Electra-Bert-Base-Discrim-Google: Language model for masked language modeling and general-purpose NLP tasks](https://aihub.qualcomm.com/models/electra_bert_base_discrim_google)

ELECTRABERT is a lightweight BERT model designed for efficient self-supervised learning of language representations. It can be used for identify unnatural or artificially modified text and as a backbone for various NLP tasks.

This is based on the implementation of Electra-Bert-Base-Discrim-Google found [here](https://github.com/google-research/electra/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/electra_bert_base_discrim_google).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[electra-bert-base-discrim-google]"
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.electra_bert_base_discrim_google.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.electra_bert_base_discrim_google.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Electra-Bert-Base-Discrim-Google can be found
  [here](https://github.com/google-research/electra/blob/master/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)
* [Source Model Implementation](https://github.com/google-research/electra/)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
