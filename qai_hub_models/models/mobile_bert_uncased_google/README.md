# [Mobile-Bert-Uncased-Google: Language model for masked language modeling and general-purpose NLP tasks](https://aihub.qualcomm.com/models/mobile_bert_uncased_google)

MOBILEBERT is a lightweight BERT model designed for efficient self-supervised learning of language representations. It can be used for masked language modeling and as a backbone for various NLP tasks.

This is based on the implementation of Mobile-Bert-Uncased-Google found [here](https://github.com/google-research/google-research/tree/master/mobilebert). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/mobile_bert_uncased_google).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[mobile-bert-uncased-google]"
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.mobile_bert_uncased_google.demo { --quantize w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.mobile_bert_uncased_google.export { --quantize w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Mobile-Bert-Uncased-Google can be found
  [here](https://github.com/google-research/google-research/blob/master/LICENSE).


## References
* [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984)
* [Source Model Implementation](https://github.com/google-research/google-research/tree/master/mobilebert)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
