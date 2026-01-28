# [Distil-Bert-Base-Uncased-Hf: Language model for masked language modeling and general-purpose NLP tasks](https://aihub.qualcomm.com/models/distil_bert_base_uncased_hf)

DistilBERT is a lightweight BERT model designed for efficient self-supervised learning of language representations. It can be used for masked language modeling and as a backbone for various NLP tasks.

This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/distil_bert_base_uncased_hf).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[distil-bert-base-uncased-hf]"
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.distil_bert_base_uncased_hf.demo { --quantize w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.distil_bert_base_uncased_hf.export { --quantize w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Distil-Bert-Base-Uncased-Hf can be found
  [here](https://huggingface.co/distilbert/distilbert-base-uncased/blob/f8354bcfbd3068faf2c5149654881cb4214e931d/LICENSE).


## References
* [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
