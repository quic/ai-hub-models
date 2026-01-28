# [OpusMT-En-Es: OpusMT English to Spanish neural machine translation model based on MarianMT transformer architecture](https://aihub.qualcomm.com/models/opus_mt_en_es)

OpusMT English to Spanish translation model is a state-of-the-art neural machine translation system designed for translating English text into Spanish. This model is based on the Marian transformer architecture and has been optimized for edge inference by splitting into encoder and decoder components with modified attention mechanisms. It exhibits robust performance for real-world translation tasks, making it highly reliable for practical applications. The model supports input sequences up to 256 tokens and can generate Spanish translations with high accuracy.

This is based on the implementation of OpusMT-En-Es found [here](https://github.com/Helsinki-NLP/Opus-MT). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/opus_mt_en_es).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[opus-mt-en-es]"
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.opus_mt_en_es.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.opus_mt_en_es.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of OpusMT-En-Es can be found
  [here](https://github.com/Helsinki-NLP/Opus-MT/blob/master/LICENSE).


## References
* [OPUS-MT – Building open translation services for the World](https://aclanthology.org/2020.eamt-1.61/)
* [Source Model Implementation](https://github.com/Helsinki-NLP/Opus-MT)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
