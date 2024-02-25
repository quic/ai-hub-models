[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Whisper-Base: Automatic speech recognition (ASR) model for multilingual transcription as well as translation](https://aihub.qualcomm.com/models/whisper_asr)

State-of-art model encoder-decoder transformer. The encoder takes an audio chunk (around 30 second) converted to a log-Mel spectrogram.  The decoder predicts the corresponding text caption intermixed with special tokens that can be used to direct the single model to perform various speech tasks.

This is based on the implementation of Whisper-Base found
[here](https://github.com/openai/whisper/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/whisper_asr).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[whisper_asr]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.whisper_asr.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.whisper_asr.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Whisper-Base can be found
  [here](https://github.com/openai/whisper/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf).

## References
* [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
* [Source Model Implementation](https://github.com/openai/whisper/tree/main)

## Community
* Join [our AI Hub Slack community](https://join.slack.com/t/qualcomm-ai-hub/shared_invite/zt-2dgf95loi-CXHTDRR1rvPgQWPO~ZZZJg) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


