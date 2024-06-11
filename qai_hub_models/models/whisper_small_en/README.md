[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Whisper-Small-En: Automatic speech recognition (ASR) model for English transcription as well as translation](https://aihub.qualcomm.com/models/whisper_small_en)

OpenAI’s Whisper ASR (Automatic Speech Recognition) model is a state-of-the-art system designed for transcribing spoken language into written text. It exhibits robust performance in realistic, noisy environments, making it highly reliable for real-world applications. Specifically, it excels in long-form transcription, capable of accurately transcribing audio clips up to 30 seconds long. Time to the first token is the encoder's latency, while time to each additional token is decoder's latency, where we assume a mean decoded length specified below.

This is based on the implementation of Whisper-Small-En found
[here](https://github.com/openai/whisper/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/whisper_small_en).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[whisper_small_en]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.whisper_small_en.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.whisper_small_en.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Whisper-Small-En can be found
  [here](https://github.com/openai/whisper/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)

## References
* [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
* [Source Model Implementation](https://github.com/openai/whisper/tree/main)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


