# [Whisper-Small: Transformer-based automatic speech recognition (ASR) model for multilingual transcription and translation available on HuggingFace](https://aihub.qualcomm.com/models/whisper_small)

HuggingFace Whisper-Small ASR (Automatic Speech Recognition) model is a state-of-the-art system designed for transcribing spoken language into written text. This model is based on the transformer architecture and has been optimized for edge inference by replacing Multi-Head Attention (MHA) with Single-Head Attention (SHA) and linear layers with convolutional (conv) layers. It exhibits robust performance in realistic, noisy environments, making it highly reliable for real-world applications. Specifically, it excels in long-form transcription, capable of accurately transcribing audio clips up to 30 seconds long. Time to the first token is the encoder's latency, while time to each additional token is decoder's latency, where we assume a max decoded length specified below.

This is based on the implementation of Whisper-Small found [here](https://github.com/huggingface/transformers/tree/v4.42.3/src/transformers/models/whisper).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/whisper_small).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[whisper-small]"
```

### 2. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.whisper_small.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.whisper_small.export
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Whisper-Small can be found
  [here](https://github.com/huggingface/transformers/blob/v4.42.3/LICENSE).

## References
* [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
* [Source Model Implementation](https://github.com/huggingface/transformers/tree/v4.42.3/src/transformers/models/whisper)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
