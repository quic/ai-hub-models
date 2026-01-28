# [Zipformer: Transformer-based automatic speech recognition (ASR) model for English and Chinese language](https://aihub.qualcomm.com/models/zipformer)

Zipformer streaming ASR (Automatic Speech Recognition) model is a state-of-the-art system designed for transcribing spoken language into written text streamingly. This model is based on the transformer architecture and has been optimized for edge inference by replacing linear layers with convolutional (conv) layers. It exhibits robust performance in realistic, noisy environments, making it highly reliable for real-world applications. Specifically, it excels in long-form transcription, capable of accurately transcribing audios. Time to the first token is the encoder's latency, while time to each additional token is joiner's latency, where we assume a max decoded length specified below.

This is based on the implementation of Zipformer found [here](https://github.com/k2-fsa/icefall). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/zipformer).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install torch==2.9.0+cpu -f https://download.pytorch.org/whl/torch/
pip install "qai-hub-models[zipformer]" k2==1.24.4.dev20251029+cpu.torch2.9.0 -f https://k2-fsa.github.io/k2/cpu.html
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.zipformer.demo { --quantize w16a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.zipformer.export { --quantize w16a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Zipformer can be found
  [here](https://github.com/huggingface/transformers/blob/v4.42.3/LICENSE).


## References
* [Zipformer A faster and better encoder for automatic speech recognition](https://openreview.net/forum?id=9WD9KwssyT)
* [Source Model Implementation](https://github.com/k2-fsa/icefall)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
