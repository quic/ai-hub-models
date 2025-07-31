# [Whisper-Large-V3-Turbo: Transformer-based automatic speech recognition (ASR) model for multilingual transcription and translation available on HuggingFace](https://aihub.qualcomm.com/models/whisper_large_v3_turbo)

Whisper large-v3-turbo is a finetuned version of a pruned Whisper large-v3. In other words, it's the exact same model, except that the number of decoding layers have reduced from 32 to 4. As a result, the model is way faster, at the expense of a minor quality degradation. This model is based on the transformer architecture and has been optimized for edge inference by replacing Multi-Head Attention (MHA) with Single-Head Attention (SHA) and linear layers with convolutional (conv) layers. It exhibits robust performance in realistic, noisy environments, making it highly reliable for real-world applications. Specifically, it excels in long-form transcription, capable of accurately transcribing audio clips up to 30 seconds long. Time to the first token is the encoder's latency, while time to each additional token is decoder's latency, where we assume a max decoded length specified below.

This is based on the implementation of Whisper-Large-V3-Turbo found [here](https://github.com/huggingface/transformers/tree/v4.42.3/src/transformers/models/whisper). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/whisper_large_v3_turbo).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[whisper-large-v3-turbo]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.whisper_large_v3_turbo.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.whisper_large_v3_turbo.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Whisper-Large-V3-Turbo can be found
  [here](https://github.com/huggingface/transformers/blob/v4.42.3/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
* [Source Model Implementation](https://github.com/huggingface/transformers/tree/v4.42.3/src/transformers/models/whisper)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
