[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Llama-v2-7B-Chat: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/llama_v2_7b_chat_quantized)

Llama 2 is a family of LLMs. The "Chat" at the end indicates that the model is optimized for chatbot-like dialogue. The model is quantized to 4-bit weights and 16-bit activations making it suitable for on-device deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-KVCache-Quantized's latency.

This is based on the implementation of Llama-v2-7B-Chat found
[here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v2_7b_chat_quantized).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.

## Deploying Llama 2 on-device

Large Language Model (LLM) such as [Llama 2](https://llama.meta.com/llama2/) has the following complexities to deploy on-device:
1. Model size is too large to fit in device memory for inference
2. Multi-Head Attention (MHA) has large activations leading to fallback from accelerators
3. High model load and inference time

We can tackle the above constraints with the following steps:
1. Quantize weights to reduce on-disk model size, e.g., int8 or int4 weights
2. Quantize activations to reduce inference time memory pressure
3. Graph transformations to reduce inference time memory pressure, e.g., Multi-Head to Split-Head Attention (MHA -> SHA)
4. Graph transformations to convert or decompose operations into more accelerator friendly operations e.g. Linear to Conv
5. For LLM with 7B or more parameters, above steps are still not good enough on mobile,
  hence we go one step further and split model into sub-parts.

Here, we divide the model into 4 parts in order to
1. Make model exportable with low memory usage
2. Avoid inference time out-of-memory errors

In order to export Llama 2, please ensure
1. Host machine has >40GB memory (RAM+swap-space)
2. If you don't have enough memory, export.py will dump instructions to increase swap space accordingly



## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[llama_v2_7b_chat_quantized]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.llama_v2_7b_chat_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.llama_v2_7b_chat_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Llama-v2-7B-Chat can be found
  [here](https://github.com/facebookresearch/llama/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://github.com/facebookresearch/llama/blob/main/LICENSE)

## References
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Source Model Implementation](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation


