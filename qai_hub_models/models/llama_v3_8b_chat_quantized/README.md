[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Llama-v3-8B-Chat: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/llama_v3_8b_chat_quantized)

Llama 3 is a family of LLMs. The "Chat" at the end indicates that the model is optimized for chatbot-like dialogue. The model is quantized to w4a16(4-bit weights and 16-bit activations) and part of the model is quantized to w8a16(8-bit weights and 16-bit activations) making it suitable for on-device deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-KVCache-Quantized's latency.

This is based on the implementation of Llama-v3-8B-Chat found
[here](https://github.com/meta-llama/llama3/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v3_8b_chat_quantized).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying Llama 3 on-device

Large Language Model (LLM) such as [Llama 2](https://llama.meta.com/llama3/) has the following complexities to deploy on-device:
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

In order to export Llama 3, please ensure
1. Host machine has >40GB memory (RAM+swap-space)
2. If you don't have enough memory, export.py will dump instructions to increase swap space accordingly

## Sample output prompts generated on-device
1. --prompt "where is California?"
```
------- Response Summary --------
Prompt: where is California?
Response: California is a state located on the West Coast of
```

2. --prompt "what is 2+3?" --max-output-tokens 30
```
-------- Response Summary --------
Prompt: what is 2+3?
Response: 2 + 3 = 5
```

3. --prompt "what is superposition in Quantum Physics?" --max-output-tokens 30
```
Prompt: what is superposition in Quantum Physics?
Response: Superposition is a fundamental concept in quantum mechanics, which is a branch of physics that studies the behavior of matter and energy at a very
```



## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[llama_v3_8b_chat_quantized]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.llama_v3_8b_chat_quantized.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.llama_v3_8b_chat_quantized.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of Llama-v3-8B-Chat can be found
  [here](https://github.com/facebookresearch/llama/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://github.com/facebookresearch/llama/blob/main/LICENSE)

## References
* [LLaMA: Open and Efficient Foundation Language Models](https://ai.meta.com/blog/meta-llama-3/)
* [Source Model Implementation](https://github.com/meta-llama/llama3/tree/main)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
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


