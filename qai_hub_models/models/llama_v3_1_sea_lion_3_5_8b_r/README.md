# [Llama-SEA-LION-v3.5-8B-R: Llama-SEA-LION-v3.5-8B-R is a hybrid model offering versatile functionality, handling both complex reasoning tasks and general text generation, with mode selection managed through the tokenizer's chat template](https://aihub.qualcomm.com/models/llama_v3_1_sea_lion_3_5_8b_r)

SEA-LION is a collection of Large Language Models (LLMs) which have been pretrained and instruct-tuned for the Southeast Asia (SEA) region. The model is quantized to w4a16 (4-bit weights and 16-bit activations) and part of the model is quantized to w8a16 (8-bit weights and 16-bit activations) making it suitable for on-device deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-Quantized's latency.

This is based on the implementation of Llama-SEA-LION-v3.5-8B-R found [here](https://github.com/aisingapore/sealion/blob/main/models/sea-lion-v3.5/llama-sea-lion-v3.5-8B.md). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v3_1_sea_lion_3_5_8b_r).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying Llama-SEA-LION-v3.5-8B-R on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.



## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[llama-v3-1-sea-lion-3-5-8b-r]"
```

Note: GPU is unnecessary if you wish to only export the model for on-device deployment.

For llama_v3_1_sea_lion_3_5_8b_r, a dedicated CUDA enabled GPU (40 GB VRAM for 3B models to 80 GB VRAM for 8B models) is needed to quantize the model on your local machine. GPU can also increase the speed of evaluation and demo of your quantized model significantly.
Install the GPU package via pip:
```bash
pip install "qai-hub-models[llama-v3-1-sea-lion-3-5-8b-r]" onnxruntime-gpu==1.22 https://github.com/quic/aimet/releases/download/2.10.0/aimet_onnx-2.10.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl -f https://download.pytorch.org/whl/torch_stable.html
```



Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Llama-SEA-LION-v3.5-8B-R can be found
  [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE)


## References
* [Source Model Implementation](https://github.com/aisingapore/sealion/blob/main/models/sea-lion-v3.5/llama-sea-lion-v3.5-8B.md)



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
