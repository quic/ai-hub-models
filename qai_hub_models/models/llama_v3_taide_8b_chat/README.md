# [Llama3-TAIDE-LX-8B-Chat-Alpha1: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/llama_v3_taide_8b_chat)

The Llama3-TAIDE-LX-8B-Chat-Alpha1 LLM model is based on Meta's released LLaMA3-8b model, fine-tuned on Traditional Chinese data, and enhanced for office tasks and multi-turn dialogue capabilities through instruction tuning. The TAIDE model is incorporating text and training materials from various fields in Taiwan to enhance the model's ability to respond in Traditional Chinese and perform specific tasks such as automatic summarization, writing emails, articles, and translating between Chinese and English.

This is based on the implementation of Llama3-TAIDE-LX-8B-Chat-Alpha1 found [here](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v3_taide_8b_chat).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying Llama-3-TAIDE on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.



## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[llama-v3-taide-8b-chat]"
```

Note: GPU is unnecessary if you wish to only export the model for on-device deployment.

For llama_v3_taide_8b_chat, a dedicated CUDA enabled GPU (40 GB VRAM for 3B models to 80 GB VRAM for 8B models) is needed to quantize the model on your local machine. GPU can also increase the speed of evaluation and demo of your quantized model significantly.
Install the GPU package via pip:
```bash
pip install "qai-hub-models[llama-v3-taide-8b-chat]" onnxruntime-gpu==1.22 https://github.com/quic/aimet/releases/download/2.10.0/aimet_onnx-2.10.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl -f https://download.pytorch.org/whl/torch_stable.html
```



Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.llama_v3_taide_8b_chat.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.llama_v3_taide_8b_chat.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Llama3-TAIDE-LX-8B-Chat-Alpha1 can be found
  [here](https://en.taide.tw/download.html).
* The license for the compiled assets for on-device deployment can be found [here](https://en.taide.tw/download.html)


## References
* [LLaMA: Open and Efficient Foundation Language Models](https://ai.meta.com/blog/meta-llama-3/)
* [Source Model Implementation](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)



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
