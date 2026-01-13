# [Llama-v3-ELYZA-JP-8B: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/llama_v3_elyza_jp_8b)

Llama-3-ELYZA-JP-8B is a lightweight LLM with only 8 billion parameters, yet achieves Japanese language performance comparable to GPT-3.5 Turbo. The model has undergone additional pre-training and post-training of Japanese to expand instruction-following capabilities in Japanese.

This is based on the implementation of Llama-v3-ELYZA-JP-8B found [here](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
across various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v3_elyza_jp_8b).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Deploying Llama 3 Elyza JP 8B on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.



## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[llama-v3-elyza-jp-8b]"
```

For llama_v3_elyza_jp_8b, some additional functionality can be faster or is availiable
only with a GPU on the host machine.

- 🟢 Exporting the model for on-device deployment (GPU not required)
- 🟡 Running the demo (GPU recommended for speed, but not required)
- 🟡 Running evaluation (GPU recommended for speed, but not required)
- 🔴 Quantizing the model (GPU required)

If you are quantizing your own variant of llama_v3_elyza_jp_8b, a dedicated CUDA enabled
GPU (40 GB VRAM for 3B models to 80 GB VRAM for 8B models) is recommended. A GPU
can also increase the speed of evaluation and demo of your quantized model
significantly but it not strictly required.

Install the GPU package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[llama-v3-elyza-jp-8b]" onnxruntime-gpu==1.22 https://github.com/quic/aimet/releases/download/2.20.0/aimet_onnx-2.20.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl -f https://download.pytorch.org/whl/torch_stable.html
```



Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.llama_v3_elyza_jp_8b.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.llama_v3_elyza_jp_8b.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Llama-v3-ELYZA-JP-8B can be found
  [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE).


## References
* [GPT-4 を上回る日本語性能のLLM Llama-3-ELYZA-JP を開発しました](https://note.com/elyza/n/n360b6084fdbd)
* [Source Model Implementation](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B)



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
