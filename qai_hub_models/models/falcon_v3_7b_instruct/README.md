# [Falcon3-7B-Instruct: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/falcon_v3_7b_instruct)

Falcon3 family of Open Foundation Models is a set of pretrained and instruct LLMs ranging from 1B to 10B.

This is based on the implementation of Falcon3-7B-Instruct found [here](https://huggingface.co/tiiuae/Falcon3-7B-Instruct). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/falcon_v3_7b_instruct).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying Falcon3-7B-Instruct on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.



## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[falcon-v3-7b-instruct]"
```

For falcon_v3_7b_instruct, some additional functionality can be faster or is availiable
only with a GPU on the host machine.

- 🟢 Exporting the model for on-device deployment (GPU not required)
- 🟡 Running the demo (GPU recommended for speed, but not required)
- 🟡 Running evaluation (GPU recommended for speed, but not required)
- 🔴 Quantizing the model (GPU required)

If you are quantizing your own variant of falcon_v3_7b_instruct, a dedicated CUDA enabled
GPU (40 GB VRAM for 3B models to 80 GB VRAM for 8B models) is recommended. A GPU
can also increase the speed of evaluation and demo of your quantized model
significantly but it not strictly required.

Install the GPU package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[falcon-v3-7b-instruct]" onnxruntime-gpu==1.22 https://github.com/quic/aimet/releases/download/2.14.0/aimet_onnx-2.14.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl -f https://download.pytorch.org/whl/torch_stable.html
```



Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.falcon_v3_7b_instruct.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.falcon_v3_7b_instruct.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Falcon3-7B-Instruct can be found
  [here](https://falconllm.tii.ae/falcon-terms-and-conditions.html).
* The license for the compiled assets for on-device deployment can be found [here](https://falconllm.tii.ae/falcon-terms-and-conditions.html)


## References
* [Source Model Implementation](https://huggingface.co/tiiuae/Falcon3-7B-Instruct)



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
