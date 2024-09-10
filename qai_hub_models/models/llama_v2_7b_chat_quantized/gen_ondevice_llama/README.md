# Generating Genie-compatible QNN binaries from AI Hub

In this tutorial we will show how an end to end workflow of deploying HuggingFace Llama2 7b model to run on Snapdragon® platform such as Snapdragon® 8 Gen 2 (e.g., Samsung Galaxy S23 family) and 8 Gen 3 chipset (e.g., Samsung Galaxy S24 family) and Snapdragon® X (e.g. Snapdragon® based Microsoft Surface Pro). Windows platform must have at least 16GB memory.

## Overview

On x86 Linux host PC
1. Get access to [llama2 weights from huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf])
2. Use Qualcomm AI Hub Model to split model into 8 parts (4 Prompt Processor, 4 Token Generator)
3. Use Qualcomm AI Hub Compile ONNX model with quantization encoding to shared model library
4. Install [QNN SDK](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)
5. Convert Prompt processor and Token generator library into weight shared QNN context binary to be run on Genie, an on-device app for running LLMs on Snapdragon® platform
6. Push assets required by Genie from x86 Linux host to device

On Android / Windows PC with Snapdragon® platform
7. Run Genie on device with an example prompt

Note that because this is a large model, it may take over 2-3 hours to generate required assets.
This is an early preview. We're working on simplifying the workflow.

If you have more questions, please feel free to post on [AI Hub slack channel](https://aihub.qualcomm.com/community/slack)

## Requirements

1. x86 Linux host (We tested on Ubuntu 22.04)
2. [QNN SDK](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)
3. [qai-hub-models](https://pypi.org/project/qai-hub-models/)
4. [qai-hub](https://pypi.org/project/qai-hub/)

## 1. Generate Genie compatible QNN binaries from AI Hub

### Set up virtual envs

On x86 Linux host (e.g., Ubuntu 22.04), we will create two virtual envs. One
for qai-hub-models, and the other for QNN SDK. This is to avoid conflict in
dependency requirements between the two.  We recommend a
[virtualenv](https://virtualenv.pypa.io/en/latest/) with
python3.10, but
[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) works as well.

For clarity, we recommend using two shell sessions, one for each venv.

```
# In shell session 1 (Hub Model)
python3.10 -m venv hub_model
```
```
# In shell session 2 (QNN)
python3.10 -m venv qnn
```

### Export Llama model via AI Hub Models

In shell session 1, install `qai-hub-models` under `hub_model` virtual env

```bash
source hub_model/bin/activate
pip install "qai_hub_models[llama-v2-7b-chat-quantized]>=0.12.0"
```

Ensure at least 40GB of memory (RAM + swap). On Ubuntu you can check it by

```
free -h
```

Increase swap size if needed.

We use
[qai-hub-models](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/)
to adapt Huggingface Llama models for on-device inference.

```bash
# In shell session 1
python -m qai_hub_models.models.llama_v2_7b_chat_quantized.export --skip-downloading --skip-profiling --skip-inferencing
```

This can take a few hours to complete. Once finished, you should see something
like

**Sample output**
```bash
...
output of export.py
...
Run compiled models on a hosted device on sample data using:
python qai_hub_models/models/llama_v2_7b_chat_quantized/demo.py --on-device --hub-model-id {comma-separated-model-ids} --device {device}
```

Please note down `{comma-separated-model-ids}` portion for next steps.

Also visit the compilation job on Hub to find the QNN version used on AI Hub.
We'll need to use the same QNN version in the next steps.

### Generate Genie-compatible QNN binaries

Unless specified otherwise, this section will happen in shell session 2.

Install [QNN
SDK](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)
QNN version matching AI Hub's (which is 2.25 as of 2024-9-5) for x86 Linux host. Note that the first time after log in you would be redirected
to qpm home page. Click on the link again to get to QNN download page.

If successful, you'd see message like

```
SUCCESS: Installed qualcomm_ai_engine_direct.Core at /opt/qcom/aistack/qairt/2.25.0.240728
```

Record `/opt/qcom/aistack/qairt/2.26.1.240828` path and set it as
`$QNN_SDK_ROOT`

```
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.25.0.240728
```

Now, install python dependency for QNN converter in `qnn` virtual env

```bash
source qnn/bin/activate
pip install -r requirements.txt
```

Setup required environment variables via

```
source ${QNN_SDK_ROOT}/bin/envsetup.sh
```

(Optional) See [QNN
doc](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/setup.html?product=1601111740010412) for more details



Use `gen_ondevice_llama.py` to generate genie-compatible QNN binaries. Make
sure you have 130GB or more of disk space available before running.

Please run following to get information about all the options:

```bash
python gen_ondevice_llama.py -h
```

For example, to generate Genie-compatible binary and bundle:

(a) For Snapdragon® 8 Gen 3 Android device:

```bash
python gen_ondevice_llama.py --hub-model-id <model_ids_from_last_step> --output-dir ./export --tokenizer-zip-path ./tokenizer.zip --target-gen snapdragon-gen3 --target-os android
```

(b) For Windows with Snapdragon® X Elite

```bash
python gen_ondevice_llama.py --hub-model-id <model_ids_from_last_step> --output-dir ./export --tokenizer-zip-path ./tokenizer.zip --target-gen snapdragon-gen2 --target-os windows
```

The commands above may take over 15 mins to finish.

## 2. Running generated QNN binaries on-device

### Copy generated assets to target device

1. Copy content from `export/shared_bin_{target_gen}` to device which includes
    - 4 binaries with each binary having two graph sharing weights
    - htp configuration
    - genie configuration
    - all required libraries copied from QNN SDK

### Run Genie on Windows devices with Snapdragon® X

In Powershell, navigate to the directory containing the above contents and run

```
.\genie-t2t-run -c htp-model-config-llama2-7b.json -p "<<SYS>>\nYou are a helpful AI assistant.<</SYS>>\n\n[INST] have we been to Mars? [/INST]"
```

See below for sample outputs.

### Run Genie on Android devices with Snapdragon® 8 Gen 2 and Gen 3

Set `LD_LIBRARY_PATH` to directory content from 1. copied into

```bash
export LD_LIBRARY_PATH=<path to where all the libraries are copied into>
```

Input prompt for genie requires tags similar to source llama2 model and must follow similar pattern for better results

```text
<<SYS>>\nYou are a helpful AI assistant.<</SYS>>\n\n[INST] have we been to the Mars? [/INST]
```

```bash
# Optional: connect to device (e.g. adb shell)
cd {parent dir of genie binary}
./genie-t2t-run -c htp-model-config-llama2-7b.json -p "<<SYS>>\nYou are a helpful AI assistant.<</SYS>>\n\n[INST] have we been to Mars? [/INST]"
```

**Sample output**

```text
130|e2q:/data/local/tmp/llama2 $ ./genie-t2t-run -c htp-model-config-llama2-7b.json -p "<<SYS>>\nYou are a helpful AI assistant.<</SYS>>\n\n[INST] have we been to Mars? [/INST]"
Using libGenie.so version 1.0.0

[WARN]  "Unable to initialize logging in backend extensions."
[INFO]  "Allocated total size = 431294976 across 8 buffers"
[PROMPT]: <<SYS>>\nYou are a helpful AI assistant.<</SYS>>\n\n[INST] have we been to Mars? [/INST]

[BEGIN]:   As a helpful AI assistant, I can tell you that humans have not yet visited Mars.Ћ However, there have been several robotic missions to Mars, including NASA's Curiosity rover, which has been exploring the planet since 2012. These missions have helped scientists learn more about the Martian environment and geology, and have provided valuable insights into the potential habitability of the planet.
```
