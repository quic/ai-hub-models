# Quantize Llama 3 for Edge Deployment

Most large language models (LLMs) are initially trained with high precision
(e.g., 32-bit floating point). However, to run them efficiently on more
constrained devices such as edge, mobile, and automotive platforms, the weights
and activations must be quantized. Quantizing the weights of LLMs to 4 bits is
a common practice also for cloud deployment. This reduces the model's size,
allowing it to fit in memory and lowering memory bandwidth, improving latency.
For edge deployment, we often take it a step further by also quantizing the
activations. This makes the model more suitable for edge hardware and leads to
better performance. To ensure good accuracy, we primarily use 16-bit
activations, leveraging 8-bit activations only where feasible. In this tutorial
we are going to use Llama 3.2 3B. Please change the model name appropriately to
quantize other variants. Right now only Llama 3 variants expose the necessary
scripts.

## Requirements

The quantization process requires a computer with:

* Linux
* CUDA GPU

For Llama 3.2 3B, you will need a GPU with 40 GB of VRAM (32 GB is borderline,
so you may need to re-run if it fails with memory errors). For Llama 3.0 8B and
3.1 8B, you will need 80 GB of VRAM.

## Quick Start

This section is meant to quickly walk you through an example of how to quantize a Llama 3 LLM. This tutorial works for Llama 3.0 8B, 3.1 8B, or 3.2 3B, and all fine-tuned variants that use the same tokenizer and network structure. Let's use the Hugging Face repo `meta-llama/Llama-3.2-3B-Instruct` as an example.

### Quantize

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.quantize \
    --checkpoint meta-llama/Llama-3.2-3B-Instruct \
    --output-dir ./quantized_model
```

### Evaluate & Demo

Using the output checkpoint will run evaluation on quantized model and passing Hugging Face repo name will run evaluation on the floating point model.

Evaluate command:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.evaluate \
    --checkpoint ./quantized_model \
    --task wikitext-ppl
```

Demo command:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.demo \
    --checkpoint ./quantized_model \
    --prompt "What is gravity?"
```

### Deploy on-device

Export to QNN context binaries that can be on device.

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export \
    --checkpoint ./quantized_model \
    --device "Snapdragon 8 Elite QRD" \
    --skip-inferencing \
    --skip-profiling \
    --output-dir genie_bundle_8_elite
```

Now we will go through each step in greater detail.

## Custom weights

Models available in AI Hub Models provide pre-computed
quantization parameters that were produced with this workflow. You can
re-compute them, but they will essentially be the same. However, if you bring
your own weights via a local checkpoint or Hugging Face repo, this is a good
reason to re-evaluate the model and potentially re-quantize it.

## Checkpoints

Several commands take `--checkpoint` to define the source model. This can be
one of several things:

* Floating Point Checkpoint
    * `"DEFAULT_UNQUANTIZED"`
        - Uses the default floating point checkpoint. This is useful for comparison
          or the source model if re-quantizing.
    * Local folder:
        - Sharded weights files: `*.safetensors`.
        - Index file that maps parameter names to the sharded weights files.
        - `config.json`: Model config that describes the details of the model.
        - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: Tokenizer used by the model.
    * Name of Hugging Face model (including organization, e.g., `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)

* Quantized Checkpoint
    * `"DEFAULT"`
        - Uses the default pre-quantized checkpoint. This can be deployed directly.
    * Local folder:
        - Quantization parameters (`model.encodings`).
        - ONNX models of various input sequence lengths (e.g., `model_ar128_cl4096.onnx`) with a shared weights file (`model.data`).
        - `config.json`: Model config that describes the details of the model.
        - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: Tokenizer used by the model.

The `quantize` command expects a floating point checkpoint and the `export` command
expects a quantized checkpoint. For other commands, such as `demo` and
`evaluate`, you can provide either depending on if you want to test the
original floating point model or a simulation of the quantized version.

## Data

To quantize an LLM, we need representative input samples for two reasons:

* **Calibration data**: Data used during the quantization calibration process.
* **Evaluation data**: Data used to ensure that calibration was successful.

Ideally, these two datasets should be disjoint and represent the same
underlying data distribution. The easiest way is to take a single dataset and
split it up into two. If a split exists, you can use that as well. By default, the quantize and evaluate scripts that we
will cover use English-language WikiText. If you are primarily targeting a
non-English language, you may want to change this dataset to something else.

## Quantize

The models that we allow via a floating point checkpoint must be very similar
to the original model:

* The architecture, model type, number of hidden layers, hidden size, number of attention layers, number of key value layers cannot be changed at this time. These configurations are important to how we split the model for deploying on-device. Changing these may require downstream changes that are currently not documented. Other modifications to the config file are permitted but may elicit untested behavior.

We will use `meta-llama/Llama-3.2-3B-Instruct` as a placeholder for a custom
checkpoint. This aligns with the default checkpoint and could thus be omitted,
but we include it explicitly to make the flow for custom checkpoints clear.

To kick off the quantization process, run:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.quantize \
    --checkpoint meta-llama/Llama-3.2-3B-Instruct \
    -o ./quantized_model
```
This script will run [AIMET](https://github.com/quic/aimet) to quantize the
weights and activations of the model. This could take up to an hour. The
`./quantized_model` refers to a folder name that will be created. This folder
will contain the quantized model in the form of an ONNX file (`.onnx`), its
accompanying external weights file (`.data`), and an AIMET encodings file
containing the calibration parameters (`.encodings`). The model config will be copied to the output checkpoint folder so that later steps can load the quantized model correctly.

For even better results, we can add an algorithm called [Sequential
MSE](https://quic.github.io/aimet-pages/releases/latest/featureguide/seq_mse.html).
This can greatly improve the weight quantization, although is expensive to
compute. The quantization process can now take 5+ hours, depending on the size
of the model and the speed of your CUDA GPU.

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.quantize \
    --checkpoint meta-llama/Llama-3.2-3B-Instruct \
    --use-seq-mse \
    -o ./quantized_model
```

## Evaluate

Once the model is quantized, we need to evaluate the quantized model's accuracy
to ensure that it is within acceptable accuracy. The same evaluation can also be run on the unquantized model to get a good baseline. We outline how to do this
quantitatively using [perplexity](https://en.wikipedia.org/wiki/Perplexity) (PPL) and [tinyMMLU](https://arxiv.org/abs/2402.14992), as well as
qualitatively on select prompts.

### Quantitatively with PPL

To evaluate on the quantized model, you will have to provide the Hugging Face model name or the default model config will be used.

Evaluate PPL score on [WikiText (English)](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/datasets/wikitext.py) using the unquantized model:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.evaluate \
    --checkpoint meta-llama/Llama-3.2-3B-Instruct \
    --task wikitext-ppl
```

Evaluate using the quantized model:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.evaluate \
    --checkpoint ./quantized_model \
    --task wikitext-ppl
```

For example, Llama 3.2 with the original weights, the unquantized model's PPL is 10.15 and the PPL after quantization (with Sequential MSE) is 11.94. From this, we can see that the quantized model got a little bit worse in terms
of PPL. Please note that PPL may not accurately reflect the model's ability to
perform its intended tasks.  While it provides some insight into the success of
the calibration process, further evaluation will be needed on a more
task-oriented benchmark.

If the PPL is OK, please proceed to the next step.
If the PPL looks too high, please refer to the [AIMET debugging
guidelines](https://quic.github.io/aimet-pages/releases/latest/userguide/debugging_guidelines.html).

### Quantitatively with MMLU

To evaluate on the quantized model, you will have to provide the Hugging Face model name or default model config will be used.

Evaluate [tinyMMLU](https://Hugging Face.co/datasets/tinyBenchmarks/tinyMMLU) using the unquantized model:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.evaluate \
    --checkpoint meta-llama/Llama-3.2-3B-Instruct \
    --task tiny-mmlu
```

Evaluate using the quantized model:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.evaluate \
    --checkpoint ./quantized_model \
    --task tiny-mmlu
```
For example, Llama 3.2 with the original weights, the unquantized model's
tinyMMLU is 64% and the tinyMMLU after quantization (with Sequential MSE) is
53%. This showed a pretty wide gap. However, tinyMMLU contains only 100
samples and experience tells us this is results in a noisy metric.

The script also allows running the entire
[MMLU](https://huggingface.co/datasets/cais/mmlu) (`--task mmlu`), however it
takes a long time. We can trade off signal strength with evaluation time by
setting the number of samples. We recommend `--task mmlu --num-samples 1000`.
This gave 61% for the unquantized and 57% for the quantized. This shows how
noisy tinyMMLU is.

MMLU is a widely used metric and may or may not reflect the model's ability to
perform the intended downstream task. Further evaluation should be done using
other task-oriented benchmarks if necessary.

### Qualitatively on prompts

It is also a good idea to try out the model on a few prompts. This can help
ensure early on that the model retains its ability to the perform the task of
interest; PPL may not capture this adequately. Note that the script itself will
add the system prompt appropriate for the model.

```sh
python -m qai_hub_models.models.llama_v3_2_3b_instruct.demo \
    --checkpoint ./quantized_model \
    --prompt "What is gravity? Answer concisely."
```

The output could look like this:
```
-------- Response Summary --------
Prompt: What is gravity? Answer concisely.
Response: Gravity is a fundamental force of nature that attracts two objects with mass towards each other, holding them together and keeping them on their respective paths.
```

The prompt style by default is the same as the original Llama model. It can be modified using `--raw --prompt "<full prompt>"`.

## Exporting

Once we are happy with the off-target evaluation of the model, it is time to
export it to a deployable format and integrate it into an application. For this
process, please follow the [LLM on-device
deployment](https://github.com/quic/ai-hub-apps/blob/main/tutorials/llm_on_genie/README.md)
tutorial. The only difference is that when you call the export command, please
make sure to add the `--checkpoint` option so that it picks up our custom
quantization parameters:

```sh
python -m qai_hub_models.models.llama_v3_2_3b_chat_quantized.export \
    --checkpoint ./quantized_model \
    --device "Snapdragon 8 Elite QRD" \
    --skip-inferencing \
    --skip-profiling \
    --output-dir genie_bundle_8_elite
```

## Run LLMs on-device

This tutorial walks you through how to run [LLMs end to end on-device](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie)

## Contact us

Please reach out to us if you need encounter any issues with this tutorial:

* [Slack Community](https://aihub.qualcomm.com/community/slack)
* [Email Support](ai-hub-support@qti.qualcomm.com<mailto:ai-hub-support@qti.qualcomm.com)
