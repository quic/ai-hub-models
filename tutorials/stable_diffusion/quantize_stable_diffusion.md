# Quantize Stable Diffusion for Edge Deployment

Stable diffusion models are often fine-tuned to generate images in different
styles and characters. Here we show how easy it is to bring your own fine-tuned
weights to Qualcomm devices using post-training quantization.

## Requirements

The quantization process requires a computer with:

* Linux
* 32GB RAM

Note that CUDA-enabled GPU can help but is not required.

## Quick Start

Let's say we want to deploy
[`yandex/stable-diffusion-2-1-alchemist`](https://huggingface.co/yandex/stable-diffusion-2-1-alchemist)
from Huggingface. We need to find a model from AI Hub Model that has the same architecture as this model. In this case, we can use `stable_diffusion_v2_1`.

We first download the weights and run floating point evaluation in PyTorch:

```sh
# See the options
python -m qai_hub_models.models.stable_diffusion_v2_1.demo -h

# Run inference
export PROMPT="realistic futuristic city-downtown with short buildings, sunset"
python -m qai_hub_models.models.stable_diffusion_v2_1.demo --eval-mode fp --checkpoint yandex/stable-diffusion-2-1-alchemist --num-steps 20 --prompt "$PROMPT"
```

This will output an image to `export/torch_fp32/image.png` that might look
like

![Image Generated from Torch Floating Point (Original Model)](assets/fp.png)

If it looks good, we can start to quantize the each component of model.

```sh
# text_encoder
python -m qai_hub_models.models.stable_diffusion_v2_1.quantize --component text_encoder --checkpoint yandex/stable-diffusion-2-1-alchemist -o build/alchemist

# unet
python -m qai_hub_models.models.stable_diffusion_v2_1.quantize --component unet --checkpoint yandex/stable-diffusion-2-1-alchemist -o build/alchemist

# vae
python -m qai_hub_models.models.stable_diffusion_v2_1.quantize --component vae --checkpoint yandex/stable-diffusion-2-1-alchemist -o build/alchemist
```

By default we run 20 diffusion steps on 100 prompts. But you can change
that with `--num-steps` and `--num-samples` (see `python -m qai_hub_models.models.stable_diffusion_v2_1.quantize -h`). This will
take a few hours on CPU. Also, don't run unet and vae concurrently, because
they use the same set of calibration data (latent noise). Running them
sequentially allows the second job to reuse the cached calibration data.

The quantized models will be exported to `build/alchemist`. We can generate the image using quantized model via simulated quantization with

```sh
python -m qai_hub_models.models.stable_diffusion_v2_1.demo --eval-mode quantsim --checkpoint build/alchemist --num-steps 20 --prompt "$PROMPT"
```

The image might look like

![Image Generated from Simulated Quantization](assets/quantsim.png)

If that looks good, we can proceed to compile the quantized model for on-device profiling and
inference. Note that you'd need to first register an account on [Qualcomm AI Hub](https://app.aihub.qualcomm.com/) and set up the token if you haven't already.

```sh
python -m qai_hub_models.models.stable_diffusion_v2_1.export --checkpoint build/alchemist --chipset qualcomm-snapdragon-x-elite
```

This will submit 3 compile jobs and 3 profiling jobs and download the compiled
artifacts.

Using the compiled artifacts, you can deploy it to X Elite device with the [demo
app](https://github.com/quic/ai-hub-apps/tree/main/apps/windows/python/StableDiffusion). (We don't have a sample app for Android at the moment unfortunately.)
