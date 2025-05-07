# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import onnx
import torch
from qai_hub.client import DatasetEntries
from tqdm import tqdm
from transformers import CLIPTokenizer

from qai_hub_models.models._shared.stable_diffusion.app import (
    run_diffusion_steps_on_latents,
)


def export_onnx_in_memory(
    torch_model: torch.nn.Module,
    example_input: tuple[torch.Tensor, ...],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    **kwargs,
) -> onnx.ModelProto:
    """
    Exports a PyTorch model to ONNX format and loads it into memory without saving to disk.

    Args:
        torch_model: The PyTorch model to export.
        example_input: A tensor representing the input shape to the model.

    Returns:
        onnx.ModelProto: The ONNX model loaded in memory.
    """
    buffer = io.BytesIO()
    torch.onnx.export(
        torch_model,
        example_input,
        buffer,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
    )
    buffer.seek(0)
    onnx_model = onnx.load_model(buffer)
    return onnx_model


def clip_extreme_values(
    model: onnx.ModelProto,
    extreme_value_threshold: float = -1e15,
    clip_value: float = -1e4,
) -> onnx.ModelProto:
    """
    Clips extreme values in the initializers of an ONNX model.

    This is needed until AIMET-4029 is resolved.

    Parameters:
    model: The ONNX model to process.
    extreme_value_threshold: The threshold below which values will be clipped. Default is -1e15.
    clip_value: The value to which extreme values will be clipped. Default is -1e4.

    Returns:
    onnx.ModelProto: The modified ONNX model with extreme values clipped.
    """
    extreme_value_threshold_np = np.float32(extreme_value_threshold)
    clip_value_np = np.float32(clip_value)

    updated_layers = []
    for initializer in model.graph.initializer:
        tensor_values = onnx.numpy_helper.to_array(initializer)

        if tensor_values.dtype != np.float32:
            tensor_values = tensor_values.astype(np.float32)

        mask = tensor_values < extreme_value_threshold_np
        if np.any(mask):
            tensor_values = tensor_values.copy()

            tensor_values[mask] = clip_value_np

            new_initializer = onnx.numpy_helper.from_array(
                tensor_values, name=initializer.name
            )

            model.graph.initializer.remove(initializer)
            model.graph.initializer.append(new_initializer)

            updated_layers.append(initializer.name)

    if updated_layers:
        print(
            f"Extreme values clipped for {len(updated_layers)} layers: {updated_layers}"
        )
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            onnx.save(model, temp_file.name)
            model = onnx.load(temp_file.name)
    else:
        print("No extreme values were found; no changes made.")

    return model


def make_calib_data_unet_vae(
    export_dir,
    prompt_path,
    tokenizer,
    text_encoder_hf,
    unet_hf,
    scheduler,
    num_steps: int = 20,
    num_samples: int = 100,
    guidance_scale: float = 7.5,
) -> tuple[str, str]:
    """Generate calibration data for Unet and Vae and save as npz.

    This will generate

    Number of samples for unet: num_samples * num_steps * 2 (*2 for cond and
    uncond text embeddings). e.g., 100 samples * 20 steps  * 2 = 4k

    Number of samples for vae: num_samples.

    num_samples: up to 500 (the number of prompts in our dataset)

    Returns:
    - export_path_unet
    - export_path_vae
    """
    export_path_unet = os.path.join(
        export_dir, f"unet_calib_n{num_samples}_t{num_steps}.npz"
    )
    export_path_vae = os.path.join(
        export_dir, f"vae_calib_n{num_samples}_t{num_steps}.npz"
    )
    cond_tokens, uncond_token = load_calib_tokens(
        prompt_path, tokenizer, num_samples=num_samples
    )

    # uncond_emb doesn't change for each prompt
    uncond_emb = None
    if guidance_scale > 0:
        uncond_emb = text_encoder_hf(uncond_token)

    calib_unet = dict(
        latent=[],
        time_emb=[],
        cond_emb=[],
    )  # type: ignore
    if guidance_scale > 0:
        calib_unet["uncond_emb"] = []
    calib_vae = dict(latent=[])  # type: ignore

    for i, cond_token in tqdm(
        enumerate(cond_tokens),
        total=len(cond_tokens),
        desc=f"Running {num_steps} diffusion steps on {len(cond_tokens)} samples",
    ):
        cond_emb = text_encoder_hf(cond_token)
        latent, all_steps = run_diffusion_steps_on_latents(
            unet_hf,
            scheduler=scheduler,
            cond_embeddings=cond_emb,
            uncond_embeddings=uncond_emb,
            num_steps=num_steps,
            return_all_steps=True,
            guidance_scale=guidance_scale,
        )
        calib_unet["latent"].extend(all_steps["latent"])
        calib_unet["time_emb"].extend(all_steps["time_emb"])
        calib_unet["cond_emb"].extend([cond_emb] * num_steps)
        if guidance_scale > 0:
            calib_unet["uncond_emb"].extend([uncond_emb] * num_steps)
        calib_vae["latent"].append(latent)

    for k in list(calib_unet.keys()):
        calib_unet[k] = np.concatenate([v.detach().numpy() for v in calib_unet[k]])
    for k in list(calib_vae.keys()):
        calib_vae[k] = np.concatenate([v.detach().numpy() for v in calib_vae[k]])

    np.savez(export_path_unet, **calib_unet)
    np.savez(export_path_vae, **calib_vae)

    print(f"Data saved to {export_path_unet}")
    print(f"Data saved to {export_path_vae}")
    return export_path_unet, export_path_vae


def load_calib_tokens(
    path: str,
    tokenizer: CLIPTokenizer,
    num_samples: int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Returns

    - tokens: List of length `num_samples` (500 if None) of
    torch.Tensor(int32) representing conditional tokens.

    - uncond_tokens: torch.Tensor(int32) of shape (1, 77) to represent
    unconditional tokens (padding).
    """
    with open(path) as f:
        prompts = f.readlines()
    if num_samples is not None:
        prompts = prompts[:num_samples]
    print(f"Loading {len(prompts)} prompts")
    cond_tokens = [run_tokenizer(tokenizer, prompt)[0] for prompt in prompts]
    uncond_token = run_tokenizer(tokenizer, prompts[0])[1]
    return cond_tokens, uncond_token


def load_unet_calib_dataset_entries(
    path: str, num_samples: int | None = None
) -> DatasetEntries:
    npz = np.load(path)
    num_diffusion_samples = npz["latent"].shape[0]
    latent = np.split(npz["latent"], num_diffusion_samples, axis=0)
    # time embeddings range from 999 to 0
    time_emb = [
        np.asarray([[999 - (999 * (i - 1) / (num_diffusion_samples - 1))]]).astype(
            np.float32
        )
        for i in range(1, num_diffusion_samples + 1)
    ]
    cond_emb = np.split(npz["cond_emb"], num_diffusion_samples, axis=0)
    if "uncond_emb" in npz:
        uncond_emb = np.split(npz["uncond_emb"], num_diffusion_samples, axis=0)
        calib_data = dict(
            latent=latent * 2,
            timestep=time_emb * 2,
            text_emb=cond_emb + uncond_emb,
        )
    else:
        calib_data = dict(
            latent=latent,
            timestep=time_emb,
            text_emb=cond_emb,
        )
    if num_samples is not None and num_samples < len(calib_data["latent"]):
        rng = np.random.RandomState(42)
        idx = rng.choice(num_diffusion_samples * 2, num_samples, replace=False)
        calib_data = {k: [v[i] for i in idx] for k, v in calib_data.items()}
    return calib_data


def load_vae_calib_dataset_entries(
    path: str, num_samples: int | None = None
) -> DatasetEntries:
    npz = np.load(path)
    num_diffusion_samples = npz["latent"].shape[0]
    calib_data = dict(latent=np.split(npz["latent"], num_diffusion_samples, axis=0))
    if num_samples is not None and num_samples < num_diffusion_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(num_diffusion_samples, num_samples, replace=False)
        calib_data = {k: [v[i] for i in idx] for k, v in calib_data.items()}
    return calib_data


def run_tokenizer(
    tokenizer: CLIPTokenizer, prompt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: cond and uncond token ids, each a int32 torch.Tensor of
    shape [1, tokenizer.model_max_length=77]
    """
    with torch.no_grad():
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        bsz = text_input.input_ids.shape[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * bsz,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    return (
        text_input.input_ids.to(torch.int32),
        uncond_input.input_ids.to(torch.int32),
    )


def count_op_type(model: onnx.ModelProto, op_type: str) -> int:
    """
    Count how many nodes in the given ONNX model have the specified operator type.

    Parameters
    ----------
    model : onnx.ModelProto
        The loaded ONNX model to inspect.
    op_type : str
        The operator type to count (e.g., "Conv", "Relu", "MatMul").

    Returns
    -------
    int
        The number of nodes in the model whose op_type matches the given string.
    """
    return sum(1 for node in model.graph.node if node.op_type == op_type)
