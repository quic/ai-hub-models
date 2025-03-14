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
from tqdm import tqdm

from qai_hub_models.models._shared.stable_diffusion.app import (
    run_diffusion_steps_on_latents,
)
from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.model import (
    load_calib_tokens,
)


def export_onnx_in_memory(
    torch_model: torch.nn.Module,
    example_input: tuple[torch.Tensor, ...],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
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
    text_encoder_hf,
    unet_hf,
    time_embedding_hf,
    scheduler,
    num_steps: int = 20,
    num_samples_unet: int = 100,
    num_samples_vae: int = 500,
) -> tuple[str, str]:
    """Generate calibration data for Unet and Vae and save as npz.

    This will generate

    Number of samples for unet: num_samples_unet * num_steps * 2 (*2 for cond and
    uncond text embeddings). e.g., 100 samples * 20 steps  * 2 = 20k

    Number of samples for vae: num_samples_vae.

    The max for num_samples_unet and num_samples_vae is 500 (the number of
    prompts in our dataset)

    Returns:
    - export_path_unet
    - export_path_vae
    """
    export_path_unet = os.path.join(
        export_dir, f"unet_calib_n{num_samples_unet}_t{num_steps}.npz"
    )
    export_path_vae = os.path.join(
        export_dir, f"vae_calib_n{num_samples_vae}_t{num_steps}.npz"
    )
    num_samples = max(num_samples_unet, num_samples_vae)
    cond_tokens, uncond_token = load_calib_tokens(num_samples=num_samples)

    # uncond_emb doesn't change for each prompt
    uncond_emb = text_encoder_hf(uncond_token)

    calib_unet = dict(
        latent=[],
        time_emb=[],
        cond_emb=[],
        uncond_emb=[],
    )  # type: ignore
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
            time_embedding=time_embedding_hf,
            cond_embeddings=cond_emb,
            uncond_embeddings=uncond_emb,
            num_steps=num_steps,
            return_all_steps=True,
        )
        if i < num_samples_unet:
            calib_unet["latent"].extend(all_steps["latent"])
            calib_unet["time_emb"].extend(all_steps["time_emb"])
            calib_unet["cond_emb"].extend([cond_emb] * num_steps)
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
