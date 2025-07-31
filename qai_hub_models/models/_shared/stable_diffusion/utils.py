# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from typing import DefaultDict

import cv2
import diffusers
import numpy as np
import onnx
import PIL
import torch
from diffusers.utils import PIL_INTERPOLATION
from qai_hub.client import DatasetEntries
from tqdm import tqdm
from transformers import CLIPTokenizer

from qai_hub_models.models._shared.stable_diffusion.app import (
    UNET_EXTRA_INPUT_NAMES,
    run_diffusion_steps_on_latents,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol


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


def make_calib_data(
    export_path_unet: str | os.PathLike,
    export_path_vae: str | os.PathLike,
    prompt_path: str,
    tokenizer: CLIPTokenizer,
    text_encoder_hf: ExecutableModelProtocol,
    unet_hf: ExecutableModelProtocol,
    scheduler: diffusers.DPMSolverMultistepScheduler,
    num_steps: int = 20,
    num_samples: int = 100,
    guidance_scale: float = 7.5,
    controlnet_hf: ExecutableModelProtocol | None = None,
    export_path_controlnet: str | os.PathLike = "",
    image_cond_path: str | os.PathLike = "",
) -> None:
    """
    Generate calibration data for Unet, Vae, and controlnet if specified.

    Args:

    - export_path_*: must end with .pt

    - prompt_path: txt file where each line is a prompt

    - image_cond_path: .pth file which is a list of canny images in NCHW
    torch.Tensor. Required for controlnet. Must have the same length as
    prompts in prompt_path
    """
    for path in [export_path_unet, export_path_vae, export_path_controlnet]:
        assert str(path).endswith(".pt") or str(path) == ""

    use_controlnet = controlnet_hf is not None
    cond_tokens, uncond_token = load_calib_tokens(
        prompt_path, tokenizer, num_samples=num_samples
    )
    uncond_emb: torch.Tensor | None = None
    if guidance_scale > 0:
        uncond_emb = text_encoder_hf(uncond_token)

    calib_unet: DefaultDict[str, list[torch.Tensor]] = defaultdict(list)
    calib_vae: DefaultDict[str, list[torch.Tensor]] = defaultdict(list)
    if use_controlnet:
        calib_controlnet: DefaultDict[str, list[torch.Tensor]] = defaultdict(list)
        image_conds = torch.load(image_cond_path, weights_only=False)

    for i, cond_token in tqdm(
        enumerate(cond_tokens),
        desc=f"Running {num_steps} diffusion steps on " f"{len(cond_tokens)} samples",
    ):
        cond_emb = text_encoder_hf(cond_token)
        extra_inputs = {}
        if use_controlnet:
            extra_inputs["controlnet"] = controlnet_hf
            extra_inputs["cond_image"] = image_conds[i]

        latent, intermediates = run_diffusion_steps_on_latents(
            unet_hf,
            scheduler=scheduler,
            cond_embeddings=cond_emb,
            uncond_embeddings=uncond_emb,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            return_all_steps=True,
            **extra_inputs,  # type: ignore
        )

        # Add the output to calib_*
        components = [
            ("unet", calib_unet, export_path_unet),
            ("vae", calib_vae, export_path_vae),
        ]

        if use_controlnet:
            components.append(("controlnet", calib_controlnet, export_path_controlnet))

        for name, calib, export_path in components:
            for k, v in intermediates[name].items():
                calib[k].extend(v)
            torch.save(calib, export_path)
            print(f"Data saved to {export_path}")


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
    """
    Load calib data. If uncond_emb is present, duplicate the other inputs
    (latent, timestep etc)
    """
    dict_data = torch.load(path, weights_only=False)
    num_copies = 1
    text_emb = dict_data["cond_emb"]
    if "uncond_emb" in dict_data:
        text_emb.extend(dict_data["uncond_emb"])
        num_copies = 2
    calib_data = {}
    for name in ["latent", "timestep"]:
        calib_data[name] = dict_data[name] * num_copies
    calib_data["text_emb"] = text_emb
    for name in UNET_EXTRA_INPUT_NAMES:
        if name in dict_data:
            calib_data[name] = dict_data[name] * num_copies

    # torch.Tensor -> numpy
    for k, v in calib_data.items():
        calib_data[k] = [v.detach().numpy() for v in calib_data[k]]

    if num_samples is not None and num_samples < len(calib_data["latent"]):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(calib_data["latent"]), num_samples, replace=False)
        calib_data = {k: [v[i] for i in idx] for k, v in calib_data.items()}
    return calib_data


def load_calib_dataset_entries(
    path: str, num_samples: int | None = None
) -> DatasetEntries:
    """
    Use this for vae and controlnet data where we don't need to duplicate
    inputs for cond vs uncond text emb
    """
    calib_data = torch.load(path, weights_only=False)
    # torch.Tensor -> np.ndarray
    for input_name, input_list in calib_data.items():
        calib_data[input_name] = [v.numpy() for v in input_list]

    num_total = len(calib_data["latent"])
    if num_samples is not None and num_samples < num_total:
        rng = np.random.RandomState(42)
        idx = rng.choice(num_total, num_samples, replace=False)
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


def make_canny(
    image: PIL.Image.Image,
    target_height: int,
    target_width: int,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> torch.Tensor:
    """
    Resize and convert from PIL Image to NCHW torch.Tensor.
    """
    img = np.array(image)
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    canny_image = np.concatenate([img, img, img], axis=2)
    canny_image = PIL.Image.fromarray(canny_image)

    # Normalize and make it NCHW
    canny_image = canny_image.convert("RGB")
    canny_image = canny_image.resize(
        (target_width, target_height), resample=PIL_INTERPOLATION["lanczos"]
    )
    canny_image = np.array(canny_image)[None, :].astype(np.float32) / 255.0
    # NHWC -> NCHW
    canny_image = canny_image.transpose(0, 3, 1, 2)
    return torch.from_numpy(canny_image)
