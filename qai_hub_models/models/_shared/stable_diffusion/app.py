# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections import defaultdict
from typing import Any

import diffusers
import torch
from transformers import CLIPTokenizer

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.inference import OnDeviceModel

OUT_H, OUT_W = 512, 512

UNET_EXTRA_INPUT_NAMES = []
UNET_EXTRA_INPUT_NAMES.extend([f"controlnet_downblock{i}" for i in range(12)])
UNET_EXTRA_INPUT_NAMES.append("controlnet_midblock")


class StableDiffusionApp:
    """
    StableDiffusionApp represents the application code needed to string
    together the various neural networks that make up the Stable Diffusion
    algorithm. This code is written in Python and uses PyTorch and is meant to
    serve as a reference implementation for application in other languages and
    for other platforms.

    Please run the app via `demo.py`.

    References
    ----------
    * https://arxiv.org/abs/2112.10752
    * https://github.com/apple/ml-stable-diffusion
    """

    def __init__(
        self,
        text_encoder: ExecutableModelProtocol,
        vae_decoder: ExecutableModelProtocol,
        unet: ExecutableModelProtocol,
        tokenizer: CLIPTokenizer | Any,
        scheduler: diffusers.DPMSolverMultistepScheduler,
        channel_last_latent: bool,
        host_device: torch.device = torch.device("cpu"),
        controlnet: ExecutableModelProtocol | None = None,
    ):
        """
        Initializes StableDiffusionApp with required neural networks for end-to-end pipeline.

        Parameters
        ----------
        text_encoder:
            Encoder input text
        vae_decoder:
            Decoder to decode latent space into output image
        unet:
            Denoises image in latent space
        tokenizer:
            Tokenizer for input text.
            Output of Tokenizer is fed to text_encoder.
            One can experiments with different tokenizers available based on Clip-ViT.
        scheduler:
            Solver for diffusion steps.
            Updates latent space during each iteration.
        channel_last_latent:
            True if unet outputs latent of shape like (1, 64, 64, 4). False
            for (1, 4, 64, 64)
        """

        self.text_encoder = text_encoder
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.channel_last_latent = channel_last_latent
        self.host_device = host_device
        self.controlnet = controlnet

    def _encode_text_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes a text prompt and returns a tensor with its text embedding.

        Parameters
        ----------
        prompt: The text prompt to encode.

        Returns
        -------
        cond_embedding

        uncond_embedding

        Note that uncond_embedding is the same for any prompt (since it's not
        conditioned on the prompt). So in deploymenet this should be
        cached instead of computed every time. We compute it here for better
        clarity.
        """
        with torch.no_grad():
            # Tokenize input prompt
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

            # Tokenize empty prompt
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # Embed using the text encoder neural network
            # Encode input and empty prompt in one go
            print(f"\nExtracting embeddings (inference on TextEncoder)\n{'-' * 50}")
            if isinstance(self.text_encoder, OnDeviceModel):
                # Batch data into one inference job
                embeddings = self.text_encoder(
                    [
                        text_input.input_ids.int(),
                        uncond_input.input_ids.int(),
                    ]
                )
                assert isinstance(embeddings, torch.Tensor)
                cond_embeddings, uncond_embeddings = torch.split(embeddings, 1, 0)
            else:
                cond_embeddings = self.text_encoder(
                    text_input.input_ids.type(torch.int32)
                )
                uncond_embeddings = self.text_encoder(
                    uncond_input.input_ids.type(torch.int32)
                )
            return cond_embeddings, uncond_embeddings

    def predict(self, *args, **kwargs):
        # See generate_image.
        return self.generate_image(*args, **kwargs)

    def generate_image(
        self,
        prompt: str,
        num_steps: int = 50,
        seed: int = 0,
        guidance_scale: float = 7.5,
        cond_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate an image using the PyTorch reference neural networks. This
        code can be used as a reference for how to glue together the neural
        networks in an application. Note that this code relies on a tokenizer
        and scheduler from the HuggingFace's diffusers library, so those would
        have to be ported to the application as well.

        Parameters
        ----------
        prompt:
            The text prompt to generate an image from.
        num_steps:
            The number of steps to run the diffusion process for. Higher value
            may lead to better image quality.
        seed:
            The seed to use for the random number generator.
        guidance_scale:
            Classifier-free guidance is a method that allows us to control how
            strongly the image generation is guided by the prompt. This is done
            by always processing two samples at once: an unconditional (using a
            text embedding of an empty prompt) and a conditional (using a text
            embedding of the provided prompt). Given the noise prediction of
            both of these, we linearly interpolate between them based on the
            guidance_scale. A guidance scale of 0 is the same as using an empty
            prompt. A guidance scale of 1 turns off classifier-free guidance
            and is computationally less expensive since it only processes one
            sample at a time. Intuitively you may think the rest of guidance
            scales are between 0 and 1, but it is common to use a scale greater
            than 1 as a method of amplifying the prompt's influence on the
            image, pushing it further away from the unconditional sample.

        Returns
        -------
        torch.Tensor
            The generated image in RGB scaled in [0, 1] with tensor shape
            (OUT_H, OUT_W, 3). The height and the width may depend on the
            underlying Stable Diffusion version, but is typically 512x512.
        """

        # Encode text prompt
        cond_embeddings, uncond_embeddings = self._encode_text_prompt(prompt)

        latents = run_diffusion_steps_on_latents(
            unet=self.unet,
            scheduler=self.scheduler,
            cond_embeddings=cond_embeddings,
            uncond_embeddings=uncond_embeddings,
            num_steps=num_steps,
            seed=seed,
            guidance_scale=guidance_scale,
            channel_last_latent=self.channel_last_latent,
            host_device=self.host_device,
            controlnet=self.controlnet,
            cond_image=cond_image,
        )
        # Decode generated image from latent space
        if self.channel_last_latent:
            latents = _make_channel_last_torch(latents).to(self.host_device)
        image = self.vae_decoder(latents)
        return image.to("cpu")  # move to cpu in case it was run on gpu


def run_diffusion_steps_on_latents(
    unet: ExecutableModelProtocol,
    scheduler: diffusers.DPMSolverMultistepScheduler,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor | None = None,
    num_steps: int = 20,
    seed: int = 0,
    guidance_scale: float = 7.5,
    channel_last_latent: bool = False,
    return_all_steps: bool = False,
    host_device: torch.device = torch.device("cpu"),
    controlnet: ExecutableModelProtocol | None = None,
    cond_image: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Runs the diffusion steps on latents to generate the final latent sample.

    When guidance_scale is nonzero, classifier-free guidance is applied by computing
    both conditional and unconditional noise predictions. In that case, `uncond_embeddings`
    must be provided. If guidance_scale is 0, no guidance is applied and only the conditional
    branch is used. If return_all_steps is True, this function returns a tuple of the
    final latent and a dictionary of intermediate inputs.

    The returned intermediates has:
    - intermediates["unet"]: dict with keys "latent", "cond_emb", and optionally "uncond_emb",
      each a list of torch.Tensor matching each diffusion step's inputs to the UNet.
    - intermediates["vae"]: the final latent input to the VAE decoder.

    Parameters
    ----------
    unet:
        The denoising network.
    scheduler:
        The scheduler controlling the diffusion process.
    cond_embeddings:
        Conditional text embeddings.
    uncond_embeddings:
        Unconditional text embeddings. This is required if guidance_scale != 0.
    num_steps:
        Number of diffusion steps.
    seed:
        Seed for random number generation.
    guidance_scale:
        Scale for classifier-free guidance. If nonzero, both conditional and unconditional
        noise predictions are computed.
    channel_last_latent:
        True if the unet outputs latents in channel-last format.
    return_all_steps:
        If True, returns intermediate inputs for calibration along with final latent.
    host_device:
        Device on which to perform computation.
    cond_image:
        canny image in NCHW torch.Tensor.

    Returns
    -------
    torch.Tensor
        Final latent sample.
    tuple[torch.Tensor, dict[str, Any]]
        Tuple of final latent and intermediates dict (only if return_all_steps is True).
    """
    use_controlnet = controlnet is not None
    unet_extra_input_names = []
    if use_controlnet:
        unet_extra_input_names = UNET_EXTRA_INPUT_NAMES[:]
    with torch.no_grad():
        # Prepare scheduler and initial noise
        scheduler.set_timesteps(num_steps)  # type: ignore[attr-defined]
        latents_shape = (1, 4, OUT_H // 8, OUT_W // 8)
        generator = torch.manual_seed(seed)
        latents = torch.randn(latents_shape, generator=generator, device=host_device)
        latents = latents * scheduler.init_noise_sigma  # type: ignore[attr-defined]

        # Initialize storage for UNet calibration data
        unet_inputs = defaultdict(list)

        if use_controlnet:
            controlnet_inputs = defaultdict(list)

        for i, t in enumerate(scheduler.timesteps):  # type: ignore[attr-defined]
            print(f"\nStep: {i + 1}\n{'-' * 10}")

            time_input = torch.as_tensor([[t]], dtype=torch.float32).to(host_device)

            latent_input = scheduler.scale_model_input(  # type: ignore[attr-defined]
                latents, t
            )
            if channel_last_latent:
                latent_input = _make_channel_last_torch(latent_input).to(host_device)

            controlnet_out: tuple[torch.Tensor, ...] = tuple()
            if use_controlnet:
                controlnet_inputs["latent"].append(latent_input)
                controlnet_inputs["timestep"].append(time_input)
                controlnet_inputs["text_emb"].append(cond_embeddings)
                controlnet_inputs["image_cond"].append(cond_image)
                assert controlnet is not None
                controlnet_out = controlnet(
                    latent_input, time_input, cond_embeddings, cond_image
                )
                if channel_last_latent:
                    controlnet_out = tuple(
                        _make_channel_first_torch(v) for v in controlnet_out
                    )

            # Store inputs for calibration
            unet_inputs["latent"].append(latent_input)
            unet_inputs["timestep"].append(time_input)
            unet_inputs["cond_emb"].append(cond_embeddings)
            if use_controlnet:
                for i, name in enumerate(unet_extra_input_names):
                    unet_inputs[name].append(controlnet_out[i])

            if guidance_scale != 0:
                if uncond_embeddings is None:
                    raise ValueError(
                        "uncond_embeddings must be provided when guidance_scale"
                        " is nonzero"
                    )

                unet_inputs["uncond_emb"].append(uncond_embeddings)

                if isinstance(unet, OnDeviceModel):
                    # Batch data into one inference job
                    unet_extra_inputs = tuple([t, t] for t in controlnet_out)
                    noise = unet(
                        [latent_input, latent_input],
                        [time_input, time_input],
                        [cond_embeddings, uncond_embeddings],
                        *unet_extra_inputs,
                    )
                    noise_cond, noise_uncond = torch.split(noise, 1, 0)
                else:
                    noise_cond = unet(
                        latent_input, time_input, cond_embeddings, *controlnet_out
                    )
                    noise_uncond = unet(
                        latent_input, time_input, uncond_embeddings, *controlnet_out
                    )
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                if isinstance(unet, OnDeviceModel):
                    # Batch data into one inference job
                    unet_extra_inputs = tuple([t] for t in controlnet_out)
                    noise_pred = unet(
                        [latent_input],
                        [time_input],
                        [cond_embeddings],
                        *unet_extra_inputs,
                    )
                else:
                    noise_pred = unet(
                        latent_input, time_input, cond_embeddings, *controlnet_out
                    )

            if channel_last_latent:
                noise_pred = _make_channel_first_torch(noise_pred).to(host_device)
            latents = scheduler.step(  # type: ignore[attr-defined]
                noise_pred, t, latents
            ).prev_sample

        if return_all_steps:
            vae_inputs = {"latent": [latents]}
            intermediates = {"unet": unet_inputs, "vae": vae_inputs}
            if use_controlnet:
                intermediates["controlnet"] = controlnet_inputs
            # Detach grad and move to cpu
            for model, inputs in intermediates.items():
                for input_name, input_list in intermediates[model].items():
                    intermediates[model][input_name] = [
                        v.detach().cpu() for v in input_list
                    ]
            return latents, intermediates

        return latents


# Helper method to go back and forth from channel-first to channel-last
def _make_channel_last_torch(input_tensor):
    return torch.permute(input_tensor, [0, 2, 3, 1])


def _make_channel_first_torch(input_tensor):
    return torch.permute(torch.Tensor(input_tensor), [0, 3, 1, 2])
