# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Tuple

import diffusers
import torch
from diffusers.models.embeddings import get_timestep_embedding
from transformers import CLIPTokenizer

from qai_hub_models.utils.inference import OnDeviceModel

OUT_H, OUT_W = 512, 512


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
        text_encoder: OnDeviceModel | torch.nn.Module,
        vae_decoder: OnDeviceModel | torch.nn.Module,
        unet: OnDeviceModel | torch.nn.Module,
        tokenizer: CLIPTokenizer | Any,
        scheduler: diffusers.DPMSolverMultistepScheduler,
        time_embedding: diffusers.embeddings.TimeEmbedding,
        channel_last_latent: bool,
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
        time_embedding:
            Projects time-step into embedding used during denoising in latent space.
        channel_last_latent:
            True if unet outputs latent of shape like (1, 64, 64, 4). False
            for (1, 4, 64, 64)
        """

        self.text_encoder = text_encoder
        self.vae_decoder = vae_decoder
        self.unet = unet
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.time_embedding = time_embedding
        self.channel_last_latent = channel_last_latent

    def get_time_embedding(self, timestep):
        """
        Since these time embeddings aren't dependent on prompt, they can be
        pre-computed (for a pre-defined set of timesteps) in deployment and
        skip these computation. We include them in demo for better clarity.
        """
        timestep = torch.tensor([timestep])
        # TODO: pull 320 from UNet block output dim
        t_emb = get_timestep_embedding(timestep, 320, True, 0)
        emb = self.time_embedding(t_emb)

        return emb

    def _encode_text_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
        embeddings = self.text_encoder(
            [
                text_input.input_ids.type(torch.int32),
                uncond_input.input_ids.type(torch.int32),
            ]
        )
        cond_embeddings, uncond_embeddings = torch.split(embeddings, 1, 0)
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
        self.scheduler.set_timesteps(num_steps)
        self.scheduler.config.prediction_type = "epsilon"

        # Channel last input
        latents_shape = (1, 4, OUT_H // 8, OUT_W // 8)

        generator = torch.manual_seed(seed)
        latents = torch.randn(latents_shape, generator=generator)

        latents = latents * self.scheduler.init_noise_sigma

        # Helper method to go back and forth from channel-first to channel-last
        def _make_channel_last_torch(input_tensor):
            return torch.permute(input_tensor, [0, 2, 3, 1])

        def _make_channel_first_torch(input_tensor):
            return torch.permute(torch.Tensor(input_tensor), [0, 3, 1, 2])

        for i, t in enumerate(self.scheduler.timesteps):
            print(f"\nStep: {i + 1}\n{'-' * 10}")
            time_emb = self.get_time_embedding(t)
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            if self.channel_last_latent:
                latent_model_input = _make_channel_last_torch(latent_model_input)

            print(f"\nDenoising image in latent space (inference on UNet)\n{'-' * 50}")
            # Denoise image in latent space
            noise = self.unet(
                [latent_model_input, latent_model_input],
                [time_emb, time_emb],
                [cond_embeddings, uncond_embeddings],
            )

            noise_cond, noise_uncond = torch.split(noise, 1, 0)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            if self.channel_last_latent:
                noise_pred = _make_channel_first_torch(noise_pred)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        print(f"\nDecoding generated image (inference on VAEDecoder)\n{'-' * 50}")
        # Decode generated image from latent space
        if self.channel_last_latent:
            latents = _make_channel_last_torch(latents)
        image = self.vae_decoder(latents)
        return image
