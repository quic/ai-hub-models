# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
from diffusers.models.embeddings import get_timestep_embedding
from PIL import Image
from torchvision import transforms

OUT_H, OUT_W = 512, 512


class ControlNetApp:
    """
    ControlNetApp represents the application code needed to string
    together the various neural networks that make up the ControlNet
    algorithm. This code is written in Python and pipeline uses PyTorch
    while running neural networks on-device. This is meant to serve as a
    reference implementation for this application in other languages and
    for other platforms.

    Please run the app via `demo.py`.

    References
    ----------
    * https://arxiv.org/abs/2302.05543
    * https://github.com/lllyasviel/ControlNet
    """

    def __init__(
        self,
        text_encoder: Callable[..., tuple[torch.Tensor, ...]],
        vae_decoder: Callable[..., tuple[torch.Tensor, ...]],
        unet: Callable[..., tuple[torch.Tensor, ...]],
        controlnet: Callable[..., tuple[torch.Tensor, ...]],
        tokenizer: Any,
        scheduler: Any,
        time_embedding: Any,
    ):
        """
        Initializes ControlNetApp with required neural networks for end-to-end pipeline.

        Parameters
        ----------
        text_encoder:
            Encoder input text
        vae_decoder:
            Decoder to decode latent space into output image
        unet:
            Denoises image in latent space
        controlnet:
            Conditions denoise w.r.t. input image
        tokenizer:
            Tokenizer for input text.
            Output of Tokenizer is fed to text_encoder.
            One can experiments with different tokenizers available based on Clip-ViT.
        scheduler:
            Solver for diffusion steps.
            Updates latent space during each iteration.
        time_embeddings:
            Projects time-step into embedding used during denoising in latent space.
        """

        self.text_encoder = text_encoder
        self.vae_decoder = vae_decoder
        self.unet = unet
        self.controlnet = controlnet
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.time_embedding = time_embedding

    def get_time_embedding(self, timestep):
        timestep = torch.tensor([timestep])
        t_emb = get_timestep_embedding(timestep, 320, True, 0)
        emb = self.time_embedding(t_emb)

        return emb

    def _make_canny_image(self, input_image: Image.Image):
        image = np.asarray(input_image)

        # Get edges for input with Canny Edge Detection
        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)

        # Make image channel-first and scale
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        torch_image = torch.Tensor(image).unsqueeze(0)

        # Resize input image to supported size
        return transforms.Resize(size=(OUT_H, OUT_W))(torch_image)

    def _encode_text_prompt(self, prompt: str) -> torch.Tensor:
        """
        Takes a text prompt and returns a tensor with its text embedding.

        Parameters
        ----------
        prompt:
            The text prompt to encode.
        """
        # Tokenize input prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

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
        input_image: Image.Image,
        num_steps: int = 5,
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
        input_image:
            Path to input image for conditioning image generation.
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
            The generated image in RGB scaled in [0, 1] with tensor shape (H,
            W, 3). The height and the width may depend on the underlying Stable
            Diffusion version, but is typically 512x512.
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

        # Get image with edges for conditioning
        canny_image = self._make_canny_image(input_image)
        canny_image = _make_channel_last_torch(canny_image)

        for i, t in enumerate(self.scheduler.timesteps):
            print(f"\nStep: {i + 1}\n{'-' * 10}")
            time_emb = self.get_time_embedding(t)
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            latent_model_input = _make_channel_last_torch(latent_model_input)

            # Denoise input
            print(
                f"\nDenoising image in latent space (inference on ControlNet)\n{'-' * 60}"
            )
            controlnet_out = self.controlnet(
                [latent_model_input] * 2,
                [time_emb] * 2,
                [cond_embeddings, uncond_embeddings],
                [canny_image] * 2,
            )
            controlnet_out_split = []
            for each in controlnet_out:
                controlnet_out_split.append(torch.split(each, 1, 0))

            print(f"\nDenoising image in latent space (inference on UNet)\n{'-' * 50}")
            noise_pred = self.unet(
                [latent_model_input] * 2,
                [time_emb] * 2,
                [cond_embeddings, uncond_embeddings],
                *controlnet_out_split,
            )
            noise_cond, noise_uncond = torch.split(noise_pred, 1, 0)

            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            noise_pred = _make_channel_first_torch(noise_pred)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        print(f"\nDecoding generated image (inference on VAEDecoder)\n{'-' * 50}")
        # Decode generated image from latent space
        latents_vae = _make_channel_last_torch(latents)
        image = self.vae_decoder(latents_vae)
        return image
