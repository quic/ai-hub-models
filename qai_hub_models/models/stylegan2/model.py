# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from qai_hub.client import Device

from qai_hub_models.utils.asset_loaders import SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

STYLEGAN2_SOURCE_REPOSITORY = "https://github.com/NVlabs/stylegan3"
STYLEGAN2_SOURCE_REPO_COMMIT = "c233a919a6faee6e36a316ddd4eddababad1adf9"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl"
)


class StyleGAN2(BaseModel):
    """Exportable StyleGAN2 image generator."""

    def __init__(
        self,
        generator: torch.nn.Module,
        noise_mode="const",
    ) -> None:
        """
        Create a StyleGAN2 model

        Parameters:
            generator:
                Generator object loaded from the StyleGAN repositoru.
            noise_mode:
                Controls noise model introduces into the input.
                Options: 'const', 'random', 'none'
        """
        super().__init__()
        self.generator = generator
        self.output_size: int = self.generator.z_dim  # type: ignore
        self.num_classes: int = self.generator.c_dim  # type: ignore
        self.noise_mode = noise_mode
        assert noise_mode in ["const", "random", "none"]

    @staticmethod
    def from_pretrained(model_url_or_path: str = DEFAULT_WEIGHTS):
        """Load StyleGAN2 from a pickled styleGAN2 file."""
        return StyleGAN2(_load_stylegan2_source_model_from_weights(model_url_or_path))

    def forward(self, image_noise: torch.Tensor, classes: torch.Tensor | None = None):
        """
        Generate an image.

        Parameters:
            image_noise: torch.Tensor | None
                Random state vector from which images should be generated.
                Shape: [ N, self.output_size ]

            classes: torch.tensor
                Tensor of shape [N, self.num_classes].
                If a value of class_idx[b, n] is 1, that class will be generated.
                A maximum of 1 class can be set to 1 per batch.

        Returns:
            A tensor of N generated RGB images. It has shape [N, self.output_size, self.output_size, 3].
        """
        if classes is None:
            classes = torch.zeros((image_noise.shape[0], self.num_classes))
            if self.num_classes != 0:
                classes[:, 0] = 1  # Select first class as default

        return self.generator(
            image_noise,
            classes,
            truncation_psi=1,
            noise_mode=self.noise_mode,
            force_fp32=True,
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1, output_size: int = 512, num_classes: int = 0
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit a profiling job on Qualcomm AI Hub.
        """
        inputs = {"image_noise": ((batch_size, output_size), "float32")}
        if num_classes != 0:
            inputs["classes"] = ((batch_size, num_classes), "float32")
        return inputs

    @staticmethod
    def get_output_names() -> List[str]:
        return ["output_image"]

    def _get_input_spec_for_instance(self, batch_size: int = 1) -> InputSpec:
        return self.__class__.get_input_spec(
            batch_size, self.output_size, self.num_classes
        )

    def sample_inputs(
        self, input_spec: InputSpec | None = None, seed=None
    ) -> Dict[str, List[np.ndarray]]:
        if not input_spec:
            input_spec = self._get_input_spec_for_instance()

        inputs = {
            "image_noise": [
                np.random.RandomState(seed)
                .randn(*input_spec["image_noise"][0])
                .astype(np.float32)
            ]
        }
        if "classes" in input_spec:
            classes = np.zeros(input_spec["classes"][0], dtype=np.float32)
            if input_spec["classes"][0][1] != 0:
                classes[:, 0] = 1  # Select first class as default
            inputs["classes"] = [classes]

        return inputs

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, other_compile_options, device
        )
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--compute_unit" not in compile_options
        ):
            compile_options = compile_options + " --compute_unit gpu"
        return compile_options

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--compute_unit" not in profile_options
        ):
            profile_options = profile_options + " --compute_unit gpu"
        return profile_options


def _get_qaihm_upfirdn2d_ref(misc: Any, conv2d_gradfix: Callable, upfirdn2d: Any):
    """
    Get patched upfirdn2d function implementation that is export compatible.
    This replaces an implementation provided by the stylegan3 repository.
    Params are imports from the stylegan3 repository (see _load_stylegan2_source_model_from_weights).
    """

    @misc.profiled_function
    def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
        """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops."""
        # Validate arguments.
        assert isinstance(x, torch.Tensor) and x.ndim == 4
        if f is None:
            f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
        assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
        assert f.dtype == torch.float32 and not f.requires_grad
        batch_size, num_channels, in_height, in_width = x.shape
        upx, upy = upfirdn2d._parse_scaling(up)
        downx, downy = upfirdn2d._parse_scaling(down)
        padx0, padx1, pady0, pady1 = upfirdn2d._parse_padding(padding)

        # Upsample by inserting zeros.

        # ===== Local change start =====
        # Avoid rank 6.
        # x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
        x = x.reshape([batch_size * num_channels, in_height, 1, in_width, 1])
        # ===== Local change end =====

        x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
        x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = torch.nn.functional.pad(
            x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)]
        )
        x = x[
            :,
            :,
            max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
            max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
        ]

        # Setup filter.
        f = f * (gain ** (f.ndim / 2))
        f = f.to(x.dtype)
        if not flip_filter:
            f = f.flip(list(range(f.ndim)))

        # Convolve with the filter.
        f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
        if f.ndim == 4:
            x = conv2d_gradfix.conv2d(input=x, weight=f, groups=num_channels)
        else:
            x = conv2d_gradfix.conv2d(
                input=x, weight=f.unsqueeze(2), groups=num_channels
            )
            x = conv2d_gradfix.conv2d(
                input=x, weight=f.unsqueeze(3), groups=num_channels
            )

        # Downsample by throwing away pixels.
        x = x[:, :, ::downy, ::downx]
        return x

    return _upfirdn2d_ref


def _load_stylegan2_source_model_from_weights(
    model_url_or_path: str,
) -> torch.nn.Module:
    # Load StyleGAN model from the source repository using the given weights.
    with SourceAsRoot(
        STYLEGAN2_SOURCE_REPOSITORY,
        STYLEGAN2_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        # Patch rank 6 tensor that can't be exported
        from torch_utils import misc
        from torch_utils.ops import conv2d_gradfix, upfirdn2d

        upfirdn2d._upfirdn2d_ref = _get_qaihm_upfirdn2d_ref(
            misc, conv2d_gradfix, upfirdn2d
        )

        # Load model
        import dnnlib
        import legacy

        with dnnlib.util.open_url(model_url_or_path) as f:
            # Get generator
            return legacy.load_network_pkl(f)["G_ema"]
