# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from typing_extensions import Self

from qai_hub_models.models.statetransformer.model_patch import custom_one_hot

# Import necessary utilities and patches for the StateTransformer model
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

# Repository and model configuration
STR_SOURCE_REPOSITORY = "https://github.com/Tsinghua-MARS-Lab/StateTransformer.git"
STR_SOURCE_REPO_COMMIT = "b82f9bcf5c9056d0fc5afe9da3350d9bd1c5a9c5"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
STR_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "str_patch.diff")
    )
]
# Default path to pretrained model weights
MODEL_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "checkpoint-66000.zip"
)


def fixup_repo(repo_path: str) -> None:
    find_replace_in_repo(
        repo_path,
        "nuplan_simulation/str_trajectory_generator.py",
        "self._model.cuda()",
        "",
    )
    find_replace_in_repo(
        repo_path,
        "transformer4planning/models/backbone/str_base.py",
        "ModelCls.from_pretrained(model_args.model_pretrain_name_or_path, config=config_p)",
        "ModelCls.from_pretrained(model_args.model_pretrain_name_or_path, config=config_p, ignore_mismatched_sizes=True)",
    )


class StateTransformer(BaseModel):
    """
    A wrapper class for the STR-based transformer model used in planning tasks.
    Provides methods for loading pretrained weights, running inference, and
    specifying input/output formats for deployment or profiling.
    """

    def __init__(self, model: StateTransformer) -> None:
        super().__init__()
        self.model: torch.nn.Module = model
        self.encoder: torch.nn.Module = self.model.encoder
        self.embedding_to_hidden: torch.nn.Module = self.model.embedding_to_hidden
        self.traj_decoder: torch.nn.Module = self.model.traj_decoder
        self.generate_trajs: Any = self.traj_decoder.generate_trajs
        self.generate: torch.nn.Module = self.model.generate

    @classmethod
    def from_pretrained(
        cls, weights_path: str | CachedWebModelAsset | Path = MODEL_PATH
    ) -> Self:
        """
        Load a pretrained StateTransformer model from the specified weights path.

        Parameters
        ----------
        weights_path
            Path to the model weights. Defaults to MODEL_PATH.

        Returns
        -------
        model : Self
            An instance of the StateTransformer class with the loaded model.
        """
        with SourceAsRoot(
            STR_SOURCE_REPOSITORY,
            STR_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            fixup_repo(repo_path=repo_path)

            from transformer4planning.models.backbone.str_base import (
                build_model_from_path,
            )

            if isinstance(weights_path, CachedWebModelAsset):
                weights_path = weights_path.fetch(extract=True).joinpath(
                    "checkpoint-66000"
                )
            model = build_model_from_path(str(weights_path))
        return cls(model)

    def forward(
        self,
        high_res_raster: torch.Tensor,
        low_res_raster: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the trajectory prediction model.

        This function processes multi-resolution bird's-eye-view (BEV) raster inputs
        along with temporal context of past ego actions to predict the future trajectory
        of the ego vehicle. It combines fine-grained local details with global scene
        context and motion history to generate a robust motion plan.

        During inference, the `generate` method of the model is used to produce
        raw prediction outputs, which include trajectory logits and optionally
        key points for trajectory refinement.

        Parameters
        ----------
        high_res_raster
            Rasterized high-resolution BEV image of the scene.
            Shape: (batch_size, 224, 224, 58)
            Dtype: float32
            Represents a fine-grained, small-range view around the ego vehicle,
            capturing detailed road geometry, nearby agents, and traffic elements.
        low_res_raster
            Rasterized low-resolution BEV image covering a larger spatial range.
            Shape: (batch_size, 224, 224, 58)
            Dtype: float32
            Provides global context for high-speed or long-distance planning.
        context_actions
            Past ego motion states or control context describing the vehicle's
            recent actions.
            Shape: (batch_size, 4, 7)
            Dtype: float32
            Typically represents 4 timesteps of ego states such as position (x, y),
            orientation (yaw), velocity, acceleration, and steering.

        Returns
        -------
        traj_logits : torch.Tensor
            Predicted trajectory tensor representing the model's future motion plan.
            Shape: (batch_size, 80, 4)
            Dtype: float32
            Each sample corresponds to an 8-second predicted trajectory consisting of
            80 timesteps and 4 features per step (e.g., x, y, yaw, speed).
        traj_scores : torch.Tensor
            Confidence scores for the predicted trajectories.
            Shape: (batch_size, 1)
            Dtype: float32

        Notes
        -----
        The `generate` method is used for inference only and returns a dictionary
        called `prediction_generation` with the following keys:
            - 'traj_logits': ndarray of shape (batch_size, 80, 4)
            Predicted trajectory logits.
            - 'key_points_logits' (optional): ndarray of shape (batch_size, N, 4)
            Predicted key points for trajectory refinement.
        """
        prepared_data = {
            "high_res_raster": high_res_raster,
            "low_res_raster": low_res_raster,
            "context_actions": context_actions,
        }
        F.one_hot = custom_one_hot if torch.jit.is_tracing() else F.one_hot
        if not torch.jit.is_tracing():
            out_dict = self.generate(**prepared_data)
            return out_dict["traj_logits"], out_dict["traj_scores"]
        input_embeds, info_dict = self.encoder(is_training=False, **prepared_data)
        transformer_outputs = self.embedding_to_hidden(input_embeds)
        hidden_state = transformer_outputs["last_hidden_state"]
        traj_logits = self.generate_trajs(hidden_state, info_dict)
        traj_scores = torch.ones([traj_logits.shape[0], 1]).to(traj_logits.device)
        return traj_logits, traj_scores

    @staticmethod
    def get_input_spec(batch_size: int = 1) -> InputSpec:
        """
        Returns the input specification for the model.

        This specification includes the expected shapes and data types for each input.

        Parameters
        ----------
        batch_size
            The batch size to use in the input specification. Default is 1.

        Returns
        -------
        input_spec : InputSpec
            A dictionary mapping input names to a tuple of (shape, dtype).
        """
        return {
            "high_res_raster": ((batch_size, 224, 224, 58), "float32"),
            "low_res_raster": ((batch_size, 224, 224, 58), "float32"),
            "context_actions": ((batch_size, 4, 7), "float32"),
        }

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["high_res_raster", "low_res_raster"]

    @staticmethod
    def get_output_names() -> list[str]:
        """
        Returns the names of the model outputs.

        Returns
        -------
        output_names : list[str]
            A list containing the names of the outputs produced by the model.
        """
        return ["traj_logits", "traj_scores"]
