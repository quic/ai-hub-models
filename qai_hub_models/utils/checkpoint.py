# pyupgrade: skip-file
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import warnings
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, TypeVar, Union

import onnx
import torch

from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.input_spec import make_torch_inputs

if TYPE_CHECKING:
    # this import is only for the type‐checker, never executed at runtime
    from transformers import PreTrainedModel


CheckpointSpec = Union[os.PathLike, Literal["DEFAULT", "DEFAULT_UNQUANTIZED"]]


@unique
class CheckpointType(Enum):

    # For most models, the default is the pretrained fp checkpoint, which can be
    # optionally quantized via submit_quantize_job.
    #
    # For models requiring AIMET to quantize locally, the default uses quantized
    # checkpoint (quantized by AIMET), instead of the fp checkpoint. For these
    # models use DEFAULT_UNQUANTIZED to force the fp checkpoint.
    DEFAULT = "DEFAULT"

    # Default pretrained weights without encodings. Used for creating
    # pre-calibrated model and encodings
    DEFAULT_UNQUANTIZED = "DEFAULT_UNQUANTIZED"

    # A single PyTorch checkpoint file (.pth or .pt)
    TORCH_STATE_DICT = "TORCH_STATE_DICT"

    # HuggingFace-style local checkpoint: typically contains config.yaml and model.safetensor
    HF_LOCAL = "HF_LOCAL"

    # Huggingface repo id
    HF_REPO = "HF_REPO"

    # Aimet-exported ONNX checkpoint: usually includes model.onnx,
    # model.encodings, and optionally model.data
    AIMET_ONNX_EXPORT = "AIMET_ONNX_EXPORT"

    # invalid or unrecognized checkpoint
    INVALID = "INVALID"


def hf_repo_exists(repo_id: str) -> bool:
    """
    Return True if `repo_id` is a valid, existing Hugging Face repo.
    If huggingface_hub isn't installed, warn once and return False.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        warnings.warn(
            "huggingface_hub is not installed; "
            f"Unable to check if {repo_id} is a valid HF repo",
            ImportWarning,
        )
        return False

    api = HfApi()
    try:
        api.model_info(repo_id)
        return True
    except Exception:
        return False


def determine_checkpoint_type(
    checkpoint: CheckpointSpec, subfolder: str = ""
) -> CheckpointType:
    """
    Determines the type of checkpoint referred to by `checkpoint`, which may be:
      - one of the special enums DEFAULT[_…]
      - a local directory containing HF_LOCAL / ONNX / Torch files
      - a remote HF repo ID string
      - invalid / unrecognized
    """
    # 1) Handle our two built-in sentinels
    if checkpoint in ["DEFAULT_UNQUANTIZED", CheckpointType.DEFAULT_UNQUANTIZED]:
        return CheckpointType.DEFAULT_UNQUANTIZED
    if checkpoint in ["DEFAULT", CheckpointType.DEFAULT]:
        return CheckpointType.DEFAULT

    # 2) Non-existent strings -> remote HF lookup
    if isinstance(checkpoint, str):
        cp_path = Path(checkpoint)
        if not cp_path.exists():
            return (
                CheckpointType.HF_REPO
                if hf_repo_exists(checkpoint)
                else CheckpointType.INVALID
            )

    # 3) From here on, we must have an existing directory
    cp_path = Path(checkpoint)
    if subfolder != "":
        cp_path = cp_path / subfolder
    if not cp_path.is_dir():
        return CheckpointType.INVALID

    # 4) Local HF‐style (config + safetensor)
    if (cp_path / "config.yaml").is_file() and (cp_path / "model.safetensor").is_file():
        return CheckpointType.HF_LOCAL

    # 5) Aimet ONNX export (ONNX + encodings)
    if cp_path.glob("model*.onnx") and (cp_path / "model.encodings").is_file():
        return CheckpointType.AIMET_ONNX_EXPORT

    # 6) Single PyTorch state‐dict
    torch_files = list(cp_path.glob("*.pth")) + list(cp_path.glob("*.pt"))
    if len(torch_files) == 1:
        return CheckpointType.TORCH_STATE_DICT

    # 7) Nothing matched
    return CheckpointType.INVALID


T = TypeVar("T", bound="FromPretrainedMixin")


class FromPretrainedMixin(Generic[T]):
    """
    FromPretrainedMixin helps models quantized by AIMET loads checkpoints
    (both fp and quantized).

    Mixin providing:
      - torch_from_pretrained(...)
      - onnx_from_pretrained(...)
      - from_pretrained(...)

    Expects subclasses to define:
      - hf_repo_id:         default repo id e.g., "stabilityai/stable-diffusion-2-1-base"
      - hf_model_cls:       a transformers.PreTrainedModel subclass
      - default_subfolder:  str (automatically defined by
              CollectionModel.add_component)

    Optionally, a subclass can implement:
      @classmethod
      def adapt_torch_model(cls, model: nn.Module, **kwargs) -> nn.Module: ...
      @classmethod
      def get_input_spec(cls) -> Mapping[str, torch.Tensor]: ...
      @classmethod
      def get_output_names(cls) -> List[str]: ...
      @classmethod
      def get_calibrated_aimet_model(cls) -> Tuple[str, str]: ...
    """

    hf_repo_id: str = ""
    hf_model_cls: type[PreTrainedModel]
    default_subfolder: str = ""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @classmethod
    def torch_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: Union[torch.device, str] = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> torch.nn.Module:
        subfolder = subfolder or cls.default_subfolder
        ckpt_type = determine_checkpoint_type(checkpoint, subfolder=subfolder)

        if ckpt_type == CheckpointType.TORCH_STATE_DICT:
            raise NotImplementedError(
                "FromPretrainedMixin does not support torch state dict checkpoint"
            )

        required_ckpt_types = [
            CheckpointType.DEFAULT,
            CheckpointType.DEFAULT_UNQUANTIZED,
            CheckpointType.HF_LOCAL,
            CheckpointType.HF_REPO,
        ]
        if ckpt_type not in required_ckpt_types:
            raise ValueError(
                f"{checkpoint} ({subfolder=}) is an unsupported "
                f"checkpoint type {ckpt_type}. for torch_from_pretrained"
            )

        # 2) load from HF Hub or local
        if ckpt_type in [
            CheckpointType.DEFAULT,
            CheckpointType.DEFAULT_UNQUANTIZED,
        ]:
            if cls.hf_repo_id == "":
                raise NotImplementedError(
                    "Default Huggingface repo " "not defined in cls.hf_repo_id"
                )
            model = cls.hf_model_cls.from_pretrained(
                cls.hf_repo_id,
                subfolder=subfolder,
            )
        elif ckpt_type == CheckpointType.HF_REPO:
            model = cls.hf_model_cls.from_pretrained(
                checkpoint,
                subfolder=subfolder,
            )
        else:  # ckpt_type == CheckpointType.HF_LOCAL
            model = cls.hf_model_cls.from_pretrained(str(Path(checkpoint) / subfolder))

        # 3) move to target device
        model = model.to(torch.device(host_device)).eval()

        # 4) if subclass provides adapt_torch_model, call it
        if hasattr(cls, "adapt_torch_model"):
            adapt_torch_model_options = adapt_torch_model_options or {}
            model = cls.adapt_torch_model(model, **adapt_torch_model_options)
            model = model.to(torch.device(host_device)).eval()
        else:
            assert adapt_torch_model_options is None

        return model

    @classmethod
    def onnx_from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: Union[torch.device, str] = torch.device("cpu"),
        torch_to_onnx_options: dict | None = None,
    ) -> tuple[onnx.ModelProto, Optional[str]]:
        """
        Load the checkpoint into ONNX, possibly with AIMET encodings
        if the checkpoint is already quantized.

        Args:
          checkpoint:

            - "DEFAULT": load pre-calibrated model + encodings

            - "DEFAULT_UNQUANTIZED": load public pretrained model WITHOUT encodings

            - Path to HF checkpoint (checkpoint/{config.yaml,model.safetensor}):
              load weights but NO encodings (must quantize before export/run)

            - Path to Aimet-ONNX checkpoint (checkpoint/{model.onnx,model.encodings}):
              load both model + encodings

        Returns
        - onnx_model, aimet_encodings_path
        """
        subfolder = subfolder or cls.default_subfolder
        host_device = torch.device(host_device)
        ckpt_type = determine_checkpoint_type(checkpoint, subfolder=subfolder)

        is_quantized_src = ckpt_type in (
            CheckpointType.DEFAULT,
            CheckpointType.AIMET_ONNX_EXPORT,
        )
        aimet_encodings = None

        if is_quantized_src:
            if ckpt_type == CheckpointType.DEFAULT:
                onnx_path, aimet_encodings = cls.get_calibrated_aimet_model()
            else:
                # AIMET-exported directory
                subfolder_path = Path(checkpoint) / subfolder
                onnx_path = str(subfolder_path / "model.onnx")
                aimet_encodings = str(subfolder_path / "model.encodings")
            onnx_model = onnx.load(onnx_path)

        else:
            cp = checkpoint
            # torch.nn.Module has no notion of DEFAULT_UNQUANTIZED
            if ckpt_type == CheckpointType.DEFAULT_UNQUANTIZED:
                cp = "DEFAULT"
            fp_model = cls.torch_from_pretrained(
                checkpoint=cp,
                subfolder=subfolder,
                host_device=host_device,
            )

            input_spec = cls.get_input_spec()  # type: ignore

            example_input = tuple(make_torch_inputs(input_spec))  # type: ignore
            example_input = tuple([t.to(host_device) for t in example_input])

            torch_to_onnx_options = torch_to_onnx_options or {}
            with qaihm_temp_dir() as tmpdir:
                out_onnx = os.path.join(tmpdir, "model.onnx")
                torch.onnx.export(
                    fp_model,
                    example_input,
                    out_onnx,
                    input_names=list(input_spec.keys()),  # type: ignore
                    output_names=cls.get_output_names(),  # type: ignore
                    **torch_to_onnx_options,
                )
                onnx_model = onnx.load(out_onnx)

        return onnx_model, aimet_encodings

    @classmethod
    def from_pretrained(
        cls: type[T],
        *args: Any,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        **kwargs: Any,
    ) -> T:
        """
        This assumes that the class takes a single torch.nn.Module in
        __init__. Override if not.
        """
        host_device = torch.device(host_device)
        return cls(
            cls.torch_from_pretrained(
                checkpoint=checkpoint,
                subfolder=subfolder,
                host_device=host_device,
            )
        )

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        # Returns .onnx and .encodings paths
        raise NotImplementedError()
