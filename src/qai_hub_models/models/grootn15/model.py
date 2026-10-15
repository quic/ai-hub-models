# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any, cast

import aimet_onnx
import onnx
import torch
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.experimental.spinquant import apply_spinquant as apply_spinquant_onnx
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.utils import duplicate_shared_initializers
from onnxruntime.transformers.optimizer import optimize_model as ort_optimize_model
from onnxsim import simplify
from qai_hub.client import Device
from torch import nn
from transformers import AutoConfig

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.grootn15.constants import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EMBODIMENT_TAG,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    VLM_LANGUAGE_TOKENS,
)
from qai_hub_models.models.grootn15.evaluator import LeRobotEvaluator
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.transform.base import (
    ComposedModalityTransform,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.experiment.data_config import (
    DATA_CONFIG_MAP,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.backbone.eagle_backbone import (
    DEFAULT_EAGLE_PATH,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.policy import (
    Gr00tPolicy,
)
from qai_hub_models.models.grootn15.model_adaptations import (
    ActionHeadDiTOpt,
    EagleBackboneOpt,
    SelfAttentionTransformerOpt,
    SiglipVisionTransformerOpt,
    bypass_prepare_attention_mask,
    prepare_4d_bidirectional_mask,
    prepare_4d_causal_attention_mask_with_cache_position,
)
from qai_hub_models.models.grootn15.utils import (
    onnx_remove_large_initializers,
    onnx_restore_large_initializers,
)
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.aimet.encodings import apply_propagate_memory_encodings
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_evaluator import BaseEvaluator, _DataLoader
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.checkpoint import CheckpointSpec, FromPretrainedMixin
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, make_torch_inputs
from qai_hub_models.utils.onnx.helpers import (
    ONNXBundle,
    extract_io_types_from_onnx_model,
)
from qai_hub_models.utils.qai_hub_helpers import (
    ensure_hexagon_version,
    export_torch_to_onnx_zip,
)
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
    aimet_quant_types,
    ensure_min_aimet_onnx_version,
)

_GROOT_AIMET_CONFIG = str(Path(__file__).parent / "aimet_config.json")

DEFAULT_CHECKPOINT_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "model_finetuned_gr1_picknplace.tar.gz"
)

DEFAULT_DATASET_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "robot_sim_PickNPlace.tar.gz"
)


@lru_cache(maxsize=1)
def load_checkpoint(
    checkpoint: str,
    data_config: str,
    embodiment_tag: str,
    device: str = "cpu",
) -> Gr00tPolicy:
    """Load and cache a Gr00tPolicy from a model path or HF hub ID."""
    if checkpoint == "DEFAULT":
        checkpoint = str(DEFAULT_CHECKPOINT_ASSET.fetch(extract=True))
    print(f"Loading Groot checkpoint: {checkpoint}")
    data_cfg = DATA_CONFIG_MAP[data_config]
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()
    assert isinstance(modality_transform, ComposedModalityTransform)
    policy = Gr00tPolicy(
        model_path=checkpoint,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    policy.model.eval()
    policy.model = policy.model.float()
    return policy


class LoadGrootMixin(FromPretrainedMixin):
    """
    Shared mixin for all Groot component classes.
    Handles checkpoint loading and ONNX export to AIHub format.
    """

    @classmethod
    def torch_from_pretrained(  # type: ignore[override, unused-ignore]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
    ) -> Gr00tPolicy:
        return load_checkpoint(
            checkpoint=str(checkpoint),
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            device=str(host_device),
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        class_name = self.__class__.__name__
        path = Path(output_path) / f"{class_name}.onnx"
        assert input_spec is not None
        if path.exists():
            return str(path)
        return export_torch_to_onnx_zip(
            self.to("cpu"),  # type: ignore[attr-defined]
            str(path),
            make_torch_inputs(input_spec),
            input_names=list(input_spec.keys()),
            skip_zip=False,
            output_names=output_names,
        )

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        return ensure_hexagon_version(
            min_version=73,
            target_runtime=target_runtime,
            device=device,
            model_name="GR00T-N1.5",
        )

    def get_input_spec(self) -> InputSpec:
        """Return ordered list of input tensor names for this component."""
        raise NotImplementedError

    def get_output_spec(self) -> OutputSpec:
        """Return ordered list of output tensor names for this component."""
        raise NotImplementedError

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        class_name = self.__class__.__name__
        path = Path(output_dir) / f"{class_name}.onnx"
        assert input_spec is not None
        if path.exists():
            return path
        return Path(
            export_torch_to_onnx_zip(
                self.to("cpu"),  # type: ignore[attr-defined]
                str(path),
                make_torch_inputs(input_spec),
                input_names=list(input_spec.keys()),
                skip_zip=False,
            )
        )


class GrootViT(LoadGrootMixin, BaseModel):
    """
    Eagle ViT vision encoder for Groot N1.5.
    Accepts pixel_values [B*num_cameras, C, H, W] and returns
    visual embeddings [B*num_cameras, num_patches, embed_dim].

    Preprocessing is handled upstream in GrootApp.
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        self.eagle_config = AutoConfig.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
        )

        self.vision_model = SiglipVisionTransformerOpt(self.eagle_config.vision_config)

        vit_hidden_size = self.eagle_config.vision_config.hidden_size
        llm_hidden_size = self.eagle_config.text_config.hidden_size

        if self.eagle_config.mlp_connector_layers == 2:
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(
                    vit_hidden_size * int(1 / self.eagle_config.downsample_ratio) ** 2
                ),
                nn.Linear(
                    vit_hidden_size * int(1 / self.eagle_config.downsample_ratio) ** 2,
                    llm_hidden_size,
                ),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        elif (
            self.eagle_config.mlp_connector_layers == 1
            and self.eagle_config.use_pixel_shuffle
        ):
            self.mlp1 = nn.Sequential(
                nn.Linear(
                    vit_hidden_size * int(1 / self.eagle_config.downsample_ratio) ** 2,
                    llm_hidden_size,
                ),
            )
        elif (
            self.eagle_config.mlp_connector_layers == 1
            and not self.eagle_config.use_pixel_shuffle
        ):
            self.mlp1 = nn.Sequential(
                nn.Linear(vit_hidden_size, llm_hidden_size),
            )
        else:
            raise NotImplementedError(
                f"{self.eagle_config.mlp_connector_layers} is not implemented."
            )

        self.num_cameras = len(policy.modality_config["video"].modality_keys)
        self.pixel_shuffle = policy.model.backbone.eagle_model.pixel_shuffle

        # Weight copy
        self.vision_model.load_state_dict(
            policy.model.backbone.eagle_model.vision_model.vision_model.state_dict(),
            strict=False,
        )
        self.mlp1.load_state_dict(policy.model.backbone.eagle_model.mlp1.state_dict())

        self.to(policy.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values
            [B*num_cameras, C, H, W] — preprocessed images.

        Returns
        -------
        torch.Tensor
            [B*num_cameras, num_patches, embed_dim]
        """
        vit_embeds = self.vision_model(pixel_values)

        if self.eagle_config.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.eagle_config.downsample_ratio
            )  # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], -1, vit_embeds.shape[-1]
            )  # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])

        return self.mlp1(vit_embeds)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        img_size = self.eagle_config.vision_config.image_size
        return {
            "pixel_values": TensorSpec(
                shape=(self.num_cameras, 3, img_size, img_size), dtype="float32"
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {"vit_embeds": TensorSpec()}

    def component_precision(self) -> Precision:
        return Precision.float


class _GrootCachedExportMixin:
    """Mixin that overrides convert_to_hub_source_model to route through
    BaseModel's version instead of LoadPolicyMixin's version (which tries to trace the model directly).

    Also propagates AIMET encodings through memory ops (Transpose, Reshape,
    etc.) during export, mirroring what LLM models do in _adapt_aimet_encodings.
    This ensures ops like Transpose that have is_output_quantized=False still
    carry encodings through to downstream consumers (e.g. MatMul on HTP).
    """

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        return Path(
            self.convert_to_onnx_and_aimet_encodings(
                Path(output_dir),
                self.__class__.__name__,
            )
        )

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        model_name: str | None = None,
        return_zip: bool = True,
    ) -> str:
        result = super().convert_to_onnx_and_aimet_encodings(  # type: ignore[misc]
            output_dir, model_name, return_zip=False
        )
        export_dir = Path(result)
        bundle = ONNXBundle.from_bundle_path(export_dir)
        apply_propagate_memory_encodings(bundle)
        if return_zip:
            model_name = model_name or self.__class__.__name__
            zip_path = Path(output_dir) / f"{model_name}.aimet.zip"
            base_dir = Path(f"{model_name}.aimet")
            data_path = bundle.onnx_weights_path
            zip_aimet_model(
                str(zip_path),
                base_dir,
                str(bundle.onnx_graph_path),
                str(bundle.aimet_encodings_path),
                str(data_path) if data_path else "",
            )
            return str(zip_path)
        return result

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        assert self._onnx_bundle is not None, "ONNX bundle not loaded"  # type: ignore[attr-defined]
        onnx_model = self._onnx_bundle.load_onnx_model()  # type: ignore[attr-defined]
        inputs, _ = extract_io_types_from_onnx_model(onnx_model)
        return {
            name: TensorSpec(shape=details.shape, dtype=str(details.dtype))
            for name, details in inputs.items()
        }

    def get_output_names(self) -> list[str]:
        assert self._onnx_bundle is not None, "ONNX bundle not loaded"  # type: ignore[attr-defined]
        onnx_model = self._onnx_bundle.load_onnx_model()  # type: ignore[attr-defined]
        _, outputs = extract_io_types_from_onnx_model(onnx_model)
        return list(outputs.keys())


@dataclass
class QuantSimNodeExceptions:
    """Per-node quantizer overrides applied after QuantSimOnnx creation"""

    # node name patterns (regex) whose input quantizers are disabled entirely
    disable_input: list[str] = field(default_factory=list)
    # node name patterns (regex) whose output quantizers are disabled entirely
    disable_output: list[str] = field(default_factory=list)
    # key: node name pattern (regex); value: param names to disable (e.g. ["weight", "bias"])
    disable_param: dict[str, list[str]] = field(default_factory=dict)

    # key: node name pattern (regex); value: {param name -> bitwidth} (e.g. {"weight": 16})
    set_param_bitwidth: dict[str, dict[str, int]] = field(default_factory=dict)
    # key: node name pattern (regex); value: bitwidth to apply to input quantizers
    set_input_bitwidth: dict[str, int] = field(default_factory=dict)
    # key: node name pattern (regex); value: bitwidth to apply to output quantizers
    set_output_bitwidth: dict[str, int] = field(default_factory=dict)


class GrootViTQuantizable(_GrootCachedExportMixin, AIMETOnnxQuantizableMixin, GrootViT):
    """GrootViT with AIMET-ONNX quantization support."""

    model_id: str = MODEL_ID
    model_asset_version: int = MODEL_ASSET_VERSION
    default_subfolder: str = "vit"
    _opts: dict[str, Any] = {}

    def __init__(
        self,
        sim_model: QuantSimOnnx | None = None,
        onnx_bundle: ONNXBundle | None = None,
        host_device: torch.device = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
    ) -> None:
        BaseModel.__init__(self, None)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        self.host_device = host_device
        self._precision = precision

    def make_quant_sim(self) -> QuantSimOnnx | None:
        if self._onnx_bundle is None:
            return None

        loading_from_checkpoint = self._onnx_bundle.aimet_encodings_path is not None
        onnx_model = self._onnx_bundle.load_onnx_model()
        if not loading_from_checkpoint:
            onnx_model, success = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])
            if not success:
                raise RuntimeError(
                    f"onnxsim simplification failed for {self.__class__.__name__}"
                )

        param_type, act_type = aimet_quant_types(self._precision)
        sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=param_type,
            activation_type=act_type,
            config_file=_GROOT_AIMET_CONFIG,
            providers=AIMETOnnxQuantizableMixin.get_ort_providers(self.host_device),
        )
        _apply_quantsim_exceptions(sim, act_bw=int(act_type.bits))

        if loading_from_checkpoint:
            load_encodings_to_sim(sim, str(self._onnx_bundle.aimet_encodings_path))

        return sim

    @classmethod
    def torch_from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> GrootViT:
        opts = adapt_torch_model_options or cls._opts
        policy = load_checkpoint(
            checkpoint=str(checkpoint),
            data_config=opts.get("data_config", DEFAULT_DATA_CONFIG),
            embodiment_tag=opts.get("embodiment_tag", DEFAULT_EMBODIMENT_TAG),
            device=str(host_device),
        )
        model = GrootViT(policy)

        return model.to(host_device).eval()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        **kwargs: Any,
    ) -> GrootViTQuantizable:
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        cls._opts = {
            "data_config": data_config,
            "embodiment_tag": embodiment_tag,
        }
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        return cls(
            None, onnx_bundle=bundle, host_device=host_device, precision=precision
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return cast(torch.Tensor, AIMETOnnxQuantizableMixin.forward(self, pixel_values))

    def component_precision(self) -> Precision:
        return self._precision


class GrootLLMBackbone(LoadGrootMixin, BaseModel):
    """
    Groot Eagle LLM backbone (Qwen3ForCausalLM, truncated to select_layer).
    Accepts preprocessed input_embeds and 4D attention mask.
    Returns vlm_features [B, seq_len, llm_hidden_size].
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        self.eagle_config = AutoConfig.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
        )
        self.vlm_seq_len = compute_vlm_seq_len(policy)

        backbone = EagleBackboneOpt(
            backbone_cfg=policy.model.config.backbone_cfg,
            vlm_seq_len=self.vlm_seq_len,
        )

        # copy weights
        backbone.load_state_dict(policy.model.backbone.state_dict(), strict=False)
        self.backbone: EagleBackboneOpt = backbone
        self.to(policy.device)

    def forward(
        self,
        input_embeds: torch.Tensor,
        llm_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_embeds
            [B, seq_len, llm_hidden_size] — merged token + visual embeddings.
        llm_attention_mask
            [B, 1, seq_len, seq_len] 4D additive causal mask.

        Returns
        -------
        torch.Tensor
            [B, seq_len, llm_hidden_size]
        """
        return self.backbone.forward(input_embeds, llm_attention_mask)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        llm_hidden_size = self.eagle_config.text_config.hidden_size
        return dict(
            input_embeds=TensorSpec(
                shape=(batch_size, self.vlm_seq_len, llm_hidden_size), dtype="float32"
            ),
            llm_attention_mask=TensorSpec(
                shape=(batch_size, 1, self.vlm_seq_len, self.vlm_seq_len),
                dtype="float32",
            ),
        )

    def get_output_spec(self) -> OutputSpec:
        return {"vlm_embeds": TensorSpec()}

    def component_precision(self) -> Precision:
        return Precision.float


class GrootLLMBackboneQuantizable(
    _GrootCachedExportMixin, AIMETOnnxQuantizableMixin, GrootLLMBackbone
):
    """GrootLLMBackbone with AIMET-ONNX quantization support."""

    model_id: str = MODEL_ID
    model_asset_version: int = MODEL_ASSET_VERSION
    default_subfolder: str = "llm"
    _opts: dict[str, Any] = {}

    def __init__(
        self,
        sim_model: QuantSimOnnx | None = None,
        onnx_bundle: ONNXBundle | None = None,
        host_device: torch.device = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
    ) -> None:
        BaseModel.__init__(self, None)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        self.host_device = host_device
        self._precision = precision

    def make_quant_sim(self) -> QuantSimOnnx | None:
        if self._onnx_bundle is None:
            return None

        loading_from_checkpoint = self._onnx_bundle.aimet_encodings_path is not None
        onnx_model = self._onnx_bundle.load_onnx_model()
        if not loading_from_checkpoint:
            onnx_model, removed_inits = onnx_remove_large_initializers(onnx_model)
            onnx_model, success = simplify(
                onnx_model, skipped_optimizers=["fuse_qkv", "eliminate_nop_with_unit"]
            )
            if not success:
                raise RuntimeError(
                    f"onnxsim simplification failed for {self.__class__.__name__}"
                )
            onnx_model = onnx_restore_large_initializers(onnx_model, removed_inits)
            apply_spinquant_onnx(onnx_model)

        param_type, act_type = aimet_quant_types(self._precision)
        sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=param_type,
            activation_type=act_type,
            config_file=_GROOT_AIMET_CONFIG,
            providers=AIMETOnnxQuantizableMixin.get_ort_providers(self.host_device),
        )
        _apply_quantsim_exceptions(
            sim,
            self.quantsim_node_exceptions(),
            act_bw=int(act_type.bits),
        )

        if loading_from_checkpoint:
            load_encodings_to_sim(sim, str(self._onnx_bundle.aimet_encodings_path))

        return sim

    def _apply_seq_mse(self, data: _DataLoader, num_batches: int) -> None:
        """Overload to pass 'nodes_to_exclude' to apply_seq_mse."""
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.8.0")

        # exclude disabled param quantizers
        exc = self.quantsim_node_exceptions()
        nodes_to_exclude = (
            _resolve_node_patterns(
                self.quant_sim,
                list(exc.disable_param.keys()),
            )
            or None
        )
        aimet_onnx.apply_seq_mse(
            self.quant_sim,
            self._dataloader_to_numpy(data, num_batches),
            nodes_to_exclude=nodes_to_exclude,
        )

    @classmethod
    def torch_from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> GrootLLMBackbone:
        opts = adapt_torch_model_options or cls._opts
        policy = load_checkpoint(
            checkpoint=str(checkpoint),
            data_config=opts.get("data_config", DEFAULT_DATA_CONFIG),
            embodiment_tag=opts.get("embodiment_tag", DEFAULT_EMBODIMENT_TAG),
            device=str(host_device),
        )
        model = GrootLLMBackbone(policy)

        return model.to(host_device).eval()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        **kwargs: Any,
    ) -> GrootLLMBackboneQuantizable:
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        cls._opts = {
            "data_config": data_config,
            "embodiment_tag": embodiment_tag,
        }
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        return cls(
            None, onnx_bundle=bundle, host_device=host_device, precision=precision
        )

    def forward(  # type: ignore[override]
        self, input_embeds: torch.Tensor, llm_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            AIMETOnnxQuantizableMixin.forward(self, input_embeds, llm_attention_mask),
        )

    def component_precision(self) -> Precision:
        return self._precision

    @staticmethod
    def quantsim_node_exceptions() -> QuantSimNodeExceptions:
        """Run below nodes in float due to dynamic range (from quant analyzer results)"""
        return QuantSimNodeExceptions(
            disable_output=[
                "/language_model/model/layers.2/mlp/Mul",
            ],
            disable_param={
                "/language_model/model/layers.2/mlp/down_proj/MatMul": ["weight"],
            },
        )


class GrootVLMProjection(LoadGrootMixin, BaseModel):
    """
    VLM Projection head for Groot N1.5.
    Combines VLLN + VL Self-Attention to produce cross-attention
    key/value pairs consumed by the DiT action head.

    Inputs:
      - vlm_embeds [B, seq_len, hidden_dim]
      - vlm_attention_mask [B, num_heads, seq_len, seq_len]

    Outputs:
      - vlm_proj_keys   [B, num_heads, seq_len, head_dim]
      - vlm_proj_values [B, num_heads, seq_len, head_dim]
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        action_head = policy.model.action_head

        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.dit = policy.model.action_head.model

        self.vlm_seq_len = compute_vlm_seq_len(policy)
        self.vlm_proj_num_heads = self.vl_self_attention.transformer_blocks[
            0
        ].attn1.heads
        self.n_dit_blocks = len(self.dit.transformer_blocks)

        self.eagle_config = AutoConfig.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
        )

        self.vl_self_attention.forward = MethodType(
            SelfAttentionTransformerOpt.forward, self.vl_self_attention
        )

        # Bypass attention mask prep in attn forward
        for block in self.vl_self_attention.transformer_blocks:
            block.attn1.prepare_attention_mask = MethodType(
                bypass_prepare_attention_mask, block.attn1
            )

    def forward(
        self,
        vlm_embeds: torch.Tensor,
        vlm_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        vlm_embeds
            [B, seq_len, hidden_dim] — VLM feature embeddings.
        vlm_attention_mask
            [B, H, seq_len, seq_len] — 4D bidirectional attention mask.

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        x = self.vlln(vlm_embeds)
        x = self.vl_self_attention(x, attention_mask=vlm_attention_mask)

        vlm_proj_keys, vlm_proj_values = [], []
        batch_size, *_ = x.shape

        for idx, block in enumerate(self.dit.transformer_blocks):
            if idx % 2 == 1 and self.dit.config.interleave_self_attention:
                pass
            else:
                key = block.attn1.to_k(x)
                value = block.attn1.to_v(x)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // block.attn1.heads

                key = key.view(batch_size, -1, block.attn1.heads, head_dim).transpose(
                    1, 2
                )
                value = value.view(
                    batch_size, -1, block.attn1.heads, head_dim
                ).transpose(1, 2)

                if block.attn1.norm_k is not None:
                    key = block.attn1.norm_k(key)

                vlm_proj_keys.append(key)
                vlm_proj_values.append(value)

        return (*vlm_proj_keys, *vlm_proj_values)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        llm_hidden_size = self.eagle_config.text_config.hidden_size
        return dict(
            vlm_embeds=TensorSpec(
                shape=(batch_size, self.vlm_seq_len, llm_hidden_size), dtype="float32"
            ),
            vlm_attention_mask=TensorSpec(
                shape=(
                    batch_size,
                    self.vlm_proj_num_heads,
                    self.vlm_seq_len,
                    self.vlm_seq_len,
                ),
                dtype="float32",
            ),
        )

    def get_output_spec(self) -> OutputSpec:
        n_blocks = 8  # known constant (8 cross-attn transformer_blocks)
        return {
            **{f"vlm_proj_keys_{i}": TensorSpec() for i in range(n_blocks)},
            **{f"vlm_proj_values_{i}": TensorSpec() for i in range(n_blocks)},
        }

    def component_precision(self) -> Precision:
        return Precision.float


class GrootVLMProjectionQuantizable(
    _GrootCachedExportMixin, AIMETOnnxQuantizableMixin, GrootVLMProjection
):
    """GrootVLMProjection with AIMET-ONNX quantization support."""

    model_id: str = MODEL_ID
    model_asset_version: int = MODEL_ASSET_VERSION
    default_subfolder: str = "vlm_proj"
    _opts: dict[str, Any] = {}

    def __init__(
        self,
        sim_model: QuantSimOnnx | None = None,
        onnx_bundle: ONNXBundle | None = None,
        host_device: torch.device = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
    ) -> None:
        BaseModel.__init__(self, None)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)
        self.host_device = host_device
        self._precision = precision

    def make_quant_sim(self) -> QuantSimOnnx | None:
        if self._onnx_bundle is None:
            return None

        loading_from_checkpoint = self._onnx_bundle.aimet_encodings_path is not None
        onnx_model = self._onnx_bundle.load_onnx_model()
        if not loading_from_checkpoint:
            onnx_model, success = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])
            if not success:
                raise RuntimeError(
                    f"onnxsim simplification failed for {self.__class__.__name__}"
                )

        param_type, act_type = aimet_quant_types(self._precision)
        sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=param_type,
            activation_type=act_type,
            config_file=_GROOT_AIMET_CONFIG,
            providers=AIMETOnnxQuantizableMixin.get_ort_providers(self.host_device),
        )
        _apply_quantsim_exceptions(sim, act_bw=int(act_type.bits))

        if loading_from_checkpoint:
            load_encodings_to_sim(sim, str(self._onnx_bundle.aimet_encodings_path))

        return sim

    @classmethod
    def torch_from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> GrootVLMProjection:
        opts = adapt_torch_model_options or cls._opts
        policy = load_checkpoint(
            checkpoint=str(checkpoint),
            data_config=opts.get("data_config", DEFAULT_DATA_CONFIG),
            embodiment_tag=opts.get("embodiment_tag", DEFAULT_EMBODIMENT_TAG),
            device=str(host_device),
        )
        return GrootVLMProjection(policy).to(host_device).eval()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        **kwargs: Any,
    ) -> GrootVLMProjectionQuantizable:
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        cls._opts = {
            "data_config": data_config,
            "embodiment_tag": embodiment_tag,
        }
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )
        return cls(
            None, onnx_bundle=bundle, host_device=host_device, precision=precision
        )

    def forward(  # type: ignore[override]
        self, vlm_embeds: torch.Tensor, vlm_attention_mask: torch.Tensor
    ) -> tuple:
        return cast(
            tuple,
            AIMETOnnxQuantizableMixin.forward(self, vlm_embeds, vlm_attention_mask),
        )

    def component_precision(self) -> Precision:
        return self._precision


class GrootDiT(LoadGrootMixin, BaseModel):
    """
    Diffusion Transformer (DiT) action head for Groot N1.5.
    Denoises noisy actions conditioned on VLM cross-attention keys/values,
    robot state, and a cross-attention mask.

    Inputs:
      - state               [B, 1, max_state_dim]
      - actions             [B, action_horizon, action_dim]
      - vlm_proj_keys       [B, num_heads, seq_len, head_dim]
      - vlm_proj_values     [B, num_heads, seq_len, head_dim]
      - cross_attention_mask [B, num_heads, sa_seq_len, vlm_seq_len]

    Output:
      - actions_out [B, action_horizon, action_dim]
    """

    def __init__(self, policy: Gr00tPolicy) -> None:
        super().__init__()

        action_head = ActionHeadDiTOpt(
            config=policy.model.action_head.config, embodiment_tag=policy.embodiment_tag
        )
        self.vlm_seq_len = compute_vlm_seq_len(policy)

        # copy weights
        action_head.load_state_dict(policy.model.action_head.state_dict(), strict=False)
        self.action_head: ActionHeadDiTOpt = action_head
        self.to(policy.device)

    def forward(
        self,
        state: torch.Tensor,  # [B, state_dim]
        actions: torch.Tensor,  # [B, action_horizon, action_dim]
        cross_attention_mask: torch.Tensor,  # [B, H, sa_seq_len, vlm_seq_len]
        vlm_proj_keys_0: torch.Tensor,
        vlm_proj_keys_1: torch.Tensor,
        vlm_proj_keys_2: torch.Tensor,
        vlm_proj_keys_3: torch.Tensor,
        vlm_proj_keys_4: torch.Tensor,
        vlm_proj_keys_5: torch.Tensor,
        vlm_proj_keys_6: torch.Tensor,
        vlm_proj_keys_7: torch.Tensor,
        vlm_proj_values_0: torch.Tensor,
        vlm_proj_values_1: torch.Tensor,
        vlm_proj_values_2: torch.Tensor,
        vlm_proj_values_3: torch.Tensor,
        vlm_proj_values_4: torch.Tensor,
        vlm_proj_values_5: torch.Tensor,
        vlm_proj_values_6: torch.Tensor,
        vlm_proj_values_7: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        state
            [B, 1, max_state_dim] — robot state tensor.
        actions
            [B, action_horizon, action_dim] — noisy actions for diffusion init.
        cross_attention_mask
            [B, H, sa_seq_len, vlm_seq_len] — cross-attention mask.
        vlm_proj_keys_0
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_1
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_2
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_3
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_4
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_5
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_6
            [B, num_heads, seq_len, head_dim]
        vlm_proj_keys_7
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_0
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_1
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_2
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_3
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_4
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_5
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_6
            [B, num_heads, seq_len, head_dim]
        vlm_proj_values_7
            [B, num_heads, seq_len, head_dim]

        Returns
        -------
        torch.Tensor
            [B, action_horizon, action_dim] denoised actions.
        """
        vlm_proj_keys = [
            vlm_proj_keys_0,
            vlm_proj_keys_1,
            vlm_proj_keys_2,
            vlm_proj_keys_3,
            vlm_proj_keys_4,
            vlm_proj_keys_5,
            vlm_proj_keys_6,
            vlm_proj_keys_7,
        ]
        vlm_proj_values = [
            vlm_proj_values_0,
            vlm_proj_values_1,
            vlm_proj_values_2,
            vlm_proj_values_3,
            vlm_proj_values_4,
            vlm_proj_values_5,
            vlm_proj_values_6,
            vlm_proj_values_7,
        ]

        return self.action_head(
            state=state,
            actions=actions,
            vlm_proj_keys=vlm_proj_keys,
            vlm_proj_values=vlm_proj_values,
            cross_attention_mask=cross_attention_mask,
        )

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        dit_num_heads = self.action_head.model.transformer_blocks[0].attn1.heads
        num_target_vision_tokens = self.action_head.config.num_target_vision_tokens
        num_heads = dit_num_heads
        head_dim = (
            self.action_head.model.transformer_blocks[0].attn1.inner_dim
            // dit_num_heads
        )
        spec: InputSpec = dict(
            state=TensorSpec(
                shape=(batch_size, 1, self.action_head.config.max_state_dim),
                dtype="float32",
            ),
            actions=TensorSpec(
                shape=(
                    batch_size,
                    self.action_head.action_horizon,
                    self.action_head.action_dim,
                ),
                dtype="float32",
            ),
            cross_attention_mask=TensorSpec(
                shape=(
                    batch_size,
                    dit_num_heads,
                    self.action_head.action_horizon + 1 + num_target_vision_tokens,
                    self.vlm_seq_len,
                ),
                dtype="float32",
            ),
        )
        n_blocks = len(self.action_head.model.transformer_blocks) // 2
        for i in range(n_blocks):
            spec[f"vlm_proj_keys_{i}"] = TensorSpec(
                shape=(batch_size, num_heads, self.vlm_seq_len, head_dim),
                dtype="float32",
            )
        for i in range(n_blocks):
            spec[f"vlm_proj_values_{i}"] = TensorSpec(
                shape=(batch_size, num_heads, self.vlm_seq_len, head_dim),
                dtype="float32",
            )
        return spec

    def get_output_spec(self) -> OutputSpec:
        return {"actions_out": TensorSpec()}

    def component_precision(self) -> Precision:
        return Precision.float


class GrootDiTQuantizable(_GrootCachedExportMixin, AIMETOnnxQuantizableMixin, GrootDiT):
    """GrootDiT with AIMET-ONNX quantization support."""

    model_id: str = MODEL_ID
    model_asset_version: int = MODEL_ASSET_VERSION
    default_subfolder: str = "dit"
    _opts: dict[str, Any] = {}

    def __init__(
        self,
        sim_model: QuantSimOnnx | None = None,
        onnx_bundle: ONNXBundle | None = None,
        host_device: torch.device = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
        **kwargs: Any,
    ) -> None:
        BaseModel.__init__(self, None)
        self.host_device = host_device
        self._precision = precision
        self._use_seq_mse: bool = kwargs.get("use_seq_mse", False)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model, onnx_bundle=onnx_bundle)

    def make_quant_sim(self) -> QuantSimOnnx | None:
        if self._onnx_bundle is None:
            return None

        loading_from_checkpoint = self._onnx_bundle.aimet_encodings_path is not None
        onnx_model = self._onnx_bundle.load_onnx_model()
        if not loading_from_checkpoint:
            opt_model = ort_optimize_model(
                onnx_model, opt_level=1, use_gpu=False, only_onnxruntime=True
            )
            onnx_model, success = simplify(
                opt_model.model, skipped_optimizers=["fuse_qkv"]
            )
            if not success:
                raise RuntimeError(
                    f"onnxsim simplification failed for {self.__class__.__name__}"
                )
            # Above opt. creates 3d weight initializers with batch size 1
            # Squeeze the batch dim for these initializers for compatibility with AIMET apis
            _squeeze_3d_initializers(onnx_model)
            if self._use_seq_mse:
                # Each unrolled loop iteration needs its own quantizer for SeqMSE per-node optimization.
                duplicate_shared_initializers(onnx_model.graph)

        param_type, act_type = aimet_quant_types(self._precision)
        sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.min_max,
            param_type=param_type,
            activation_type=act_type,
            config_file=_GROOT_AIMET_CONFIG,
            providers=AIMETOnnxQuantizableMixin.get_ort_providers(self.host_device),
        )
        _apply_quantsim_exceptions(
            sim, node_exceptions=self.quantsim_node_exceptions(), act_bw=act_type.bits
        )

        if loading_from_checkpoint:
            load_encodings_to_sim(sim, str(self._onnx_bundle.aimet_encodings_path))

        return sim

    @classmethod
    def torch_from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        adapt_torch_model_options: dict | None = None,
    ) -> GrootDiT:
        opts = adapt_torch_model_options or cls._opts
        policy = load_checkpoint(
            checkpoint=str(checkpoint),
            data_config=opts.get("data_config", DEFAULT_DATA_CONFIG),
            embodiment_tag=opts.get("embodiment_tag", DEFAULT_EMBODIMENT_TAG),
            device=str(host_device),
        )
        return GrootDiT(policy).to(host_device).eval()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
        precision: Precision = Precision.w8a16,
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        **kwargs: Any,
    ) -> GrootDiTQuantizable:
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        cls._opts = {
            "data_config": data_config,
            "embodiment_tag": embodiment_tag,
        }
        bundle = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )

        return cls(
            None,
            onnx_bundle=bundle,
            host_device=host_device,
            precision=precision,
            **kwargs,
        )

    def forward(self, *args: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return cast(torch.Tensor, AIMETOnnxQuantizableMixin.forward(self, *args))

    def component_precision(self) -> Precision:
        return self._precision

    @staticmethod
    def quantsim_node_exceptions() -> QuantSimNodeExceptions:
        """Run below params in INT16 due to dynamic range (from quant analyzer results)"""
        return QuantSimNodeExceptions(
            set_param_bitwidth={
                # state_encoder layer1
                "/action_head/state_encoder/layer1/MatMul": {"weight": 16},
                # action_decoder layer2
                # regex matches the 4 per-timestep unrolled layers
                "/action_head/action_decoder/layer2(_\d+)?/MatMul$": {"weight": 16},
            },
        )


### Preprocess methods
def preprocess_llm(
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    image_token_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merges vit_embeds into input_embeds and builds the 4D causal attention mask."""
    input_embeds = torch.nn.functional.embedding(
        input_ids, embedding_weight
    )  # [B, N, C]
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)

    # Replace image token positions with ViT embeddings
    selected = input_ids_flat == image_token_index
    input_embeds_flat[selected] = vit_embeds.reshape(-1, C)
    input_embeds = input_embeds_flat.reshape(B, N, C)

    # Build 4D causal attention mask
    llm_attention_mask = prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=N,
        target_length=N,
        dtype=input_embeds.dtype,
        device=input_embeds.device,
        cache_position=torch.arange(N, device=input_embeds.device),
        batch_size=B,
    )
    return input_embeds, llm_attention_mask


def preprocess_vlm_proj(
    attention_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Builds the 4D bidirectional attention mask, repeated for each attention head."""
    vlm_attention_mask = prepare_4d_bidirectional_mask(attention_mask)
    return vlm_attention_mask.repeat_interleave(num_heads, dim=1)


def preprocess_dit(
    attention_mask: torch.Tensor,
    num_heads: int,
    action_horizon: int,
    num_target_vision_tokens: int,
) -> torch.Tensor:
    """Builds the 4D bidirectional mask for DiT cross-attention, repeated per head."""
    cross_attention_mask = prepare_4d_bidirectional_mask(
        attention_mask,
        target_len=action_horizon
        + 1
        + num_target_vision_tokens,  # actions + state token + future tokens
    )
    return cross_attention_mask.repeat_interleave(num_heads, dim=1)


### Utils
def compute_vlm_seq_len(policy: Gr00tPolicy) -> int:
    """
    Compute the padded VLM input sequence length from a Gr00tPolicy instance.
    The sequence is laid out as:
        [language tokens] + [vision tokens per camera x num_cameras]
    """
    eagle_cfg = AutoConfig.from_pretrained(
        DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
    )
    num_cameras = len(policy.modality_config["video"].modality_keys)
    raw_patches = (
        eagle_cfg.vision_config.image_size // eagle_cfg.vision_config.patch_size
    ) ** 2
    if getattr(eagle_cfg, "use_pixel_shuffle", False):
        vit_tokens = int(raw_patches * eagle_cfg.downsample_ratio**2)
    else:
        vit_tokens = raw_patches

    return VLM_LANGUAGE_TOKENS + num_cameras * vit_tokens


def _apply_quantsim_exceptions(
    quant_sim: QuantSimOnnx,
    node_exceptions: QuantSimNodeExceptions | None = None,
    act_bw: int = 16,
) -> None:
    """
    Utility to apply quantization exceptions to a QuantSimOnnx model.
    Supports regex patterns for node names.
    Raises ValueError if any pattern fails to compile or if a pattern matches no node
    """
    compiled: dict[str, re.Pattern] = {}
    match_counts: dict[str, int] = {}

    if node_exceptions is not None:
        all_patterns: list[str] = (
            node_exceptions.disable_input
            + node_exceptions.disable_output
            + list(node_exceptions.disable_param.keys())
            + list(node_exceptions.set_param_bitwidth.keys())
            + list(node_exceptions.set_input_bitwidth.keys())
            + list(node_exceptions.set_output_bitwidth.keys())
        )
        compiled = {p: re.compile(p) for p in all_patterns}
        match_counts = dict.fromkeys(all_patterns, 0)

    def _matching_patterns(op_name: str, patterns: list[str]) -> list[str]:
        return [p for p in patterns if compiled[p].search(op_name)]

    for op in quant_sim.connected_graph.get_all_ops().values():
        input_qs, output_qs, param_qs = quant_sim.get_op_quantizers(op)
        name = op.name_op

        # RMSNorm / LayerNorm weights -> act_bw symmetric
        if op.type in (
            "LayerNormalization",
            "LayerNorm",
            "RMSNormalization",
            "RmsNorm",
        ):
            q = param_qs.get("weight")
            if q is not None and q.enabled:
                q.bitwidth = act_bw
                q.use_symmetric_encodings = True

        if node_exceptions is None:
            continue

        for pat in _matching_patterns(name, node_exceptions.disable_input):
            match_counts[pat] += 1
            for q in input_qs:
                if q is not None:
                    q.enabled = False

        for pat in _matching_patterns(name, node_exceptions.disable_output):
            match_counts[pat] += 1
            for q in output_qs:
                if q is not None:
                    q.enabled = False

        for pat in _matching_patterns(name, list(node_exceptions.disable_param.keys())):
            match_counts[pat] += 1
            for key in node_exceptions.disable_param[pat]:
                if key not in param_qs:
                    raise ValueError(
                        f"disable_param: param key '{key}' "
                        f"not found in op '{name}' (matched by pattern '{pat}'). "
                        f"Available params: {list(param_qs.keys())}"
                    )
                if param_qs[key] is not None:
                    param_qs[key].enabled = False

        for pat in _matching_patterns(
            name, list(node_exceptions.set_param_bitwidth.keys())
        ):
            match_counts[pat] += 1
            for key, bw in node_exceptions.set_param_bitwidth[pat].items():
                if key not in param_qs:
                    raise ValueError(
                        f"set_param_bitwidth: param key '{key}' "
                        f"not found in op '{name}' (matched by pattern '{pat}'). "
                        f"Available params: {list(param_qs.keys())}"
                    )
                if param_qs[key] is not None:
                    param_qs[key].bitwidth = bw

        for pat in _matching_patterns(
            name, list(node_exceptions.set_input_bitwidth.keys())
        ):
            match_counts[pat] += 1
            bw = node_exceptions.set_input_bitwidth[pat]
            for q in input_qs:
                if q is not None:
                    q.bitwidth = bw

        for pat in _matching_patterns(
            name, list(node_exceptions.set_output_bitwidth.keys())
        ):
            match_counts[pat] += 1
            bw = node_exceptions.set_output_bitwidth[pat]
            for q in output_qs:
                if q is not None:
                    q.bitwidth = bw

    if node_exceptions is not None:
        unmatched = [p for p, count in match_counts.items() if count == 0]
        if unmatched:
            raise ValueError(
                "The following patterns matched no ops "
                "in the connected graph - check for typos or stale node names:\n"
                + "\n".join(f"  - '{p}'" for p in unmatched)
            )

    # Handle any decomposed norm weights -> act_bw symmetric
    for init in quant_sim.model.model.graph.initializer:
        if "norm.weight" in init.name:
            q = quant_sim.qc_quantize_op_dict.get(init.name)
            if q is not None and q.enabled:
                q.bitwidth = act_bw
                q.use_symmetric_encodings = True


def _resolve_node_patterns(
    quant_sim: QuantSimOnnx,
    patterns: list[str],
) -> list[str]:
    """Resolve regex patterns against connected graph op names."""
    compiled = {p: re.compile(p) for p in patterns}
    matched: list[str] = []
    seen: set[str] = set()
    for op in quant_sim.connected_graph.get_all_ops().values():
        name = op.name_op
        for rx in compiled.values():
            if rx.search(name) and name not in seen:
                matched.append(name)
                seen.add(name)
                break
    return matched


def _squeeze_3d_initializers(model: onnx.ModelProto) -> None:
    """Squeeze (1, in, out) initializers that feed MatMul/Gemm nodes down to (in, out)."""
    weight_inputs: set[str] = set()
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
            weight_inputs.add(node.input[1])

    for init in model.graph.initializer:
        if init.name not in weight_inputs:
            continue
        dims = list(init.dims)
        if len(dims) == 3 and dims[0] == 1:
            init.dims[:] = dims[1:]


class _GrootEvalMixin:
    """Shared evaluator and dataset wiring for GrootCollection and GrootQuantizedCollection."""

    def get_evaluator(self) -> BaseEvaluator:
        from qai_hub_models.models.grootn15.dataset import GrootDataset

        return LeRobotEvaluator(dof_slices=GrootDataset.get_dof_slices())

    @classmethod
    def get_eval_dataset_classes(cls) -> list:
        from qai_hub_models.models.grootn15.dataset import GrootDataset

        return [GrootDataset]


# -----------------------------------------------------------------
# GrootCollection — float collection
# -----------------------------------------------------------------
class GrootCollection(_GrootEvalMixin, WorkbenchModelCollection):
    """
    CollectionModel for Groot N1.5.

    Components:
      - vit       : Eagle vision encoder (ViT)
      - llm       : Eagle LLM backbone (produces VLM embeddings)
      - vlm_proj  : VLM projection (keys/values for DiT VLM cross-attention)
      - dit       : DiT action head (denoising diffusion transformer)

    Usage::
        model = GrootCollection.from_pretrained()
        model.components["vit"], ["llm"], ["vlm_proj"], ["dit"]
    """

    # Set by from_pretrained so from_components can reuse the same policy.
    _last_checkpoint: str = "DEFAULT"
    _last_data_config: str = DEFAULT_DATA_CONFIG
    _last_embodiment_tag: str = DEFAULT_EMBODIMENT_TAG
    _last_host_device: str = "cpu"

    def __init__(
        self,
        vit: GrootViT,
        llm: GrootLLMBackbone,
        vlm_proj: GrootVLMProjection,
        dit: GrootDiT,
    ) -> None:
        super().__init__({"vit": vit, "llm": llm, "vlm_proj": vlm_proj, "dit": dit})

    @classmethod
    def from_policy(
        cls,
        policy: Gr00tPolicy,
        kwargs: dict[str, Any] | None = None,
    ) -> GrootCollection:
        """Build a GrootCollection from an already-loaded policy."""
        return cls(
            GrootViT(policy),
            GrootLLMBackbone(policy),
            GrootVLMProjection(policy),
            GrootDiT(policy),
        )

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        precision: Precision = Precision.float,
        **kwargs: object,
    ) -> GrootCollection | GrootQuantizedCollection:
        cls._last_checkpoint = str(checkpoint)
        cls._last_data_config = data_config
        cls._last_embodiment_tag = embodiment_tag
        cls._last_host_device = str(host_device)

        if precision != Precision.float:
            return GrootQuantizedCollection.from_pretrained(
                checkpoint=checkpoint,
                host_device=host_device,
                data_config=data_config,
                embodiment_tag=embodiment_tag,
            )

        policy = load_checkpoint(
            checkpoint=str(checkpoint),
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            device=str(host_device),
        )
        return cls.from_policy(policy, kwargs=kwargs)


# -----------------------------------------------------------------
# GrootQuantizedCollection — loads pre-quantized ONNX checkpoints
# -----------------------------------------------------------------
class GrootQuantizedCollection(_GrootEvalMixin, WorkbenchModelCollection):
    """
    CollectionModel for quantized Groot N1.5.

    Loads each component from a per-component ONNX+encodings checkpoint
    written by quantize.py. Each component's precision is baked into its
    saved checkpoint; from_pretrained restores them.
    """

    component_class_names: list[str] = ["vit", "llm", "vlm_proj", "dit"]
    component_classes: dict[str, type] = {
        "vit": GrootViTQuantizable,
        "llm": GrootLLMBackboneQuantizable,
        "vlm_proj": GrootVLMProjectionQuantizable,
        "dit": GrootDiTQuantizable,
    }

    def __init__(
        self,
        vit: GrootViTQuantizable,
        llm: GrootLLMBackboneQuantizable,
        vlm_proj: GrootVLMProjectionQuantizable,
        dit: GrootDiTQuantizable,
    ) -> None:
        super().__init__({"vit": vit, "llm": llm, "vlm_proj": vlm_proj, "dit": dit})

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        **kwargs: object,
    ) -> GrootQuantizedCollection:
        from qai_hub_models.models.grootn15.quantize import QUANT_PRECISION

        return cls(
            GrootViTQuantizable.from_pretrained(
                checkpoint=checkpoint,
                host_device=host_device,
                data_config=data_config,
                embodiment_tag=embodiment_tag,
                precision=QUANT_PRECISION["vit"],
            ),
            GrootLLMBackboneQuantizable.from_pretrained(
                checkpoint=checkpoint,
                host_device=host_device,
                data_config=data_config,
                embodiment_tag=embodiment_tag,
                precision=QUANT_PRECISION["llm"],
            ),
            GrootVLMProjectionQuantizable.from_pretrained(
                checkpoint=checkpoint,
                host_device=host_device,
                data_config=data_config,
                embodiment_tag=embodiment_tag,
                precision=QUANT_PRECISION["vlm_proj"],
            ),
            GrootDiTQuantizable.from_pretrained(
                checkpoint=checkpoint,
                host_device=host_device,
                data_config=data_config,
                embodiment_tag=embodiment_tag,
                precision=QUANT_PRECISION["dit"],
            ),
        )
