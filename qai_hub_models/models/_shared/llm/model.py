# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import onnx

try:
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

    from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin
except (ImportError, ModuleNotFoundError):
    raise NotImplementedError(
        "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Install qai-hub-models on a Linux machine to use quantized models."
    )

import gc
import glob
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import aimet_common.quantsim as qs
import torch
from aimet_common.defs import QuantScheme
from aimet_onnx import quantsim
from aimet_onnx.quantsim import load_encodings_to_sim
from onnx.external_data_helper import load_external_data_for_model
from qai_hub.client import Device
from transformers import AutoConfig
from transformers.models.llama import LlamaConfig

from qai_hub_models.models._shared.llm._utils import (
    _set_lm_head_to_8b,
    _set_tensors_to_output_8b_sym,
)
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.huggingface import ensure_has_required_transformer
from qai_hub_models.utils.input_spec import InputSpec

MIN_TRANFORMER_VERSION = "4.45.0"

# isort: off

DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_CONTEXT_LENGTH = 4096

# TODO: 10761 remove transformer version check once AIMET
# transformer restriction is uplifted.
ensure_has_required_transformer(MIN_TRANFORMER_VERSION)
from transformers import (  # noqa: E402
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


def get_tokenizer(model_ckpt: str | None) -> PreTrainedTokenizerBase:
    """
    Tokenizer to use for LLMs
    """
    assert model_ckpt is not None

    print()
    print(f"Loading tokenizer from {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, is_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    return tokenizer


def get_llm_config(
    model_ckpt: str | os.PathLike | None, _make_small_for_debugging: bool = False
) -> LlamaConfig:
    """
    Construct and return a HuggingFace LLM config.
    """

    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    if _make_small_for_debugging:
        llm_config.num_hidden_layers = 8
        llm_config.num_attention_heads = 4
        llm_config.num_key_value_heads = 2
        llm_config.vocab_size = 13
        embed_dim = 8
        llm_config.head_dim = embed_dim * 2
        llm_config.hidden_size = llm_config.num_attention_heads * embed_dim * 2
    llm_config._attn_implementation = "eager"
    llm_config._attn_implementation_internal = "eager"

    # Force use_cache=true for all LLMs
    llm_config.use_cache = True

    return llm_config


def get_onnx_model(
    fp_model: torch.nn.Module,
    context_length: int,
    sequence_length: int,
    path: str,
    server_device: torch.device,
    return_model: bool = False,
) -> onnx.ModelProto | None:
    # Create the checkpoint directory if it does not exist.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fp_model.to(server_device)
    input_specs = fp_model.get_input_spec(
        context_length=context_length,
        sequence_length=sequence_length,
    )
    print()
    print(
        f"Exporting ONNX model with sequence length {sequence_length} and context length {context_length}. This would take around 10 minutes."
    )

    example_input = [
        torch.zeros(
            input_specs[name][0], dtype=getattr(torch, input_specs[name][1])
        ).to(server_device)
        for name in input_specs.keys()
    ]
    torch.onnx.export(
        fp_model,
        tuple(example_input),
        path,
        input_names=list(input_specs.keys()),
        output_names=fp_model.get_output_names(),
        opset_version=17,
    )

    # Clean up floating point model
    del example_input
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    onnx_model = onnx.load(path)
    # Clean up multiple weights files
    for file in glob.glob(os.path.join(os.path.dirname(path), "*.weight")):
        os.remove(file)
    for file in glob.glob(os.path.join(os.path.dirname(path), "onnx__*")):
        os.remove(file)

    onnx.save_model(
        onnx_model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
    )

    load_external_data_for_model(onnx_model, os.path.dirname(path))
    if not return_model:
        del onnx_model
        gc.collect()
    return onnx_model if return_model else None


class LLM_AIMETOnnx(AIMETOnnxQuantizableMixin, BaseModel, ABC):
    def __init__(
        self,
        sim_model: QuantSimOnnx,
        onnx_model: onnx.ModelProto,
        sequence_length: int,
        context_length: int,
        huggingface_model_name: str | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        server_device: torch.device | None = None,
    ):
        BaseModel.__init__(self)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.onnx_model = onnx_model
        self.server_device = server_device

        self.tokenizer = get_tokenizer(huggingface_model_name)

        if checkpoint is not None:
            try:
                self.llm_config = get_llm_config(checkpoint)
            except Exception:
                print()
                print(f"No model config in {checkpoint}")
            # Needed for default grab-and-go
            finally:
                self.llm_config = get_llm_config(huggingface_model_name)

        else:
            self.tokenizer = get_tokenizer(huggingface_model_name)
            self.llm_config = get_llm_config(huggingface_model_name)

        self.checkpoint = checkpoint

    @classmethod
    def from_pretrained(
        cls,
        server_device: torch.device,
        sequence_length: int,
        context_length: int,
        fp_model: torch.nn.Module | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
    ) -> LLM_AIMETOnnx:
        """
        Load weight from local checkpoint of Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Args:

        - server_device: Device to use: GPU/CPU
        - sequence_length: Sequence Length for the model
        - context_length: Context Length for the model
        - fp_model: Floating point version of this model.
        This is quantized as part of this class and QuantSim model is created.
        - checkpoint: Path to previously calibrated AIMET encodings and
        ONNX models. Note that encodings are sensitive to AIMET ONNX versions
        because loading back the


        """
        if server_device is None:
            server_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Two copies are needed. One for QuantSim and one for passing to
        # quantize function for applying Sequencial MSE.
        # Deepcopy causes error on GPU.
        onnx_path = None
        onnx_file_exists = False
        tmp_dir = tempfile.TemporaryDirectory()
        onnx_tmpfile = os.path.join(tmp_dir.name, "model.onnx")

        if checkpoint is not None:
            onnx_path = os.path.join(
                checkpoint, f"model_seqlen{sequence_length}_cl{context_length}.onnx"
            )
            onnx_file_exists = os.path.exists(onnx_path) and os.path.exists(
                os.path.join(checkpoint, "model.data")
            )

        # If the file and external weights exist in the checkpoint, we should not export again.
        if not onnx_file_exists:
            # Floating model is created if not passed when from_pretrained() is called and an ONNX model doesn't exist.
            if fp_model is None:
                fp_model = cls.create_torch_model(
                    context_length, sequence_length, checkpoint, server_device
                )

            onnx_model = get_onnx_model(
                fp_model=fp_model,
                context_length=context_length,
                sequence_length=sequence_length,
                path=onnx_path or onnx_tmpfile,
                server_device=server_device,
                return_model=True,
            )
            del fp_model
            gc.collect()
            qs_onnx_model = onnx.load(onnx_path or onnx_tmpfile)
        else:
            print()
            print(f"Loading onnx model from {onnx_path}")
            assert onnx_path is not None
            onnx_model, qs_onnx_model = onnx.load(onnx_path), onnx.load(onnx_path)

        if onnx_path is None:
            tmp_dir.cleanup()

        print()
        print("Creating a QuantSim model using AIMET ONNX.")
        quant_sim = cls.create_quantsim(qs_onnx_model, server_device)

        # Cleanup the ONNX model that creates the QuantSim model
        del qs_onnx_model
        gc.collect()

        # Encodings are not produced yet.
        if checkpoint is not None:
            aimet_encodings = os.path.join(checkpoint, "model.encodings")
            if os.path.exists(aimet_encodings):
                print()
                print(
                    f"Loading the encodings from path {checkpoint} to load the QuantSim model."
                )
                load_encodings_to_sim(quant_sim, aimet_encodings)
                export_sequence_lengths = list(
                    {
                        1,
                        sequence_length,
                        context_length // 2,
                        DEFAULT_CONTEXT_LENGTH // 2,
                        DEFAULT_SEQUENCE_LENGTH,
                    }
                )
                cls.create_onnx_models(
                    checkpoint=checkpoint,
                    context_length=context_length,
                    export_sequence_lengths=export_sequence_lengths,
                    server_device=server_device,
                )

        assert onnx_model is not None
        return cls(
            sim_model=quant_sim,
            onnx_model=onnx_model,
            sequence_length=sequence_length,
            context_length=context_length,
            server_device=server_device,
            checkpoint=checkpoint,
        )

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        pass

    @classmethod
    def create_quantsim(
        cls, onnx_model: onnx.ModelProto, server_device: torch.device
    ) -> QuantSimOnnx:
        """
        onnx_model: ONNX Model to create QuantSim model.
        server_device: Device that the QuantSim model must be placed on.
        """
        default_config = get_aimet_config_path("default_config_llama")
        # Tie Quantizers for Concat Op
        quantsim.op_types_to_tie_qtzrs = ["Concat"]
        quantsim._tie_qtzrs = True
        # Ignore Slice outputs
        quantsim.op_outputs_to_ignore.append("Slice")
        qs.encoding_version = "0.6.1"
        quant_sim = QuantSimOnnx(
            model=onnx_model,
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=4,
            config_file=default_config,
            use_cuda=(server_device.type == "cuda"),
        )
        # Setting kv_cache and some other layers to 8-bit
        _set_tensors_to_output_8b_sym(quant_sim)
        # Setting the LM head weights to 8-bit.
        _set_lm_head_to_8b(quant_sim)
        return quant_sim

    @classmethod
    @abstractmethod
    def create_torch_model(
        cls,
        context_length: int,
        sequence_length: int,
        input_checkpoint: str | os.PathLike | Path | None,
        device: torch.device,
    ) -> torch.nn.Module:
        pass

    def save_calibrated_checkpoint(
        self,
        output_checkpoint: str | os.PathLike | Path,
        input_checkpoint: str | os.PathLike | Path,
    ) -> None:
        """
        output_checkpoint: Path to the directory which must store the checkpoint.
        It would contain the encodings file, external data file and multiple ONNX
        models that will be needed by the user.
        """
        # Make the directory for the output checkpoint
        os.makedirs(output_checkpoint, exist_ok=True)
        export_sequence_lengths = [1, self.context_length // 2]
        # If the sequence length is ARs to be exported then export model as part of QuantSim.
        print(f"Creating a checkpoint of quantized model at {output_checkpoint}.")
        self.quant_sim.export(str(output_checkpoint), "model")
        del self.quant_sim
        # Save ONNX model and data file in the checkpoint.
        shutil.copy(
            os.path.join(output_checkpoint, "model.onnx"),
            os.path.join(
                output_checkpoint,
                f"model_seqlen{self.sequence_length}_cl{self.context_length}.onnx",
            ),
        )
        # Create the multiple ONNX models.
        self.create_onnx_models(
            checkpoint=output_checkpoint,
            context_length=self.context_length,
            export_sequence_lengths=export_sequence_lengths,
            input_checkpoint=input_checkpoint,
            server_device=self.server_device,
        )

    @classmethod
    def create_onnx_models(
        cls,
        checkpoint: str | os.PathLike | Path,
        context_length: int,
        export_sequence_lengths: list[int],
        input_checkpoint: str | os.PathLike | Path | None = None,
        server_device: torch.device = torch.device("cpu"),
    ) -> None:
        external_weights_file = os.path.join(checkpoint, "model.data")
        onnx_file = os.path.join(checkpoint, "model.onnx")
        # Make floating point model
        for seq_len in export_sequence_lengths:
            expected_onnx_model = os.path.join(
                checkpoint, f"model_seqlen{seq_len}_cl{context_length}.onnx"
            )
            if not os.path.exists(expected_onnx_model) or not os.path.exists(
                external_weights_file
            ):
                # Export to ONNX for any sequence length needed.
                # The external weights is made multiple times but is overwritten each
                # time so only one copy is ther at a given time.
                fp_model = cls.create_torch_model(
                    context_length, seq_len, input_checkpoint, server_device
                )
                get_onnx_model(
                    fp_model=fp_model,
                    context_length=context_length,
                    sequence_length=seq_len,
                    path=onnx_file,
                    server_device=server_device,
                )
                # Rename the model per sequence_length
                shutil.move(
                    onnx_file,
                    expected_onnx_model,
                )

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | os.PathLike | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
        external_weights: bool = False,
        bundle_external_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        if model_name is None:
            model_name = self.__class__.__name__

        base_path = os.path.join(output_dir, f"{model_name}.aimet")
        os.makedirs(base_path, exist_ok=True)
        assert self.checkpoint is not None

        src_onnx_filepath = os.path.join(
            self.checkpoint,
            f"model_seqlen{self.sequence_length}_cl{self.context_length}.onnx",
        )
        src_external_weights_filepath = os.path.join(self.checkpoint, "model.data")
        src_encodings_filepath = os.path.join(self.checkpoint, "model.encodings")

        dst_onnx_filepath = os.path.join(base_path, "model.onnx")
        dst_external_weights_filepath = os.path.join(base_path, "model.data")
        dst_encodings_filepath = os.path.join(base_path, "model.encodings")

        shutil.copy(src_onnx_filepath, dst_onnx_filepath)
        shutil.copy(src_external_weights_filepath, dst_external_weights_filepath)

        self._adapt_aimet_encodings(
            src_encodings_filepath, dst_encodings_filepath, dst_onnx_filepath
        )

        return base_path

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision = Precision.w8a16,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        assert precision == Precision.w8a16, "Only w8a16 precision is supported"
        if (
            target_runtime != TargetRuntime.QNN
            and target_runtime != TargetRuntime.PRECOMPILED_QNN_ONNX
        ):
            raise RuntimeError(
                f"Unsupported target_runtime provided: {target_runtime}."
                " Only Precompile ONN ONNX or QNN runtime is supported for Llama for now."
            )
        compile_options += (
            " --target_runtime qnn_context_binary"
            if target_runtime == TargetRuntime.QNN
            else " --target_runtime precompiled_qnn_onnx"
        )
        return compile_options

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        profile_options += " --max_profiler_iterations 50"
        return profile_options
