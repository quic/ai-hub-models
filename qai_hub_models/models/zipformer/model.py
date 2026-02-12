# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import sys
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from qai_hub.client import Device
from ruamel.yaml import YAML
from torch import Tensor, nn

from qai_hub_models.models.zipformer.model_adaption import (
    Modify_EncoderModule,
)
from qai_hub_models.utils.asset_loaders import SourceAsRoot
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    Precision,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

if TYPE_CHECKING:
    import Decoder
    import Joiner
    import Zipformer


MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class ZipformerEncoder(BaseModel):
    def __init__(self, encoder: "Zipformer", encoder_proj: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_proj = encoder_proj

    def forward(self, *args: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """
        Parameters
        ----------
        *args
            A list of input and state tensors

        Returns
        -------
        output : Tensor
            a 3-D tensor of shape (N, T', joiner_dim)
        new_states : tuple[Tensor, ...]
            list of updated state tensors
        """
        x = args[0]
        states = args[1:]
        output, _, new_states = self.encoder.streaming_forward(
            x=x, x_lens=torch.tensor([71]), states=states
        )
        output = self.encoder_proj(output)
        return output, new_states

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "x": ((1, 71, 80), "float32"),
            "cached_len_0": ((2, 1), "int32"),
            "cached_len_1": ((4, 1), "int32"),
            "cached_len_2": ((3, 1), "int32"),
            "cached_len_3": ((2, 1), "int32"),
            "cached_len_4": ((4, 1), "int32"),
            "cached_avg_0": ((2, 1, 384), "float32"),
            "cached_avg_1": ((4, 1, 384), "float32"),
            "cached_avg_2": ((3, 1, 384), "float32"),
            "cached_avg_3": ((2, 1, 384), "float32"),
            "cached_avg_4": ((4, 1, 384), "float32"),
            "cached_key_0": ((2, 128, 1, 192), "float32"),
            "cached_key_1": ((4, 64, 1, 192), "float32"),
            "cached_key_2": ((3, 32, 1, 192), "float32"),
            "cached_key_3": ((2, 16, 1, 192), "float32"),
            "cached_key_4": ((4, 64, 1, 192), "float32"),
            "cached_val_0": ((2, 128, 1, 96), "float32"),
            "cached_val_1": ((4, 64, 1, 96), "float32"),
            "cached_val_2": ((3, 32, 1, 96), "float32"),
            "cached_val_3": ((2, 16, 1, 96), "float32"),
            "cached_val_4": ((4, 64, 1, 96), "float32"),
            "cached_val2_0": ((2, 128, 1, 96), "float32"),
            "cached_val2_1": ((4, 64, 1, 96), "float32"),
            "cached_val2_2": ((3, 32, 1, 96), "float32"),
            "cached_val2_3": ((2, 16, 1, 96), "float32"),
            "cached_val2_4": ((4, 64, 1, 96), "float32"),
            "cached_conv1_0": ((2, 1, 384, 30), "float32"),
            "cached_conv1_1": ((4, 1, 384, 30), "float32"),
            "cached_conv1_2": ((3, 1, 384, 30), "float32"),
            "cached_conv1_3": ((2, 1, 384, 30), "float32"),
            "cached_conv1_4": ((4, 1, 384, 30), "float32"),
            "cached_conv2_0": ((2, 1, 384, 30), "float32"),
            "cached_conv2_1": ((4, 1, 384, 30), "float32"),
            "cached_conv2_2": ((3, 1, 384, 30), "float32"),
            "cached_conv2_3": ((2, 1, 384, 30), "float32"),
            "cached_conv2_4": ((4, 1, 384, 30), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return (
            ["encoder_out"]
            + [f"new_cached_len_{i}" for i in range(5)]
            + [f"new_cached_avg_{i}" for i in range(5)]
            + [f"new_cached_key_{i}" for i in range(5)]
            + [f"new_cached_val_{i}" for i in range(5)]
            + [f"new_cached_val2_{i}" for i in range(5)]
            + [f"new_cached_conv1_{i}" for i in range(5)]
            + [f"new_cached_conv2_{i}" for i in range(5)]
        )

    @classmethod
    def from_pretrained(cls) -> "ZipformerEncoder":
        hf_zipformer = HfZipformer.from_pretrained()

        return cls(
            hf_zipformer.encoder.encoder,
            hf_zipformer.encoder.encoder_proj,
        )

    @staticmethod
    def calibration_dataset_name() -> str:
        return "common_voice"

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--compute_unit" not in profile_options
        ):
            profile_options = profile_options + " --compute_unit gpu"
        return profile_options + " --max_profiler_iterations 10"

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="encoder_model",
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors --truncate_64bit_io "
        return compile_options


class ZipformerDecoder(BaseModel):
    def __init__(self, decoder: "Decoder", decoder_proj: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y
            the decoder output id of previous time step
            A 2-D tensor of shape (N, context_size)

        Returns
        -------
        decoder_output : Tensor
            a 2-D tensor of shape (N, joiner_dim)
        """
        decoder_output = self.decoder(y, need_pad=False)
        decoder_output = decoder_output.squeeze(1)
        return self.decoder_proj(decoder_output)

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return dict(
            y=((1, 2), "int32"),
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return ["decoder_out"]

    @classmethod
    def from_pretrained(cls) -> "ZipformerDecoder":
        hf_zipformer = HfZipformer.from_pretrained()
        return cls(
            hf_zipformer.decoder.decoder,
            hf_zipformer.decoder.decoder_proj,
        )

    @staticmethod
    def calibration_dataset_name() -> str:
        return "common_voice"

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="decoder_model",
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors --truncate_64bit_io "
        return compile_options


class ZipformerJoiner(BaseModel):
    def __init__(self, output_linear: nn.Module) -> None:
        super().__init__()
        self.output_linear = output_linear

    def forward(self, encoder_out: Tensor, decoder_out: Tensor) -> Tensor:
        """
        Parameters
        ----------
        encoder_out
            A 2-D tensor of shape (N, joiner_dim)
        decoder_out
            A 2-D tensor of shape (N, joiner_dim)

        Returns
        -------
        logit : Tensor
            a 2-D tensor of shape (N, vocab_size)
        """
        logit = encoder_out + decoder_out
        return self.output_linear(torch.tanh(logit))

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return dict(
            encoder_out=((1, 512), "float32"), decoder_out=((1, 512), "float32")
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return ["logit"]

    @classmethod
    def from_pretrained(cls) -> "ZipformerJoiner":
        hf_zipformer = HfZipformer.from_pretrained()
        return cls(hf_zipformer.joiner.output_linear)

    @staticmethod
    def calibration_dataset_name() -> str:
        return "common_voice"

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="joiner_model",
        )


@CollectionModel.add_component(ZipformerEncoder)
@CollectionModel.add_component(ZipformerDecoder)
@CollectionModel.add_component(ZipformerJoiner)
class HfZipformer(CollectionModel):
    def __init__(
        self, encoder: "Zipformer", decoder: "Decoder", joiner: "Joiner"
    ) -> None:
        super().__init__(encoder, decoder, joiner)
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.training = False
        self.offset = encoder.encoder.decode_chunk_size * 2
        self.segment = self.offset + 7
        self.num_features = encoder.encoder.num_features
        self.num_encoder_layers = encoder.encoder.num_encoder_layers
        self.num_encoders = len(self.num_encoder_layers)
        self.encoder_dims = encoder.encoder.encoder_dims
        self.decode_chunk_size = encoder.encoder.decode_chunk_size
        self.num_left_chunks = encoder.encoder.num_left_chunks

        self.context_size = decoder.decoder.context_size
        self.vocab_size = decoder.decoder.vocab_size
        self.blank_id = decoder.decoder.blank_id
        self.joiner_dim = joiner.output_linear.in_features

    def get_init_state(
        self,
        device: torch.device = torch.device("cpu"),
    ) -> list[Tensor]:
        return self.encoder.encoder.get_init_state(device)

    @classmethod
    def from_pretrained(cls) -> "HfZipformer":
        SOURCE_REPO = "https://github.com/k2-fsa/icefall"
        COMMIT_HASH = "693f069de73fd91d7c2009571245d97221cc3a3f"
        with SourceAsRoot(
            SOURCE_REPO,
            COMMIT_HASH,
            "icefall",
            1,
        ):
            sys.path.append(
                "egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
            )
            os.system(
                "cp egs/librispeech/ASR/pruned_transducer_stateless7/scaling_converter.py  egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
            )
            os.system(
                "cp egs/librispeech/ASR/transducer_stateless/encoder_interface.py  egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
            )
            os.system(
                "cp egs/librispeech/ASR/pruned_transducer_stateless7/scaling.py    egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
            )
            os.system(
                "cp egs/librispeech/ASR/pruned_transducer_stateless7/decoder.py    egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
            )
            os.system(
                "cp egs/librispeech/ASR/pruned_transducer_stateless7/joiner.py     egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
            )
            os.system(
                "cp egs/librispeech/ASR/pruned_transducer_stateless7/model.py      egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
            )
            from egs.librispeech.ASR.pruned_transducer_stateless7_streaming.decoder import (
                Decoder,
            )
            from egs.librispeech.ASR.pruned_transducer_stateless7_streaming.joiner import (
                Joiner,
            )
            from egs.librispeech.ASR.pruned_transducer_stateless7_streaming.model import (
                Transducer,
            )
            from egs.librispeech.ASR.pruned_transducer_stateless7_streaming.scaling_converter import (
                convert_scaled_to_non_scaled,
            )
            from egs.librispeech.ASR.pruned_transducer_stateless7_streaming.zipformer import (
                Zipformer,
            )
            from icefall.checkpoint import load_checkpoint

        model_config_file = os.path.join(os.path.dirname(__file__), "model_config.yaml")
        yaml = YAML(typ="safe")
        with open(model_config_file) as file:
            config = yaml.load(file)
        encoder = Zipformer(
            num_features=config["num_features"],
            output_downsampling_factor=config["output_downsampling_factor"],
            zipformer_downsampling_factors=config["downsampling_factor"],
            encoder_dims=config["encoder_dim"],
            attention_dim=config["attention_dim"],
            encoder_unmasked_dims=config["encoder_unmasked_dim"],
            nhead=config["nhead"],
            feedforward_dim=config["feedforward_dim"],
            cnn_module_kernels=config["cnn_module_kernel"],
            num_encoder_layers=config["num_encoder_layers"],
            num_left_chunks=config["num_left_chunks"],
            short_chunk_size=config["short_chunk_size"],
            decode_chunk_size=config["decode_chunk_size"],
        )
        decoder = Decoder(
            vocab_size=config["vocab_size"],
            decoder_dim=config["decoder_dim"],
            blank_id=config["blank_id"],
            context_size=config["context_size"],
        )
        joiner = Joiner(
            encoder_dim=config["encoder_dim_joiner"],
            decoder_dim=config["decoder_dim_joiner"],
            joiner_dim=config["joiner_dim"],
            vocab_size=config["vocab_size"],
        )
        orig_fp_model = Transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            encoder_dim=config["encoder_dim_joiner"],
            decoder_dim=config["decoder_dim_joiner"],
            joiner_dim=config["joiner_dim"],
            vocab_size=config["vocab_size"],
        )
        ckpt = hf_hub_download(
            repo_id="pfluo/k2fsa-zipformer-chinese-english-mixed",
            filename="exp/pretrained.pt",
            cache_dir="./ckpt",
        )
        load_checkpoint(ckpt, orig_fp_model)
        orig_fp_model.eval()
        convert_scaled_to_non_scaled(orig_fp_model, inplace=True)
        Modify_EncoderModule(orig_fp_model, config)

        return cls(
            ZipformerEncoder(orig_fp_model.encoder, orig_fp_model.joiner.encoder_proj),
            ZipformerDecoder(orig_fp_model.decoder, orig_fp_model.joiner.decoder_proj),
            ZipformerJoiner(orig_fp_model.joiner.output_linear),
        )
