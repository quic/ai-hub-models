# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

from qai_hub_models.models.protocols import FromPrecompiledProtocol
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import (
    BasePrecompiledModel,
    CollectionModel,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.llm_helpers import get_kv_cache_names

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2


def get_cached_asset(part: int) -> str:
    model_path = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "snapdragon_8_elite/models.zip"
    ).fetch(extract=True)
    return os.path.join(model_path, f"weight_sharing_model_{part}_of_4.serialized.bin")


class Mistral_7B_Instruct_v0_3_Quantized(FromPrecompiledProtocol, CollectionModel):
    """
    Mistral_7B_Instruct_v0_3_Quantized class consists of
        - Prompt Processor (split into 4 parts)
        - Token Generator (split into 4 parts)

    All models are pre-trained, quantized (int4/int8 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    def __init__(
        self,
        prompt_processor_part1: BasePrecompiledModel,
        prompt_processor_part2: BasePrecompiledModel,
        prompt_processor_part3: BasePrecompiledModel,
        prompt_processor_part4: BasePrecompiledModel,
        token_generator_part1: BasePrecompiledModel,
        token_generator_part2: BasePrecompiledModel,
        token_generator_part3: BasePrecompiledModel,
        token_generator_part4: BasePrecompiledModel,
    ) -> None:
        self.prompt_processor_part1 = prompt_processor_part1
        self.prompt_processor_part2 = prompt_processor_part2
        self.prompt_processor_part3 = prompt_processor_part3
        self.prompt_processor_part4 = prompt_processor_part4
        self.token_generator_part1 = token_generator_part1
        self.token_generator_part2 = token_generator_part2
        self.token_generator_part3 = token_generator_part3
        self.token_generator_part4 = token_generator_part4

    @classmethod
    def from_precompiled(cls) -> Mistral_7B_Instruct_v0_3_Quantized:
        return Mistral_7B_Instruct_v0_3_Quantized(
            prompt_processor_part1=PromptProcessor_Part1.from_precompiled(),
            prompt_processor_part2=PromptProcessor_Part2.from_precompiled(),
            prompt_processor_part3=PromptProcessor_Part3.from_precompiled(),
            prompt_processor_part4=PromptProcessor_Part4.from_precompiled(),
            token_generator_part1=TokenGenerator_Part1.from_precompiled(),
            token_generator_part2=TokenGenerator_Part2.from_precompiled(),
            token_generator_part3=TokenGenerator_Part3.from_precompiled(),
            token_generator_part4=TokenGenerator_Part4.from_precompiled(),
        )


class PromptProcessor_Part1(BasePrecompiledModel):
    """
    Prompt Processor Part1 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> PromptProcessor_Part1:
        return PromptProcessor_Part1(get_cached_asset(part=1))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "input_ids": ((1, 128), "int32"),
            "past_key_1_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_1_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_0_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_0_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_2_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_2_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_3_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_3_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_4_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_4_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_5_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_5_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_6_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_6_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_7_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_7_in": ((8, 1, 3968, 128), "uint8"),
            "position_ids_cos": ((1, 1, 128, 64), "uint16"),
            "position_ids_sin": ((1, 1, 128, 64), "uint16"),
            "attention_mask": ((1, 1, 128, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=0, end=8) + [
            "_model_layers_7_Add_1_Add_output_0"
        ]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_1_of_4"
        )


class PromptProcessor_Part2(BasePrecompiledModel):
    """
    Prompt Processor Part2 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> PromptProcessor_Part2:
        return PromptProcessor_Part2(get_cached_asset(part=2))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "_model_layers_7_Add_1_Add_output_0": ((1, 128, 4096), "uint16"),
            "past_key_9_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_9_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_8_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_8_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_10_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_10_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_11_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_11_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_12_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_12_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_13_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_13_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_14_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_14_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_15_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_15_in": ((8, 1, 3968, 128), "uint8"),
            "position_ids_cos": ((1, 1, 128, 64), "uint16"),
            "position_ids_sin": ((1, 1, 128, 64), "uint16"),
            "attention_mask": ((1, 1, 128, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=8, end=16) + [
            "_model_layers_15_Add_1_Add_output_0"
        ]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_2_of_4"
        )


class PromptProcessor_Part3(BasePrecompiledModel):
    """
    Prompt Processor Part3 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> PromptProcessor_Part3:
        return PromptProcessor_Part3(get_cached_asset(part=3))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "_model_layers_15_Add_1_Add_output_0": ((1, 128, 4096), "uint16"),
            "past_key_17_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_17_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_16_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_16_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_18_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_18_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_19_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_19_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_20_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_20_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_21_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_21_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_22_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_22_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_23_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_23_in": ((8, 1, 3968, 128), "uint8"),
            "position_ids_cos": ((1, 1, 128, 64), "uint16"),
            "position_ids_sin": ((1, 1, 128, 64), "uint16"),
            "attention_mask": ((1, 1, 128, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=16, end=24) + [
            "_model_layers_23_Add_1_Add_output_0"
        ]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_3_of_4"
        )


class PromptProcessor_Part4(BasePrecompiledModel):
    """
    Prompt Processor Part4 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> PromptProcessor_Part4:
        return PromptProcessor_Part4(get_cached_asset(part=4))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "_model_layers_23_Add_1_Add_output_0": ((1, 128, 4096), "uint16"),
            "past_key_25_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_25_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_24_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_24_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_26_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_26_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_27_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_27_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_28_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_28_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_29_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_29_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_30_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_30_in": ((8, 1, 3968, 128), "uint8"),
            "past_key_31_in": ((8, 1, 128, 3968), "uint8"),
            "past_value_31_in": ((8, 1, 3968, 128), "uint8"),
            "position_ids_cos": ((1, 1, 128, 64), "uint16"),
            "position_ids_sin": ((1, 1, 128, 64), "uint16"),
            "attention_mask": ((1, 1, 128, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=24, end=32) + ["logits"]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar128_cl4096_4_of_4"
        )


class TokenGenerator_Part1(BasePrecompiledModel):
    """
    Token Generator Part1 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> TokenGenerator_Part1:
        return TokenGenerator_Part1(get_cached_asset(part=1))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "input_ids": ((1, 1), "int32"),
            "past_key_1_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_1_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_0_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_0_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_2_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_2_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_3_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_3_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_4_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_4_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_5_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_5_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_6_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_6_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_7_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_7_in": ((8, 1, 4095, 128), "uint8"),
            "position_ids_cos": ((1, 1, 1, 64), "uint16"),
            "position_ids_sin": ((1, 1, 1, 64), "uint16"),
            "attention_mask": ((1, 1, 1, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=0, end=8) + [
            "_model_layers_7_Add_1_Add_output_0"
        ]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_1_of_4"
        )


class TokenGenerator_Part2(BasePrecompiledModel):
    """
    Token Generator Part2 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> TokenGenerator_Part2:
        return TokenGenerator_Part2(get_cached_asset(part=2))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "_model_layers_7_Add_1_Add_output_0": ((1, 1, 4096), "uint16"),
            "past_key_9_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_9_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_8_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_8_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_10_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_10_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_11_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_11_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_12_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_12_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_13_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_13_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_14_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_14_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_15_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_15_in": ((8, 1, 4095, 128), "uint8"),
            "position_ids_cos": ((1, 1, 1, 64), "uint16"),
            "position_ids_sin": ((1, 1, 1, 64), "uint16"),
            "attention_mask": ((1, 1, 1, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=8, end=16) + [
            "_model_layers_15_Add_1_Add_output_0"
        ]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_2_of_4"
        )


class TokenGenerator_Part3(BasePrecompiledModel):
    """
    Token Generator Part3 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> TokenGenerator_Part3:
        return TokenGenerator_Part3(get_cached_asset(part=3))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "_model_layers_15_Add_1_Add_output_0": ((1, 1, 4096), "uint16"),
            "past_key_17_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_17_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_16_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_16_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_18_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_18_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_19_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_19_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_20_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_20_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_21_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_21_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_22_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_22_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_23_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_23_in": ((8, 1, 4095, 128), "uint8"),
            "position_ids_cos": ((1, 1, 1, 64), "uint16"),
            "position_ids_sin": ((1, 1, 1, 64), "uint16"),
            "attention_mask": ((1, 1, 1, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=16, end=24) + [
            "_model_layers_23_Add_1_Add_output_0"
        ]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_3_of_4"
        )


class TokenGenerator_Part4(BasePrecompiledModel):
    """
    Token Generator Part4 for Mistral 7b.

    Pre-trained, quantized (int8/int4 weight, float32 activations)
    and compiled into serialized binary for Qualcomm Snapdragon 8 Elite.
    """

    @classmethod
    def from_precompiled(cls) -> TokenGenerator_Part4:
        return TokenGenerator_Part4(get_cached_asset(part=4))

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "_model_layers_23_Add_1_Add_output_0": ((1, 1, 4096), "uint16"),
            "past_key_25_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_25_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_24_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_24_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_26_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_26_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_27_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_27_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_28_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_28_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_29_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_29_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_30_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_30_in": ((8, 1, 4095, 128), "uint8"),
            "past_key_31_in": ((8, 1, 128, 4095), "uint8"),
            "past_value_31_in": ((8, 1, 4095, 128), "uint8"),
            "position_ids_cos": ((1, 1, 1, 64), "uint16"),
            "position_ids_sin": ((1, 1, 1, 64), "uint16"),
            "attention_mask": ((1, 1, 1, 4096), "uint16"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return get_kv_cache_names(start=24, end=32) + ["logits"]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        return (
            profile_options + " --qnn_options context_enable_graphs=ar1_cl4096_4_of_4"
        )
