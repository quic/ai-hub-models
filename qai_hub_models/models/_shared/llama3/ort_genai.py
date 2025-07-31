# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import torch
from transformers import PretrainedConfig

from qai_hub_models.models._shared.llm.model import PositionProcessorBase


def make_ort_genai_config(
    llm_config: PretrainedConfig,
    context_length: int,
    prompt_sequence_length: int,
    config_pipeline: dict[str, Any],
) -> dict:
    return {
        "model": {
            "bos_token_id": llm_config.bos_token_id,
            "context_length": context_length,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "head_size": llm_config.head_dim,
                "hidden_size": llm_config.hidden_size,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask_before_processor",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_%d_in",
                    "past_value_names": "past_value_%d_in",
                },
                "outputs": {
                    "logits": "logits_dequantized",
                    "present_key_names": "past_key_%d_out",
                    "present_value_names": "past_value_%d_out",
                },
                "num_attention_heads": llm_config.num_attention_heads,
                "num_hidden_layers": llm_config.num_hidden_layers,
                "num_key_value_heads": llm_config.num_key_value_heads,
                "sliding_window": {
                    "window_size": prompt_sequence_length,
                    "pad_value": 128,
                },
                "pipeline": [config_pipeline],
            },
            "eos_token_id": llm_config.eos_token_id,
            "pad_token_id": llm_config.eos_token_id[0],  # correct?
            "type": "decoder-pipeline",
            "vocab_size": llm_config.vocab_size,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": True,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": 2048,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": True,
            "repetition_penalty": 1.0,
            "temperature": 0.6,
            "top_k": 1,
            "top_p": 0.9,
        },
    }


def create_ort_genai_assets(
    model_name: str,
    llm_config: PretrainedConfig,
    position_processor_cls: type[PositionProcessorBase],
    encodings_path: str | Path,
    context_length: int,
    prompt_sequence_length: int,
    onnx_model_path_from_sub_component_name: dict[str, str],
    num_splits: int,
    qairt_version: str,
    output_dir: str | Path,
) -> None:
    """
    This takes a folder of context binaries and adds:
    - ONNX wrapped files (with EPContexts)
    - ORT GenAI auxiliary files (quantize, dequantize, position-processor)
    """
    output_dir = Path(output_dir)
    config_pipeline: dict[str, Any] = {}

    with open(encodings_path) as f:
        encodings = json.load(f)
    act_encodings = encodings["activation_encodings"]

    # Wrap ONNX files
    instantiations = [
        ("prompt", prompt_sequence_length),
        ("token", 1),
    ]
    cpu_session_options: dict[str, list[dict]] = {
        "provider_options": [{}],
    }
    npu_session_options = {
        "provider_options": [
            {
                "qnn": {
                    "backend_path": "QnnHtp.dll",
                    "htp_performance_mode": "burst",
                    "enable_htp_shared_memory_allocator": "1",
                    "qnn_context_priority": "high",
                }
            }
        ]
    }

    # Generate auxiliary file: Position processor
    attention_mask_before_processor = torch.randint(
        0, 2, (1, context_length), dtype=torch.int32
    )
    position_ids = torch.randint(0, 128, (1, 128), dtype=torch.int32)

    model = position_processor_cls(context_length=context_length)
    attention_mask, position_ids_cos, position_ids_sin = model(
        attention_mask_before_processor, position_ids
    )

    position_processor_base_name = "position-processor.onnx"
    position_processor_inputs = ["attention_mask_before_processor", "position_ids"]
    position_processor_outputs = [
        "attention_mask_before_quantizer",
        "position_ids_cos_before_quantizer",
        "position_ids_sin_before_quantizer",
    ]
    config_pipeline["position-processor"] = {
        "filename": position_processor_base_name,
        "inputs": position_processor_inputs,
        "outputs": position_processor_outputs,
        "session_options": cpu_session_options,
    }

    torch.onnx.export(
        model,
        (attention_mask_before_processor, position_ids),
        Path(output_dir) / position_processor_base_name,
        input_names=position_processor_inputs,
        output_names=position_processor_outputs,
        dynamic_axes={
            "position_ids": {1: "sequence_length"},
            "attention_mask_before_quantizer": {1: "sequence_length"},
        },
    )

    # Generate auxiliary file: Quantize
    qparams = {
        "attention_mask": QuantParams(
            act_encodings["attention_mask"][0]["scale"],
            -act_encodings["attention_mask"][0]["offset"],
        ),
        "position_ids_cos": QuantParams(
            act_encodings["position_ids_cos"][0]["scale"],
            -act_encodings["position_ids_cos"][0]["offset"],
        ),
        "position_ids_sin": QuantParams(
            act_encodings["position_ids_sin"][0]["scale"],
            -act_encodings["position_ids_sin"][0]["offset"],
        ),
    }
    quantizer_model, quantizer_inputs, quantizer_outputs = quantizer(qparams)
    quantizer_base_name = "quantizer.onnx"
    onnx.save(quantizer_model, Path(output_dir) / quantizer_base_name)
    config_pipeline["quantizer"] = {
        "filename": quantizer_base_name,
        "inputs": quantizer_inputs,
        "outputs": quantizer_outputs,
        "session_options": cpu_session_options,
    }

    for instantiation_name, seq_len in instantiations:
        for i in range(num_splits):
            part_name = f"{i + 1}_of_{num_splits}"
            base_name = f"{model_name}_{instantiation_name}_{part_name}"

            # onnx_model_path_from_sub_component_name["prompt_1_of_3"] = output_dir / "prompt" / "llama_v3_2_3b_instruct_prompt_1_of_3.aimet" / "llama_v3_2_3b_instruct_prompt_1_of_3.onnx"
            onnx_original_path = (
                output_dir
                / instantiation_name
                / (base_name + ".aimet")
                / (base_name + ".onnx")
            )

            # Check IO
            onnx_split = onnx.load(onnx_original_path)
            onnx_input_specs = {}

            def prepare_onnx_specs(values):
                onnx_specs = {}
                for onnx_tensor in values:
                    name = onnx_tensor.name.replace("/", "_").replace(".", "_")

                    orig_elem_type = onnx_tensor.type.tensor_type.elem_type

                    if orig_elem_type == onnx.TensorProto.FLOAT:
                        # Input is float, but will be quantized later.
                        # Unfortunately we do not have this information readily
                        # available. With #12640, we could pull from the model.
                        if "past_" in name:
                            elem_type = onnx.TensorProto.UINT8
                        else:
                            elem_type = onnx.TensorProto.UINT16
                    else:
                        elem_type = orig_elem_type

                    shape = tuple(
                        [
                            dim.dim_value
                            for dim in onnx_tensor.type.tensor_type.shape.dim
                        ]
                    )
                    onnx_specs[name] = (shape, elem_type)
                return onnx_specs

            onnx_input_specs = prepare_onnx_specs(onnx_split.graph.input)
            onnx_output_specs = prepare_onnx_specs(onnx_split.graph.output)
            qnn_bin_rel_path = f"{model_name}_part_{part_name}.bin"
            if output_dir is Path.cwd():
                qnn_bin_path = output_dir / "build" / model_name / qnn_bin_rel_path
            else:
                qnn_bin_path = output_dir / qnn_bin_rel_path
            assert qnn_bin_path.is_file()

            onnx_base_name = f"{model_name}_{instantiation_name}_{part_name}.onnx"
            onnx_output_path = output_dir / onnx_base_name

            # ORT GenAI config information
            config_pipeline[base_name] = {
                "filename": onnx_base_name,
                "inputs": list(onnx_input_specs.keys()),
                "outputs": list(onnx_output_specs.keys()),
                "session_options": npu_session_options,
                "run_on_token_gen": seq_len == 1,
                "run_on_prompt": seq_len != 1,
            }

            graph_name = (
                f"{instantiation_name}_ar{seq_len}_cl{context_length}_{part_name}"
            )

            generate_wrapper_onnx_file(
                graph_name=graph_name,
                onnx_output_path=onnx_output_path,
                onnx_input_specs=onnx_input_specs,
                onnx_output_specs=onnx_output_specs,
                qnn_context_bin_path=qnn_bin_rel_path,
                qairt_version=qairt_version,
            )

    # Generate auxiliary file: Dequantize
    dequantizer_base_name = "dequantizer.onnx"
    dequantizer_model, dequantizer_inputs, dequantizer_outputs = dequantizer()
    onnx.save(dequantizer_model, Path(output_dir) / dequantizer_base_name)
    config_pipeline["dequantizer"] = {
        "filename": dequantizer_base_name,
        "inputs": dequantizer_inputs,
        "outputs": dequantizer_outputs,
        "session_options": cpu_session_options,
    }

    # Create ORT GenAI config file
    config_dict = make_ort_genai_config(
        llm_config,
        context_length,
        prompt_sequence_length,
        config_pipeline,
    )

    config_output_path = Path(output_dir) / "genai_config.json"
    with open(config_output_path, "w") as f:
        # Order of pipeline models matter, make sure not to sort keys.
        json.dump(config_dict, f, indent=4, sort_keys=False)


def generate_wrapper_onnx_file(
    graph_name: str,
    onnx_output_path: str | Path,
    onnx_input_specs: dict[str, tuple[tuple[int, ...], onnx.TensorProto.DataType]],
    onnx_output_specs: dict[str, tuple[tuple[int, ...], onnx.TensorProto.DataType]],
    qnn_context_bin_path: str | Path,
    qairt_version: str,
):
    graph_nodes = []

    model_inputs = []
    for name, (shape, onnx_dtype) in onnx_input_specs.items():
        model_inputs.append(onnx.helper.make_tensor_value_info(name, onnx_dtype, shape))

    ep_cache_context_content = str(qnn_context_bin_path)
    ctx_embed_mode = 0

    qnn_ep_context_node = onnx.helper.make_node(
        "EPContext",
        name=graph_name,
        inputs=list(onnx_input_specs.keys()),
        outputs=list(onnx_output_specs.keys()),
        ep_cache_context=ep_cache_context_content,
        embed_mode=ctx_embed_mode,
        ep_sdk_version=qairt_version,
        source="Qnn",
        domain="com.microsoft",
    )
    graph_nodes.append(qnn_ep_context_node)

    model_outputs = []
    for name, (shape, onnx_dtype) in onnx_output_specs.items():
        model_outputs.append(
            onnx.helper.make_tensor_value_info(name, onnx_dtype, shape)
        )

    graph_def = onnx.helper.make_graph(
        graph_nodes,
        "qnn-onnx-model",
        model_inputs,
        model_outputs,
        [],
        "",
        [],
    )
    model_def = onnx.helper.make_model(graph_def)

    onnx.save(model_def, onnx_output_path)


class QuantParams:
    def __init__(self, scale: float, zero_point: float):
        self.scale = scale
        self.zero_point = int(zero_point)


def quantizer(quant_params: dict[str, QuantParams]):
    inputs = []
    outputs = []
    initializers = []
    nodes = []

    inputs.append(
        onnx.helper.make_tensor_value_info(
            "attention_mask_before_quantizer",
            onnx.TensorProto.FLOAT,
            [1, 1, "sequence_length", "context_length"],
        )
    )

    inputs.append(
        onnx.helper.make_tensor_value_info(
            "position_ids_cos_before_quantizer",
            onnx.TensorProto.FLOAT,
            [1, 1, "sequence_length", "head_dim / 2"],
        )
    )

    inputs.append(
        onnx.helper.make_tensor_value_info(
            "position_ids_sin_before_quantizer",
            onnx.TensorProto.FLOAT,
            [1, 1, "sequence_length", "head_dim / 2"],
        )
    )

    outputs.append(
        onnx.helper.make_tensor_value_info(
            "attention_mask",
            onnx.TensorProto.UINT16,
            [1, 1, "sequence_length", "context_length"],
        )
    )

    outputs.append(
        onnx.helper.make_tensor_value_info(
            "position_ids_cos",
            onnx.TensorProto.UINT16,
            [1, 1, "sequence_length", "head_dim / 2"],
        )
    )

    outputs.append(
        onnx.helper.make_tensor_value_info(
            "position_ids_sin",
            onnx.TensorProto.UINT16,
            [1, 1, "sequence_length", "head_dim / 2"],
        )
    )

    for name in ["attention_mask", "position_ids_cos", "position_ids_sin"]:
        initializers.append(
            onnx.helper.make_tensor(
                f"{name}_scale", onnx.TensorProto.FLOAT, [], [quant_params[name].scale]
            )
        )

        initializers.append(
            onnx.helper.make_tensor(
                f"{name}_offset",
                onnx.TensorProto.UINT16,
                [],
                [quant_params[name].zero_point],
            )
        )

    nodes.append(
        onnx.helper.make_node(
            "QuantizeLinear",
            [
                "attention_mask_before_quantizer",
                "attention_mask_scale",
                "attention_mask_offset",
            ],
            ["attention_mask"],
            "attention_mask_quantizer",
        )
    )

    nodes.append(
        onnx.helper.make_node(
            "QuantizeLinear",
            [
                "position_ids_cos_before_quantizer",
                "position_ids_cos_scale",
                "position_ids_cos_offset",
            ],
            ["position_ids_cos"],
            "position_ids_cos_quantizer",
        )
    )

    nodes.append(
        onnx.helper.make_node(
            "QuantizeLinear",
            [
                "position_ids_sin_before_quantizer",
                "position_ids_sin_scale",
                "position_ids_sin_offset",
            ],
            ["position_ids_sin"],
            "position_ids_sin_quantizer",
        )
    )

    return (
        onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes,
                "quantizer",
                inputs,
                outputs,
                initializers,
            ),
            producer_name="quantizer",
            ir_version=10,
            opset_imports=[onnx.helper.make_opsetid("", 21)],
        ),
        [x.name for x in inputs],
        [x.name for x in outputs],
    )


def dequantizer():
    inputs = []
    outputs = []

    inputs.append(
        onnx.helper.make_tensor_value_info(
            "logits", onnx.TensorProto.UINT16, [1, "sequence_length", "vocab_size"]
        )
    )

    outputs.append(
        onnx.helper.make_tensor_value_info(
            "logits_dequantized",
            onnx.TensorProto.FLOAT,
            [1, "sequence_length", "vocab_size"],
        )
    )

    nodes = [
        onnx.helper.make_node(
            "Cast",
            ["logits"],
            ["logits_dequantized"],
            "logits_dequantizer",
            to=onnx.TensorProto.FLOAT,
        )
    ]

    return (
        onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes,
                "dequantizer",
                inputs,
                outputs,
            ),
            producer_name="dequantizer",
            ir_version=10,
            opset_imports=[onnx.helper.make_opsetid("", 21)],
        ),
        [x.name for x in inputs],
        [x.name for x in outputs],
    )
