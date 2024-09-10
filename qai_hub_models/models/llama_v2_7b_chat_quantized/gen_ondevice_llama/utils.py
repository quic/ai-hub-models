# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import onnx
import qai_hub as hub
from qai_hub.util.dataset_entries_converters import _h5_data_to_np_array
from qai_hub.util.zipped_model import unzip_model

CONTEXT_LEN = 1024
TARGET_OS_SDK_NAME = {"android": "aarch64-android", "windows": "aarch64-windows-msvc"}
DSP_ARCH_FROM_GEN = {"snapdragon-gen2": "v73", "snapdragon-gen3": "v75"}
SOC_ID_FROM_GEN = {"snapdragon-gen2": 43, "snapdragon-gen3": 57}
QNN_SDK_ROOT = os.environ["QNN_SDK_ROOT"]

if not os.path.exists(QNN_SDK_ROOT):
    raise RuntimeError("QNN_SDK_ROOT not set.")

env = os.environ.copy()
env["PYTHONPATH"] = QNN_SDK_ROOT + "/benchmarks/QNN/:" + QNN_SDK_ROOT + "/lib/python"
env["PATH"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + env["PATH"]
env["LD_LIBRARY_PATH"] = QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
env["HEXAGON_TOOLS_DIR"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang"


def _h5_dataset_to_dict(
    h5f: h5py.File,
) -> Dict[str, List[np.ndarray]]:
    inputs = h5f["data"]
    assert isinstance(inputs, h5py.Group)

    # Determine number of inputs & initialize array with that length
    input_datas: List[Tuple[str, List[np.ndarray]]] = [("", [])] * len(inputs)
    num_batches = 0

    # Generate a Feature for each input
    for input_index, input_id in enumerate(inputs):
        # Get Data
        all_batches = inputs[input_id]
        assert isinstance(all_batches, h5py.Group)

        # Get Name
        feature_name = all_batches.attrs["name"]
        assert isinstance(feature_name, str)

        # Convert Data to Numpy Array
        feature_data = _h5_data_to_np_array(all_batches)
        if num_batches == 0:
            num_batches = len(feature_data)
        elif len(feature_data) != num_batches:
            raise ValueError(
                f"Dataset feature {feature_data} has {str(len(feature_data))} features, but previous batches have {str(num_batches)} features."
            )

        # Insert into feature mapping with correct order
        order = all_batches.attrs.get("order")
        if order is None:
            order = input_index
        input_datas[order] = (feature_name, feature_data)

    return {k: v for k, v in input_datas}


def _make_qnn_inputs(output_dir: str, input_values: Dict[str, List[np.ndarray]]) -> str:
    """
    Save np values to files and create input_list file for qnn program to
    consume via --input_list

    Args:
    - input_values: ordered named inputs in np array (dtype matters)

    Returns
    - input_list_path: path to use for --input_list in qnn prorams
    """
    input_list_path = os.path.join(output_dir, "input_list.txt")
    with open(input_list_path, "w") as f:
        for sample_idx, sample in enumerate(zip(*input_values.values())):
            name_path_pairs = {}
            for i, (name, np_val) in enumerate(zip(input_values.keys(), sample)):
                val_path = os.path.join(output_dir, f"inputs{sample_idx}_{i}.raw")
                np_val.tofile(val_path)
                name_path_pairs[name] = val_path
            line = " ".join(f"{name}:={path}" for name, path in name_path_pairs.items())
            print(line, file=f)
    return input_list_path


def _prepare_input_dim_dtype_args(
    input_shapes: Dict[str, Tuple[Tuple[int, ...], str]]
) -> List[str]:
    input_dim_args = []
    input_dtype_args = []
    for key, (shape, dtype) in input_shapes.items():
        input_dim_args += [
            "--input_dim",
            key,
            ",".join(str(x) for x in shape),
        ]
        input_dtype_args += [
            "--input_dtype",
            key,
            dtype,
        ]
    return input_dim_args + input_dtype_args


def _prepare_data_input_args(
    output_dir: str,
    input_shapes: List[Tuple[str, Tuple[Tuple[int, ...], str]]],
    calibration_dataset_path: str | None,
) -> List[str]:
    bitwidth_args = [
        "--bias_bitwidth",
        "32",
        "--weight_bw",
        "8",
        "--act_bitwidth",
        "16",
    ]

    # Ref: build/third_party/qnn_sdk/universal/docs/QNN/general/quantization.html#quantizing-a-model
    input_values: Dict[str, List[np.ndarray]]

    calibration_dataset = h5py.File(calibration_dataset_path)

    input_values = _h5_dataset_to_dict(calibration_dataset)
    input_names_dataset = input_values.keys()
    for i, (name, np_list) in enumerate(input_values.items()):
        if name not in input_names_dataset:
            raise RuntimeError(
                f"Model defines input '{name}', but the calibration dataset does not define that input."
            )
        shape_tuple = input_shapes[i][1][0]
        for j, t in enumerate(np_list):
            if t.shape != shape_tuple:
                raise RuntimeError(
                    f"Input {name} batch {j} has shape "
                    + f"{t.shape}. Expect: {shape_tuple}"
                )

    input_list_path = _make_qnn_inputs(output_dir, input_values)

    # input shape args
    input_shapes_dict = {name: shapetype for name, shapetype in input_shapes}
    input_dim_dtype_args = _prepare_input_dim_dtype_args(input_shapes_dict)

    data_input_args = (
        [
            "--input_list",
            input_list_path,
        ]
        + input_dim_dtype_args
        + bitwidth_args
    )
    return data_input_args


def _extract_shapes_from_onnx_list(
    tensor_list: Iterable[onnx.ValueInfoProto],
) -> List[Tuple[str, Tuple[Tuple[int, ...], str]]]:
    """
    Extracts the shapes from an ONNX model and return a dictionary of input
    shapes.
    """
    shapes = []
    for x in tensor_list:
        dtype = x.type.tensor_type.elem_type
        dtype_name = onnx.helper.tensor_dtype_to_np_dtype(dtype).name
        shape = tuple(
            [
                d.dim_value if d.HasField("dim_value") else -1
                for d in x.type.tensor_type.shape.dim
            ]
        )
        shapes.append((x.name, (shape, dtype_name)))
    return shapes


def _get_preserve_io_args(
    input_shapes: List[Tuple[str, Tuple[Tuple[int, ...], str]]],
    output_shapes: List[Tuple[str, Tuple[Tuple[int, ...], str]]],
) -> List[str]:
    preserve_io = []
    for key, _ in input_shapes:
        preserve_io.append(key)

    input_layout_other_args = []
    for key, (shape, dtype) in input_shapes:
        if key in preserve_io and 3 <= len(shape) <= 5:
            input_layout_other_args += ["--input_layout", key, "NONTRIVIAL"]

    for key, _ in output_shapes:
        preserve_io.append(key)

    preserve_io_args = []
    if preserve_io:
        preserve_io_args = ["--preserve_io", "layout"] + preserve_io

    return preserve_io_args + input_layout_other_args


def generate_lib(
    onnx_model_path: str,
    encoding_file_path: str,
    dataset_path: str,
    cpp_bin_dir: str,
    out_lib_path: str,
):
    target_arch = "x86_64-linux-clang"
    model_name = str(Path(os.path.basename(onnx_model_path)).with_suffix(""))

    onnx_model = onnx.load_model(onnx_model_path)
    model_input_shapes = _extract_shapes_from_onnx_list(onnx_model.graph.input)
    model_output_shapes = _extract_shapes_from_onnx_list(onnx_model.graph.output)

    dataset_dir = os.path.join(str(Path(dataset_path).with_suffix("")), "_data")
    print(dataset_path, dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    data_input_args = _prepare_data_input_args(
        output_dir=str(dataset_dir),
        input_shapes=model_input_shapes,
        calibration_dataset_path=dataset_path,
    )

    preserve_io_args = _get_preserve_io_args(model_input_shapes, model_output_shapes)

    intermediate_cpp_file = str(os.path.join(cpp_bin_dir, f"{model_name}.cpp"))
    intermediate_bin_file = str(os.path.join(cpp_bin_dir, f"{model_name}.bin"))
    args = [
        QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qnn-onnx-converter",
        "--input_network",
        onnx_model_path,
        "--output_path",
        intermediate_cpp_file,
        "--no_simplification",
        "--quantization_overrides",
        encoding_file_path,
    ]
    args += preserve_io_args
    args += data_input_args
    print(" ".join(args))

    subprocess.run(args, env=env)

    print("Generating model lib...")
    args = [
        QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qnn-model-lib-generator",
        "-c",
        intermediate_cpp_file,
        "-b",
        intermediate_bin_file,
        "-t",
        target_arch,
        "-o",
        os.path.dirname(out_lib_path),
        "-l",
        model_name,
    ]

    print(" ".join(args))
    subprocess.run(args, env=env)

    output_file = f"{os.path.dirname(out_lib_path)}/{target_arch}/lib{model_name}.so"

    if not os.path.exists(output_file):
        raise RuntimeError("The QNN graph compiler did not produce the output file")

    shutil.copyfile(output_file, out_lib_path)


def _generate_and_save_htp_config(target_gen: str, output_dir: str):
    soc_id = SOC_ID_FROM_GEN[target_gen]
    dsp_arch = DSP_ARCH_FROM_GEN[target_gen]

    # HTP backend config file (htp_backend_ext_config.json) example
    htp_backend_ext_config_data = {
        "devices": [
            {
                "soc_id": soc_id,
                "dsp_arch": dsp_arch,
                "cores": [
                    {"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}
                ],
            }
        ],
        "memory": {"mem_type": "shared_buffer"},
        "context": {"weight_sharing_enabled": True},
    }

    # write the htp config file to the destination
    with open(os.path.join(output_dir, "htp_backend_ext_config.json"), "w") as f:
        f.write(json.dumps(htp_backend_ext_config_data, indent=4))


def _generate_and_save_genie_config(
    model: str, output_dir: str, bin_names: List[str], use_mmap: bool = True
):

    if model != "llama2":
        raise RuntimeError("Only Llama2 Genie config is supported.")

    genie_config = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": CONTEXT_LEN,
                "n-vocab": 32000,
                "bos-token": 1,
                "eos-token": 2,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 4,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "spill-fill-bufsize": 15000000000,
                        "use-mmap": use_mmap,
                        "mmap-budget": 0,
                        "poll": True,
                        "pos-id-dim": 64,
                        "cpu-mask": "0xe0",
                        "kv-dim": 128,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {"version": 1, "ctx-bins": bin_names},
                },
            },
        }
    }

    with open(os.path.join(output_dir, "htp-model-config-llama2-7b.json"), "w") as f:
        f.write(json.dumps(genie_config, indent=4))


def generate_bin(
    output_dir: str,
    output_name: str,
    graph_names: List[str],
    qnn_lib_paths: List[str],
    target_gen: str,
):
    htp_backend_extensions_data = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "/tmp/htp_config.json",
        },
    }

    with open("/tmp/htp_backend_extensions.json", "w") as f:
        f.write(json.dumps(htp_backend_extensions_data, indent=4))

    # Make sure the graph names matches the names used in calling
    # `qnn-onnx-converter -o /path/to/<model_name>.cpp`
    if len(graph_names) != len(qnn_lib_paths):
        raise RuntimeError("Graph names must match provided QNN libs.")

    htp_backend_config_data = {
        "graphs": [
            {
                "graph_names": graph_names,
                "fp16_relaxed_precision": 1,
                "vtcm_mb": 0,
                "O": 3,
            }
        ],
        "devices": [
            {
                "dsp_arch": DSP_ARCH_FROM_GEN[target_gen],
                "soc_model": SOC_ID_FROM_GEN[target_gen],
            }
        ],
        "memory": {"mem_type": "shared_buffer"},
        "context": {"weight_sharing_enabled": True},
    }
    if len(qnn_lib_paths) > 1:
        # This is the key to generating QNN context binary with shared weights
        htp_backend_config_data["context"] = {
            "weight_sharing_enabled": True,
        }

    with open("/tmp/htp_config.json", "w") as f:
        f.write(json.dumps(htp_backend_config_data, indent=4))

    model_so_paths = ",".join(qnn_lib_paths)
    args = [
        QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qnn-context-binary-generator",
        "--model",
        f"{model_so_paths}",
        "--backend",
        "libQnnHtp.so",
        "--output_dir",
        output_dir,
        "--binary_file",
        output_name,
        "--config_file",
        "/tmp/htp_backend_extensions.json",
    ]
    print(" ".join(args))

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
    output, error = proc.communicate()
    print(output.decode(), error.decode())


def _copy_genie_bin(target_os_sdk_name: str, shared_bins_dir: str):
    genie_file_path = os.path.join(
        QNN_SDK_ROOT,
        "bin",
        target_os_sdk_name,
        "genie-t2t-run.exe" if "windows" in target_os_sdk_name else "genie-t2t-run",
    )
    out_file_name = os.path.basename(genie_file_path)
    shutil.copyfile(str(genie_file_path), os.path.join(shared_bins_dir, out_file_name))


def _copy_target_libs(target_gen: str, target_os_sdk_name: str, shared_bins_dir: str):
    # Copy hexgaon-v* libs first
    lib_dir = os.path.join(
        QNN_SDK_ROOT, "lib", f"hexagon-{DSP_ARCH_FROM_GEN[target_gen]}", "unsigned"
    )
    for f in os.listdir(lib_dir):
        shutil.copyfile(str(os.path.join(lib_dir, f)), os.path.join(shared_bins_dir, f))

    # Copy target os libs
    lib_dir = os.path.join(QNN_SDK_ROOT, "lib", target_os_sdk_name)
    for f in os.listdir(lib_dir):
        shutil.copyfile(str(os.path.join(lib_dir, f)), os.path.join(shared_bins_dir, f))


def _copy_tokenizer(tokenizer_path: str, shared_bins_dir: str):
    with zipfile.ZipFile(tokenizer_path, "r") as zip_ref:
        zip_ref.extractall(shared_bins_dir)


def generate_shared_bins(
    output_dir: str,
    model_ids: str,
    tokenizer_zip_path: str,
    target_gen: str,
    target_os_type: str,
):
    in_model_dir = str(os.path.join(output_dir, "intermediate_data", "input_models"))
    os.makedirs(in_model_dir, exist_ok=True)

    cpp_model_dir = str(os.path.join(output_dir, "intermediate_data", "cpp_models"))
    os.makedirs(cpp_model_dir, exist_ok=True)

    model_dir = str(os.path.join(output_dir, "intermediate_data", "model_libs"))
    os.makedirs(model_dir, exist_ok=True)

    inter_input_data = str(os.path.join(output_dir, "intermediate_data", "input_data"))
    os.makedirs(inter_input_data, exist_ok=True)

    shared_bins_dir = str(
        os.path.join(output_dir, f"shared_bins_{target_gen}_{target_os_type}")
    )
    os.makedirs(shared_bins_dir, exist_ok=True)

    num_of_splits = len(model_ids["pp"])
    serialized_output_names = [f"llama2_{i}.serialized" for i in range(num_of_splits)]
    for i in range(num_of_splits):
        graph_names = []
        model_so_list = []
        for model_type in ["pp", "tg"]:
            input_data_path = f"{inter_input_data}/data_{model_type}_{i}.h5"
            input_model_zip = f"{in_model_dir}/model_{model_type}_{i}.aimet.zip"
            extracted_model_path = f"{in_model_dir}/model_{model_type}_{i}"
            model_so = f"{model_dir}/model_{model_type}_{i}.so"

            model = model_ids[model_type][i]
            job = hub.get_model(model).producer

            graph_names.append(job.model.name.split(".")[0])
            model_so_list.append(model_so)

            if os.path.exists(model_so):
                print(f"{model_so} exists. Skipping compilation...")
                continue

            if not os.path.exists(input_data_path):
                job.calibration_dataset.download(input_data_path)

            if (
                os.path.exists(extracted_model_path)
                and len(os.listdir(extracted_model_path)) == 1
            ):
                print("Using previously extracted model")
                extracted_model_path = (
                    f"{extracted_model_path}/{os.listdir(extracted_model_path)[0]}"
                )
            else:
                if not os.path.exists(input_model_zip):
                    job.model.download(input_model_zip)

                print("Extracting model")
                extracted_model_path = unzip_model(
                    input_model_zip, extracted_model_path
                )

            input_names = []
            for name, shape_info in job.shapes.items():
                name_shape_type = (name, (shape_info[0], shape_info[1]))
                input_names.append(name_shape_type)

            onnx_model_path = None
            encoding_file_path = None
            for f_name in os.listdir(extracted_model_path):
                if f_name.endswith(".onnx"):
                    onnx_model_path = str(os.path.join(extracted_model_path, f_name))
                elif f_name.endswith(".encodings"):
                    encoding_file_path = str(os.path.join(extracted_model_path, f_name))

            if onnx_model_path is None or encoding_file_path is None:
                raise RuntimeError("ONNX model or encoding not found.")

            print(f"Generating model lib for split-{i} {model_type}")
            generate_lib(
                onnx_model_path,
                encoding_file_path,
                input_data_path,
                cpp_model_dir,
                model_so,
            )

        print(f"Generating weight shared qnn-bin for split-{i}")
        generate_bin(
            output_dir=shared_bins_dir,
            output_name=serialized_output_names[i],
            graph_names=graph_names,
            qnn_lib_paths=model_so_list,
            target_gen=target_gen,
        )

    # Copy htp config and genie config in output directory
    _generate_and_save_htp_config(target_gen, output_dir=shared_bins_dir)
    # Mmap is not supported on Windows
    use_mmap = target_os_type == "android"
    _generate_and_save_genie_config(
        model="llama2",
        output_dir=shared_bins_dir,
        bin_names=[f"{name}.bin" for name in serialized_output_names],
        use_mmap=use_mmap,
    )
    target_os_sdk_name = TARGET_OS_SDK_NAME[target_os_type]
    _copy_genie_bin(target_os_sdk_name, shared_bins_dir)
    _copy_target_libs(target_gen, target_os_sdk_name, shared_bins_dir)
    _copy_tokenizer(tokenizer_zip_path, shared_bins_dir)

    print("Genie compatible qnn binary generated.")
    print(
        f"Please copy binaries, htp and genie configuration from {shared_bins_dir} on target {target_gen} device."
    )
