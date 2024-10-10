# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import qai_hub as hub

from qai_hub_models.models.llama_v2_7b_chat_quantized import Model
from qai_hub_models.utils.base_model import TargetRuntime

CONTEXT_LEN = 1024

TARGET_GEN2 = "snapdragon-gen2"
TARGET_GEN3 = "snapdragon-gen3"
TARGET_ANDROID = "android"
TARGET_WINDOWS = "windows"
WINDOWS_XELITE = "Snapdragon X Elite CRD"
GALAXY_S23_FAMILY = "Samsung Galaxy S23 (Family)"
GALAXY_S24_FAMILY = "Samsung Galaxy S24 (Family)"

TARGET_OS_SDK_NAME = {
    TARGET_ANDROID: "aarch64-android",
    TARGET_WINDOWS: "aarch64-windows-msvc",
}
DSP_ARCH_FROM_GEN = {TARGET_GEN2: "v73", TARGET_GEN3: "v75"}
SOC_ID_FROM_GEN = {TARGET_GEN2: 43, TARGET_GEN3: 57}
QNN_SDK_ROOT = os.environ["QNN_SDK_ROOT"]

if not os.path.exists(QNN_SDK_ROOT):
    raise RuntimeError("QNN_SDK_ROOT not set.")

env = os.environ.copy()
env["PYTHONPATH"] = QNN_SDK_ROOT + "/benchmarks/QNN/:" + QNN_SDK_ROOT + "/lib/python"
env["PATH"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + env["PATH"]
env["LD_LIBRARY_PATH"] = QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
env["HEXAGON_TOOLS_DIR"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang"

ALL_COMPONENTS = [
    "llama2_0",
    "llama2_1",
    "llama2_2",
    "llama2_3",
]

# Each components is two sub-components linked together with shared weights
ALL_SUB_COMPONENTS = {
    "llama2_0": [
        "PromptProcessor_1_Quantized",
        "TokenGenerator_1_Quantized",
    ],
    "llama2_1": [
        "PromptProcessor_2_Quantized",
        "TokenGenerator_2_Quantized",
    ],
    "llama2_2": [
        "PromptProcessor_3_Quantized",
        "TokenGenerator_3_Quantized",
    ],
    "llama2_3": [
        "PromptProcessor_4_Quantized",
        "TokenGenerator_4_Quantized",
    ],
}


def export_model(
    device: str,
    output_dir: str,
) -> List[str]:
    """
    Exports Llama2 model via AI Hub and return locally downloaded model path
    """

    model_name = "llama_v2_7b_chat_quantized"
    output_path = Path(output_dir)

    components = ALL_COMPONENTS

    # 1. Initialize PyTorch model
    model = Model.from_pretrained()

    compile_jobs: Dict[str, List[hub.client.CompileJob]] = {}
    for component_name in components:
        compile_jobs[component_name] = []
        for sub_component_name in ALL_SUB_COMPONENTS[component_name]:
            # Load model part

            component = model.load_model_part(sub_component_name)

            input_spec = component.get_input_spec()

            source_model = component.convert_to_hub_source_model(
                TargetRuntime.QNN,
                output_path,
                input_spec,
                external_onnx_weights=True,
                output_names=component.get_output_names(),
            )

            quant_calibration_data = component.get_calibration_data(
                TargetRuntime.QNN, input_spec=input_spec
            )

            # 2. Compile the models to an on-device asset
            model_compile_options = component.get_hub_compile_options(
                TargetRuntime.QNN, f" --qnn_graph_name {sub_component_name}"
            )
            print(f"Optimizing model {sub_component_name} to run on-device")
            submitted_compile_job = hub.submit_compile_job(
                model=source_model,
                input_specs=input_spec,
                device=hub.Device(device),
                name=f"{model_name}_{sub_component_name}",
                calibration_data=quant_calibration_data,
                options=model_compile_options,
            )

            # Delete aimet model
            os.remove(source_model)

            compile_jobs[component_name].append(submitted_compile_job)
            # Free model part to reduce memory-pressure
            del component

    # 3. Link jobs
    link_jobs: Dict[str, hub.client.LinkJob] = {}
    for component_name, compile_jobs_list in compile_jobs.items():
        models = []
        for compile_job in compile_jobs_list:
            if compile_job.get_status().code == "FAILED":
                raise RuntimeError(
                    f"Compile job failed for {component_name}. Please re-run export."
                )
            models.append(compile_job.get_target_model())

        link_jobs[component_name] = hub.submit_link_job(
            models, name=f"{model_name}_{component_name}"
        )

    # 4. Download target context binaries
    os.makedirs(output_path, exist_ok=True)
    model_paths = []
    for component_name, compile_job in link_jobs.items():
        model_path = str(output_path / f"{component_name}.serialized.bin")
        model_paths.append(model_path)
        target_model: hub.Model = compile_job.get_target_model()  # type: ignore
        target_model.download(model_path)

    return model_paths


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
                        "spill-fill-bufsize": 431294976,
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


def _copy_genie_bin(target_os_sdk_name: str, shared_bins_dir: str):
    genie_file_path = os.path.join(
        QNN_SDK_ROOT,
        "bin",
        target_os_sdk_name,
        "genie-t2t-run.exe"
        if TARGET_WINDOWS in target_os_sdk_name
        else "genie-t2t-run",
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


def _get_hub_device(target_gen: str, target_os: str):
    if target_gen == TARGET_GEN2:
        if target_os == TARGET_WINDOWS:
            return WINDOWS_XELITE
        elif target_os == TARGET_ANDROID:
            return GALAXY_S23_FAMILY
    else:
        if target_os == TARGET_ANDROID:
            return GALAXY_S24_FAMILY

    raise RuntimeError(
        f"Unsupported target_gen {target_gen} and target_os {target_os} provided."
    )


def generate_shared_bins(
    output_dir: str,
    tokenizer_zip_path: str,
    target_gen: str,
    target_os_type: str,
):

    os.makedirs(output_dir, exist_ok=True)
    hub_device = _get_hub_device(target_gen, target_os_type)
    export_model(device=hub_device, output_dir=output_dir)

    # Copy htp config and genie config in output directory
    _generate_and_save_htp_config(target_gen, output_dir=output_dir)

    # Mmap is not supported on Windows
    use_mmap = target_os_type == TARGET_ANDROID
    _generate_and_save_genie_config(
        model="llama2",
        output_dir=output_dir,
        bin_names=[f"{name}.serialized.bin" for name in ALL_COMPONENTS],
        use_mmap=use_mmap,
    )
    target_os_sdk_name = TARGET_OS_SDK_NAME[target_os_type]
    _copy_genie_bin(target_os_sdk_name, output_dir)
    _copy_target_libs(target_gen, target_os_sdk_name, output_dir)
    _copy_tokenizer(tokenizer_zip_path, output_dir)

    print("Genie compatible qnn binary generated.")
    print(
        f"Please copy binaries, htp and genie configuration from {output_dir} on target {target_gen} device."
    )
