# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse

import torch

from qai_hub_models.models._shared.llama3_ao.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.utils.base_model import BaseModel


def llama3_quantize(
    quantized_model_cls: type[BaseModel], fp_model_cls: type[BaseModel], model_id: str
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length for the model",
    )
    parser.add_argument(
        "--calibration-sequence-length",
        type=int,
        default=DEFAULT_CALIBRATION_SEQ_LEN,
        help="Sequence length to be used during calibration (does not need to match deployment sequence length).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory to export the ONNX model and encodings.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Input directory with custom weights.",
    )
    parser.add_argument(
        "--use-seq-mse",
        action="store_true",
        default=False,
        help="Add to apply Sequential MSE.",
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        raise ValueError(
            "This model requires a CUDA GPU (V100/A100) on it to do quantization. Please re-try with GPU machine."
        )

    context_length = args.context_length
    seq_len = args.calibration_sequence_length

    # Create the floating point model
    extra = dict(sequence_length=seq_len, context_length=context_length)

    if args.checkpoint:
        extra["checkpoint"] = args.checkpoint

    fp_model = fp_model_cls.from_pretrained(**extra).to(device).eval()

    model_quant = quantized_model_cls.from_pretrained(
        context_length=context_length,
        sequence_length=seq_len,
        checkpoint=None,
        host_device=device,
        fp_model=fp_model,
    )

    # Passing the floating point model to create kvcache to do inference later.
    dataloader = model_quant.get_calibration_data(fp_model=fp_model)

    # Release GPU memory
    fp_model.to(torch.device("cpu"))
    torch.cuda.empty_cache()

    # Do calibration (and Sequential MSE if flag is set)
    if args.use_seq_mse:
        print()
        print(
            "WARNING: Sequential MSE takes about 4.5 for model with 3B parameters and about 8 hours for model with 8B parameters."
        )

    model_quant.quantize(
        data=dataloader,
        use_seq_mse=args.use_seq_mse,
    )

    model_quant.save_calibrated_checkpoint(args.output, fp_model=fp_model)
    print("Quantization completed successfully.")
    print()
    print(
        "    If you are using custom weights via checkpoint folder, please add a copy of the model config to the output checkpoint folder. This will help run the demo and evaluation correctly for your model."
    )
    print()
    print("Evaluate:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.evaluate --checkpoint {args.output} --task wikitext-ppl"
    )
    print()
    print("Demo:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.demo --checkpoint {args.output} --prompt 'What is gravity?'"
    )
    print()
    print("Export:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.export --checkpoint {args.output} --device 'Snapdragon 8 Elite QRD' --skip-profiling --skip-inferencing --output-dir output"
    )
