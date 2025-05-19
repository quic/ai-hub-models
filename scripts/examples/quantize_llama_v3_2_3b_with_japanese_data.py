# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import gc

import torch

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.wikitext import load_calibration_data
from qai_hub_models.datasets.wikitext_ja import WikiText_Japanese
from qai_hub_models.models._shared.llama3_ao.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.models.llama_v3_2_3b_instruct import Model
from qai_hub_models.models.llama_v3_2_3b_instruct.model import Llama3_2_3B as FP_Model
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader

if __name__ == "__main__":
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
        help="Calibration length to be used.",
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
    parser.add_argument(
        "--num-samples-ja",
        type=int,
        default=2,
        help="Number of Japanese WikiText samples to use. If --calibration-sequence-length is 2k, then each sample is equal to 2k tokens. The more number of samples you use, the longer the time it takes.",
    )
    parser.add_argument(
        "--num-samples-en",
        type=int,
        default=40,
        help="Number of English WikiText samples to use. If --calibration-sequence-length is 2k, then each sample is equal to 2k tokens. The more number of samples you use, the longer the time it takes.",
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        raise ValueError(
            "This model requires a GPU (V100/A100) with atleast 32 GB VRAM available on it to do quantization. Please re-try with GPU machine."
        )

    context_length = args.context_length
    seq_len = args.calibration_sequence_length

    # Create the floating point model
    extra = dict(sequence_length=seq_len, context_length=context_length)

    if args.checkpoint:
        extra["checkpoint"] = args.checkpoint

    fp_model = FP_Model.from_pretrained(**extra).to(device).eval()

    english_dataset = load_calibration_data(
        split=DatasetSplit.TRAIN,
        model=fp_model,
        device=device,
        num_samples=args.num_samples_en,
    )
    if args.num_samples_ja > 0:
        japanese_dataset = load_calibration_data(
            split=DatasetSplit.TRAIN,
            model=fp_model,
            device=device,
            num_samples=args.num_samples_ja,
            dataset_cls=WikiText_Japanese,
        )

        for name in english_dataset.keys():
            english_dataset[name].extend(japanese_dataset[name])

    dataloader = dataset_entries_to_dataloader(english_dataset)

    model_quant = Model.from_pretrained(
        context_length=context_length,
        sequence_length=seq_len,
        checkpoint=None,
        host_device=device,
        fp_model=fp_model,
    )
    del fp_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Do calibration (and Sequential MSE if flag is set)
    if args.use_seq_mse:
        print()
        print("WARNING: Sequential MSE takes about 4.5 (3B) to 10 (8B) hours.")

    model_quant.quantize(
        data=dataloader,
        original_onnx_model=model_quant.onnx_model,
        use_seq_mse=args.use_seq_mse,
    )

    model_quant.create_checkpoint()
    print("Quantization completed successfully.")
