# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import struct
from pathlib import Path

from transformers import MarianTokenizer

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models.templates.opus_mt.opusmt_metadata_json import (
    create_t2t_metadata,
)

# Language code to name mapping for supplementary file descriptions
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "jap": "Japanese",
    "de": "German",
    "fr": "French",
    "it": "Italian",
}
TOKENIZER_ENCODE_NAME = "tokenizer_encode.bin"
TOKENIZER_DECODE_NAME = "tokenizer_decode_lookups.bin"

# Models with separate source/target SentencePiece vocabularies (e.g. tc-big
# architectures), where vocab.json only reflects the target vocabulary and its ids
# don't correspond to source.spm's own ids. Must match the special case in app.py.
SEPARATE_SOURCE_VOCAB_MODELS = {"Helsinki-NLP/opus-mt-tc-big-en-ko"}


def write_opus_mt_supplementary_files(
    opus_mt_version: str,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
) -> None:
    """
    Write supplementary files for OpusMT models.

    Parameters
    ----------
    opus_mt_version
        HuggingFace model ID, e.g. "Helsinki-NLP/opus-mt-en-es"
    output_dir
        Directory to write supplementary files to
    metadata
        Model metadata object to update with supplementary file info
    """
    if metadata.runtime != TargetRuntime.VOICE_AI:
        return

    # Parse language pair from model name (e.g., "Helsinki-NLP/opus-mt-en-es" -> "en", "es")
    model_suffix = opus_mt_version.split("/")[-1]  # "opus-mt-en-es"
    parts = model_suffix.split("-")
    src_lang = parts[-2]  # "en"
    tgt_lang = parts[-1]  # "es"

    src_name = LANGUAGE_NAMES.get(src_lang, src_lang.capitalize())
    tgt_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang.capitalize())
    lang_pair = f"{src_name}-to-{tgt_name}"

    # Generate tokenizer binaries and record them as supplementary files
    encode_path, decode_path = generate_tokenizer_bins(opus_mt_version, output_dir)
    if not encode_path or not decode_path:
        raise ValueError("Failed to generate tokenizer binaries")

    # Add the tokenizer binaries to the metadata's supplementary files
    metadata.supplementary_files[encode_path.name] = (
        f"{lang_pair} Tokenizer encode binary for optimal token access during tokenization"
    )
    metadata.supplementary_files[decode_path.name] = (
        f"{lang_pair} Tokenizer decode binary for converting model output token IDs to token strings"
    )

    from qai_hub_models.models.templates.opus_mt.model import (
        MAX_SEQ_LEN_DEC,
        MAX_SEQ_LEN_ENC,
    )

    # Generate T2T metadata JSON and record it as a supplementary file
    t2t_metadata = create_t2t_metadata(
        opus_mt_version,
        output_dir,
        metadata,
        MAX_SEQ_LEN_ENC,
        MAX_SEQ_LEN_DEC,
        TOKENIZER_ENCODE_NAME,
        TOKENIZER_DECODE_NAME,
    )
    if t2t_metadata:
        t2t_path = Path(output_dir) / "config.json"
        t2t_metadata.to_json(t2t_path, exclude_defaults=False)
        if t2t_path:
            metadata.supplementary_files[t2t_path.name] = (
                f"T2T metadata JSON for {lang_pair} translation"
            )


def get_tokenizer(hf_model_name: str) -> MarianTokenizer:
    """Get the tokenizer for the specified OpusMT model."""
    return MarianTokenizer.from_pretrained(hf_model_name)


def generate_tokenizer_bins(
    hf_model_name: str, output_dir: str | os.PathLike
) -> tuple[Path, Path]:
    """
    Generate optimized tokenizer encode and decode binaries for an Opus-MT model.

    This function extracts tokenizer information from a HuggingFace model and generates
    two binary files:
    1. tokenizer_encode.bin - optimized lookup table for text encoding
    2. tokenizer_decode_lookups.bin - lookup table for token ID decoding

    Parameters
    ----------
    hf_model_name
        HuggingFace model ID, e.g. "Helsinki-NLP/opus-mt-zh-en"
    output_dir
        Output directory path for storing the generated binary files

    Returns
    -------
    tuple[Path, Path]
        Tuple of (encode_bin_path, decode_bin_path) containing relative paths to generated binaries
    """
    # 1. Load tokenizer and get vocabulary
    tokenizer = MarianTokenizer.from_pretrained(hf_model_name)
    opus_vocab_dict = tokenizer.get_vocab()  # Get token -> id mapping

    # 2. Create output directory and file paths
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encode_path = out_dir / TOKENIZER_ENCODE_NAME
    decode_path = out_dir / TOKENIZER_DECODE_NAME

    # ========== Generate encode binary file ==========
    # 3. Prepare data for encoding
    special_token_list = ["</s>", "<unk>", "<pad>"]  # Special token list
    spm_model = tokenizer.spm_source  # Get SentencePiece model

    # 4. Build vocabulary dictionary with score information
    vocab_dict = {}
    if hf_model_name in SEPARATE_SOURCE_VOCAB_MODELS:
        # These models (e.g. tc-big architectures) use a source-only SentencePiece
        # vocabulary for the encoder; vocab.json only reflects the target/decoder
        # vocabulary and its ids don't correspond to source.spm's own ids, so the
        # encode table must be built directly from source.spm (see app.py's matching
        # special case for encoding raw text with sp_source.EncodeAsIds).
        for spm_id in range(spm_model.GetPieceSize()):
            piece = spm_model.id_to_piece(spm_id)
            if piece in special_token_list:
                continue  # Skip special tokens
            score = spm_model.get_score(spm_id)
            vocab_dict[piece] = [score, spm_id, len(piece)]
    else:
        for opus_vocab_token, opus_vocab_id in opus_vocab_dict.items():
            if opus_vocab_token in special_token_list:
                continue  # Skip special tokens

            # Get token score from SentencePiece model
            spm_id = spm_model.piece_to_id(opus_vocab_token)
            score = float("-inf") if spm_id == 0 else spm_model.get_score(spm_id)
            vocab_dict[opus_vocab_token] = [score, opus_vocab_id, len(opus_vocab_token)]

    # 5. Build DAG (Directed Acyclic Graph) for efficient encoding
    def create_node(text: str, c_len: int = 0) -> dict:
        """Create DAG node"""
        return {
            "len": c_len,
            "score": 0.0,
            "token_id": 0,
            "entry_id": 0,
            "is_leaf": 0,
            "num_children": 0,
            "offset": 0,
            "text": text,
            "children": [],
        }

    root = create_node("", 0)  # Create root node
    dump_pos_idx = 1

    # 6. Build character-level DAG paths for each token
    for key, val in vocab_dict.items():
        if key in special_token_list:
            continue

        token_str, score, token_id, num_chars = (
            str(key),
            float(val[0]),
            int(val[1]),
            int(val[2]),
        )
        current = root
        chars = []

        # Build path character by character
        for char in token_str:
            chars.append(char)
            chars_str = "".join(chars)

            # Find or create child node
            curr_node = None
            for child in current["children"]:
                if child["text"] == chars_str:
                    curr_node = child
                    break

            if curr_node is None:
                # Create new child node
                curr_node = create_node(chars_str, len(chars_str.encode("utf-8")))
                current["children"].append(curr_node)
                current["num_children"] += 1
                dump_pos_idx += 1

            current = curr_node
            # Mark as leaf node if we've reached the end of the token
            if len(chars) == num_chars:
                current.update({"score": score, "is_leaf": 1, "token_id": token_id})

    # 7. Assign unique entry IDs to all nodes
    node_stack, entry_id = [root], 0
    while node_stack:
        curr_node = node_stack.pop(0)
        curr_node["entry_id"] = entry_id
        entry_id += 1
        node_stack.extend(curr_node["children"])

    # 8. Collect all nodes and calculate offsets
    all_nodes = []
    node_stack = [root]
    while node_stack:
        curr_node = node_stack.pop(0)
        all_nodes.append(curr_node)
        node_stack.extend(curr_node["children"])

    # 9. Calculate offset for each node in the binary file
    for i, node in enumerate(all_nodes):
        if i == 0:
            node["offset"] = 0
        else:
            prev_node = all_nodes[i - 1]
            # Calculate aligned offset (4-byte alignment)
            node["offset"] = (
                prev_node["offset"]
                + ((prev_node["len"] + 1 + 3) & ~0x03)
                + prev_node["num_children"] * 4
            )

    # 10. Write encode binary file
    with open(encode_path, "wb") as bin_file:
        num_entries = len(all_nodes)
        packed_data_fixed = b""  # Fixed-length data
        packed_data_variable = b""  # Variable-length data
        size_data_fixed = 0
        size_data_variable = 0

        # Pack data for each node
        for curr_node in all_nodes:
            # Fixed-length part: length, score, token_id, is_leaf, num_children, offset
            struct_format = "I f I B I I"
            # Variable-length part: text content + child node ID list
            variable_format = f"{curr_node['len'] + 1}s {curr_node['num_children']}I"
            children_ids = [child["entry_id"] for child in curr_node["children"]]

            packed_data_fixed += struct.pack(
                struct_format,
                curr_node["len"],
                curr_node["score"],
                curr_node["token_id"],
                curr_node["is_leaf"],
                curr_node["num_children"],
                curr_node["offset"],
            )
            packed_data_variable += struct.pack(
                variable_format, curr_node["text"].encode(), *children_ids
            )

            size_data_fixed += struct.calcsize(struct_format)
            size_data_variable += struct.calcsize(variable_format)

        # Write file header (num_entries, fixed_data_size, variable_data_size) and data
        bin_file.write(
            struct.pack("III", num_entries, size_data_fixed, size_data_variable)
        )
        bin_file.write(packed_data_fixed + packed_data_variable)

    # ========== Generate decode binary file ==========
    # 11. Build decode lookup table
    vocab_final = [""] * len(opus_vocab_dict)
    for token, token_id in opus_vocab_dict.items():
        vocab_final[token_id] = token  # Sort by token_id
    vocab_final.append("<pad>")  # Add padding token

    # 12. Write decode binary file
    with open(decode_path, "wb") as bin_file:
        lookups_bytes, offsets = b"", [0]

        # Encode all tokens as UTF-8 byte strings
        for token in vocab_final:
            token_bytes = token.encode("utf-8") + b"\0"  # Add null terminator
            lookups_bytes += token_bytes
            offsets.append(offsets[-1] + len(token_bytes))

        # Calculate lookup table parameters
        lookups_size = offsets[-1] + 1
        offsets = offsets[:-1]  # Remove last offset
        lookups_count = len(offsets)

        # Write decode file: lookup_count, total_size, offset_array, string_data
        struct_format = f"II{lookups_count}I{lookups_size}s"
        bin_file.write(
            struct.pack(
                struct_format, lookups_count, lookups_size, *offsets, lookups_bytes
            )
        )

    return encode_path, decode_path
