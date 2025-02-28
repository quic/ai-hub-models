# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models.trocr.model import TrOCR
from qai_hub_models.utils.model_adapters import TorchNumpyAdapter

KVCacheNp = tuple[np.ndarray, ...]


class TrOCRApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with TrOCR.

    The app uses 2 models:
        * encoder (modified to return cross attention key-value)
        * decoder

    For a given image input, the app will:
        * use the io_processor to pre-process the image (reshape & normalize)
        * call the encoder once
        * run the decoder in a loop until the "end of sentence" token is predicted,
          or the max sequence length (defined by the source model config) is reached
        * map the output tokens to a string via `io_processor`.
    """

    def __init__(self, model: TrOCR):
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        # Wraps torch Module so it takes np ndarray as input and outputs
        if isinstance(self.encoder, torch.nn.Module):
            self.encoder = TorchNumpyAdapter(self.encoder)
        if isinstance(self.decoder, torch.nn.Module):
            self.decoder = TorchNumpyAdapter(self.decoder)
        self.io_processor = model.io_processor

        self.pad_token_id = model.pad_token_id
        self.eos_token_id = model.eos_token_id
        self.start_token_id = model.start_token_id
        self.max_seq_len = model.max_seq_len

    def preprocess_image(self, image: Image) -> np.ndarray:
        """Convert a raw image (resize, normalize) into a pyTorch tensor that can be used as input to TrOCR inference.
        This also converts the image to RGB, which is the expected input channel layout for TrOCR.

        For more information on preprocessing, see https://huggingface.co/docs/transformers/preprocessing."""
        assert (
            self.io_processor is not None
        ), "TrOCR processor most be provided to use type Image as an input."
        return self.io_processor(
            image.convert("RGB"), return_tensors="pt"
        ).pixel_values.numpy()

    def predict(self, *args, **kwargs):
        # See predict_text_from_image.
        return self.predict_text_from_image(*args, **kwargs)

    def predict_text_from_image(
        self, pixel_values_or_image: np.ndarray | Image, raw_output: bool = False
    ) -> np.ndarray | list[str]:
        """
        From the provided image or tensor, predict the line of text contained within.

        Parameters:
            pixel_values_or_image: np.ndarrray
                Input PIL image (before pre-processing) or pyTorch tensor (after image pre-processing).
            raw_output: bool
                If false, return a list of predicted strings (one for each batch). Otherwise, return a tensor of predicted token IDs.

        Returns:
           The output word / token sequence (representative of the text contained in the input image).

           The prediction will be a list of strings (one string per batch) if self.io_processor != None and raw_output=False.
           Otherwise, a `np.ndarray` of shape [batch_size, predicted_sequence_length] is returned. It contains predicted token IDs.
        """
        gen = self.stream_predicted_text_from_image(pixel_values_or_image, raw_output)
        _ = last = next(gen)
        for last in gen:
            pass
        return last

    def stream_predicted_text_from_image(
        self, pixel_values_or_image: np.ndarray | Image, raw_output: bool = False
    ) -> Generator[np.ndarray | list[str], None, None]:
        """
        From the provided image or tensor, predict the line of text contained within.
        The returned generator will produce a single output per decoder iteration.

        The generator allows the client to "stream" output from the decoder
        (eg. get the prediction one word at as time as they're predicted, instead of waiting for the entire output sequence to be predicted)

        Parameters:
            pixel_values_or_image: np.ndarray
                Input PIL image (before pre-processing) or np tensor (after image pre-processing).
            raw_output: bool
                If false, return a list of predicted strings (one for each batch). Otherwise, return a tensor of predicted token IDs.

        Returns:
            A python generator for the output word / token sequence (representative of the text contained in the input image).
            The generator will produce one output for every decoder iteration.

            The prediction will be a list of strings (one string per batch) if self.io_processor != None and raw_output=False.
            Otherwise, a `np.ndarray` of shape [batch_size, predicted_sequence_length] is returned. It contains predicted token IDs.
        """
        if isinstance(pixel_values_or_image, Image):
            pixel_values = self.preprocess_image(pixel_values_or_image)
        else:
            pixel_values = pixel_values_or_image

        batch_size = pixel_values.shape[0]
        eos_token_id_tensor = np.array([self.eos_token_id], dtype=np.int32)

        # Run encoder
        kv_cache_cross_attn = cast(
            KVCacheNp, self.encoder(torch.from_numpy(pixel_values))
        )

        # Initial KV Cache
        initial_attn_cache = get_empty_attn_cache(
            batch_size,
            # -1 because current token's project kv value will be appended to
            # kv_cache.
            self.max_seq_len - 1,
            self.model.num_decoder_layers,
            self.model.decoder_attention_heads,
            self.model.embeddings_per_head,
        )
        initial_kv_cache = combine_kv_caches(kv_cache_cross_attn, initial_attn_cache)
        kv_cache = initial_kv_cache

        # Prepare decoder input IDs. Shape: [batch_size, 1]
        initial_input_ids = (
            np.ones((batch_size, 1), dtype=np.int32) * self.start_token_id
        )
        input_ids = initial_input_ids

        # Prepare decoder output IDs. Shape: [batch_size, seq_len]
        output_ids = input_ids

        # Keep track of which sequences are already finished. Shape: [batch_size]
        unfinished_sequences = np.ones(batch_size, dtype=np.int32)

        decode_pos = np.array([0], dtype=np.int32)
        while unfinished_sequences.max() != 0 and (
            self.max_seq_len is None or output_ids.shape[-1] < self.max_seq_len
        ):
            # Get next tokens. Shape: [batch_size]
            outputs = self.decoder(input_ids, decode_pos, *kv_cache)
            next_tokens = outputs[0]
            kv_cache_attn = cast(tuple[np.ndarray, ...], outputs[1:])

            # Finished sentences should have padding token appended instead of the prediction.
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (
                1 - unfinished_sequences
            )

            input_ids = np.expand_dims(next_tokens, -1)
            output_ids = np.concatenate([output_ids, input_ids], axis=-1)
            yield self.io_processor.batch_decode(
                torch.from_numpy(output_ids), skip_special_tokens=True
            ) if self.io_processor and not raw_output else output_ids

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences * (
                    np.prod(
                        np.expand_dims(next_tokens, -1)
                        != np.expand_dims(eos_token_id_tensor, 1),
                        axis=0,
                    ).astype(np.int32)
                )

            # Re-construct kv cache with new sequence.
            # kv_cache are inserted from the back. Clip 1 from the front
            kv_cache_attn = tuple([v[:, :, 1:] for v in kv_cache_attn])
            kv_cache = combine_kv_caches(kv_cache_cross_attn, kv_cache_attn)
            decode_pos += 1


def combine_kv_caches(
    kv_cache_cross_attn: KVCacheNp,
    kv_cache_attn: KVCacheNp,
) -> KVCacheNp:
    """
    Generates full KV Cache from cross attention KV cache and attention KV cache.

    Parameters:
        kv_cache_cross_attn: tuple[kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val, cv_cache_cross_attn_1_key, ...]
            Cross attn KV cache generated by CrossAttnKVGenerator.
            len(tuple) == 2 * number of source model decoder layers.

        kv_cache_attn: tuple[kv_cache_attn_0_key, kv_cache_attn_0_val cv_cache_attn_1_key, ...]
            Attn generated by the decoder, or None (generate empty cache) if the decoder has not run yet.
            len(tuple) == 2 * number of source model decoder layers.

    Returns:
        kv_cache: tuple[kv_cache_attn_0_key, kv_cache_attn_0_val,
                        kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val,
                        kv_cache_attn_1_key, ...]
            Combined KV Cache.
            len(tuple) == 4 * number of source model decoder layers.
    """
    # Construct remaining kv cache with a new empty sequence.
    kv_cache: list[Any] = [None] * len(kv_cache_cross_attn) * 2

    # Combine KV Cache.
    for i in range(0, len(kv_cache_cross_attn) // 2):
        kv_cache[4 * i] = kv_cache_attn[2 * i]
        kv_cache[4 * i + 1] = kv_cache_attn[2 * i + 1]
        kv_cache[4 * i + 2] = kv_cache_cross_attn[2 * i]
        kv_cache[4 * i + 3] = kv_cache_cross_attn[2 * i + 1]

    none_list = [v for v in kv_cache if v is None]
    assert len(none_list) == 0
    return (*kv_cache,)  # type: ignore[arg-type]


def get_empty_attn_cache(
    batch_size: int,
    max_decode_len: int,
    num_decoder_layers: int,
    decoder_attention_heads: int,
    embeddings_per_head: int,
) -> KVCacheNp:
    """
    Generates empty cross attn KV Cache for use in the first iteration of the decoder.

    Parameters:
        batch_size: Batch size.
        num_decoder_layers: Number of decoder layers in the decoder.
        decoder_attention_heads: Number of attention heads in the decoder.
        embeddings_per_head: The count of the embeddings in each decoder attention head.

    Returns:
        kv_cache: tuple[kv_cache_attn_0_key, kv_cache_attn_0_val, kv_cache_attn_1_key, ...]
            len(tuple) == 2 * number of source model decoder layers.
    """
    kv_cache = []
    for i in range(0, num_decoder_layers):
        kv_cache.append(
            np.zeros(
                (
                    batch_size,
                    decoder_attention_heads,
                    max_decode_len,
                    embeddings_per_head,
                )
            ).astype(np.float32)
        )
        kv_cache.append(
            np.zeros(
                (
                    batch_size,
                    decoder_attention_heads,
                    max_decode_len,
                    embeddings_per_head,
                )
            ).astype(np.float32)
        )
    return (*kv_cache,)
