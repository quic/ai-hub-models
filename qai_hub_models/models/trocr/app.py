# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Generator, List

import torch
from PIL.Image import Image

from qai_hub_models.models.trocr.model import KVCache, TrOCR


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
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.io_processor = model.io_processor

        self.pad_token_id = model.pad_token_id
        self.eos_token_id = model.eos_token_id
        self.start_token_id = model.start_token_id
        self.max_seq_len = model.max_seq_len

    def preprocess_image(self, image: Image) -> torch.Tensor:
        """Convert a raw image (resize, normalize) into a pyTorch tensor that can be used as input to TrOCR inference.
        This also converts the image to RGB, which is the expected input channel layout for TrOCR.

        For more information on preprocessing, see https://huggingface.co/docs/transformers/preprocessing."""
        assert (
            self.io_processor is not None
        ), "TrOCR processor most be provided to use type Image as an input."
        return self.io_processor(image.convert("RGB"), return_tensors="pt").pixel_values

    def predict(self, *args, **kwargs):
        # See predict_text_from_image.
        return self.predict_text_from_image(*args, **kwargs)

    def predict_text_from_image(
        self, pixel_values_or_image: torch.Tensor | Image, raw_output: bool = False
    ) -> torch.Tensor | List[str]:
        """
        From the provided image or tensor, predict the line of text contained within.

        Parameters:
            pixel_values_or_image: torch.Tensor
                Input PIL image (before pre-processing) or pyTorch tensor (after image pre-processing).
            raw_output: bool
                If false, return a list of predicted strings (one for each batch). Otherwise, return a tensor of predicted token IDs.

        Returns:
           The output word / token sequence (representative of the text contained in the input image).

           The prediction will be a list of strings (one string per batch) if self.io_processor != None and raw_output=False.
           Otherwise, a `torch.Tensor` of shape [batch_size, predicted_sequence_length] is returned. It contains predicted token IDs.
        """
        gen = self.stream_predicted_text_from_image(pixel_values_or_image, raw_output)
        _ = last = next(gen)
        for last in gen:
            pass
        return last

    def stream_predicted_text_from_image(
        self, pixel_values_or_image: torch.Tensor | Image, raw_output: bool = False
    ) -> Generator[torch.Tensor | List[str], None, None]:
        """
        From the provided image or tensor, predict the line of text contained within.
        The returned generator will produce a single output per decoder iteration.

        The generator allows the client to "stream" output from the decoder
        (eg. get the prediction one word at as time as they're predicted, instead of waiting for the entire output sequence to be predicted)

        Parameters:
            pixel_values_or_image: torch.Tensor
                Input PIL image (before pre-processing) or pyTorch tensor (after image pre-processing).
            raw_output: bool
                If false, return a list of predicted strings (one for each batch). Otherwise, return a tensor of predicted token IDs.

        Returns:
            A python generator for the output word / token sequence (representative of the text contained in the input image).
            The generator will produce one output for every decoder iteration.

            The prediction will be a list of strings (one string per batch) if self.io_processor != None and raw_output=False.
            Otherwise, a `torch.Tensor` of shape [batch_size, predicted_sequence_length] is returned. It contains predicted token IDs.
        """
        if isinstance(pixel_values_or_image, Image):
            pixel_values = self.preprocess_image(pixel_values_or_image)
        else:
            pixel_values = pixel_values_or_image

        batch_size = pixel_values.shape[0]
        eos_token_id_tensor = torch.tensor([self.eos_token_id], dtype=torch.int32)

        # Run encoder
        kv_cache_cross_attn = self.encoder(pixel_values)

        # Initial KV Cache
        initial_attn_cache = get_empty_attn_cache(
            batch_size,
            self.decoder.num_decoder_layers,
            self.decoder.decoder_attention_heads,
            self.decoder.embeddings_per_head,
        )
        initial_kv_cache = combine_kv_caches(kv_cache_cross_attn, initial_attn_cache)
        kv_cache = initial_kv_cache

        # Prepare decoder input IDs. Shape: [batch_size, 1]
        initial_input_ids = (
            torch.ones((batch_size, 1), dtype=torch.int32) * self.start_token_id
        )
        input_ids = initial_input_ids

        # Prepare decoder output IDs. Shape: [batch_size, seq_len]
        output_ids = input_ids

        # Keep track of which sequences are already finished. Shape: [batch_size]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.int32)

        while unfinished_sequences.max() != 0 and (
            self.max_seq_len is None or output_ids.shape[-1] < self.max_seq_len
        ):
            # Get next tokens. Shape: [batch_size]
            outputs = self.decoder(input_ids, *kv_cache)
            next_tokens = outputs[0]
            kv_cache_attn = outputs[1:]

            # Finished sentences should have padding token appended instead of the prediction.
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (
                1 - unfinished_sequences
            )

            input_ids = torch.unsqueeze(next_tokens, -1)
            output_ids = torch.cat([output_ids, input_ids], dim=-1)
            yield self.io_processor.batch_decode(
                output_ids, skip_special_tokens=True
            ) if self.io_processor and not raw_output else output_ids

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    torch.unsqueeze(next_tokens, -1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                    .type(torch.int32)
                )

            # Re-construct kv cache with new sequence.
            kv_cache = combine_kv_caches(kv_cache_cross_attn, kv_cache_attn)


def combine_kv_caches(
    kv_cache_cross_attn: KVCache,
    kv_cache_attn: KVCache,
) -> KVCache:
    """
    Generates full KV Cache from cross attention KV cache and attention KV cache.

    Parameters:
        kv_cache_cross_attn: Tuple[kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val, cv_cache_cross_attn_1_key, ...]
            Cross attn KV cache generated by CrossAttnKVGenerator.
            len(tuple) == 2 * number of source model decoder layers.

        kv_cache_attn: Tuple[kv_cache_attn_0_key, kv_cache_attn_0_val cv_cache_attn_1_key, ...]
            Attn generated by the decoder, or None (generate empty cache) if the decoder has not run yet.
            len(tuple) == 2 * number of source model decoder layers.

    Returns:
        kv_cache: Tuple[kv_cache_attn_0_key, kv_cache_attn_0_val,
                        kv_cache_cross_attn_0_key, kv_cache_cross_attn_0_val,
                        kv_cache_attn_1_key, ...]
            Combined KV Cache.
            len(tuple) == 4 * number of source model decoder layers.
    """
    # Construct remaining kv cache with a new empty sequence.
    kv_cache = [torch.Tensor()] * len(kv_cache_cross_attn) * 2

    # Combine KV Cache.
    for i in range(0, len(kv_cache_cross_attn) // 2):
        kv_cache[4 * i] = kv_cache_attn[2 * i]
        kv_cache[4 * i + 1] = kv_cache_attn[2 * i + 1]
        kv_cache[4 * i + 2] = kv_cache_cross_attn[2 * i]
        kv_cache[4 * i + 3] = kv_cache_cross_attn[2 * i + 1]

    return (*kv_cache,)


def get_empty_attn_cache(
    batch_size: int,
    num_decoder_layers: int,
    decoder_attention_heads: int,
    embeddings_per_head: int,
) -> KVCache:
    """
    Generates empty cross attn KV Cache for use in the first iteration of the decoder.

    Parameters:
        batch_size: Batch size.
        num_decoder_layers: NUmber of decoder layers in the decoder.
        decoder_attention_heads: Number of attention heads in the decoder.
        embeddings_per_head: The count of the embeddings in each decoder attention head.

    Returns:
        kv_cache: Tuple[kv_cache_attn_0_key, kv_cache_attn_0_val, kv_cache_attn_1_key, ...]
            len(tuple) == 2 * number of source model decoder layers.
    """
    kv_cache = []
    for i in range(0, num_decoder_layers):
        kv_cache.append(
            torch.zeros(
                (
                    batch_size,
                    decoder_attention_heads,
                    0,
                    embeddings_per_head,
                )
            )
        )
        kv_cache.append(
            torch.zeros(
                (
                    batch_size,
                    decoder_attention_heads,
                    0,
                    embeddings_per_head,
                )
            )
        )
    return (*kv_cache,)
