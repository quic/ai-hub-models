# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch

from qai_hub_models.models._shared.opus_mt.model import (
    MAX_SEQ_LEN_DEC,
    MAX_SEQ_LEN_ENC,
    OpusMTDecoder,
    OpusMTEncoder,
    get_tokenizer,
)


class OpusMTApp:
    """
    OpusMTApp runs OpusMT encoder and decoder to translate text between languages.
    It supports all OpusMT model variants from Helsinki-NLP.
    """

    def __init__(
        self,
        encoder: OpusMTEncoder,
        decoder: OpusMTDecoder,
        hf_model_id: str,
        max_input_length: int = MAX_SEQ_LEN_ENC,
        max_output_length: int = MAX_SEQ_LEN_DEC,
    ) -> None:
        """
        Initialize OpusMT application.

        Parameters
        ----------
        encoder
            The OpusMT encoder model
        decoder
            The OpusMT decoder model
        hf_model_id
            HuggingFace model identifier (e.g., "Helsinki-NLP/opus-mt-en-zh")
        max_input_length
            Maximum input sequence length
        max_output_length
            Maximum output sequence length
        """
        self.encoder = encoder
        self.decoder = decoder

        # Ensure models are in eval mode and on CPU
        if isinstance(self.encoder, torch.nn.Module):
            self.encoder = self.encoder.to("cpu").eval()
        if isinstance(self.decoder, torch.nn.Module):
            self.decoder = self.decoder.to("cpu").eval()

        self.hf_model_id = hf_model_id
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Load tokenizer
        self.tokenizer = get_tokenizer(hf_model_id)

        # Get special tokens
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def predict(self, *args: Any, **kwargs: Any) -> str | list[str]:
        """Alias for translate method."""
        return self.translate(*args, **kwargs)

    def translate(
        self, text: str | list[str], max_length: int | None = None
    ) -> str | list[str]:
        """
        Translate the provided text.

        Parameters
        ----------
        text
            Input text(s) to translate
        max_length
            Maximum output sequence length.

        Returns
        -------
        translated : str | list[str]
            Translated text(s)
        """
        if isinstance(text, str):
            return self._translate_single(text, max_length)
        return [self._translate_single(t, max_length) for t in text]

    def _translate_single(self, text: str, max_length: int | None = None) -> str:
        """
        Translate a single text string.

        Parameters
        ----------
        text
            Input text to translate
        max_length
            Maximum output sequence length. If not provided, uses self.max_output_length.

        Returns
        -------
        str
            Translated text
        """
        if max_length is None:
            max_length = self.max_output_length

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Pad to fixed length for consistent inference
        batch_size, seq_len = input_ids.shape
        if seq_len < self.max_input_length:
            pad_length = self.max_input_length - seq_len
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full(
                        (batch_size, pad_length),
                        self.pad_token_id,
                        dtype=input_ids.dtype,
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros((batch_size, pad_length), dtype=attention_mask.dtype),
                ],
                dim=1,
            )

        # Convert to int32 for model compatibility
        input_ids = input_ids.to(torch.int32)
        attention_mask = attention_mask.to(torch.int32)

        # Run encoder
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids, attention_mask)
        if not isinstance(encoder_outputs, (tuple, list)):
            encoder_outputs = (encoder_outputs,)

        # Initialize decoder inputs - use 65000 as starting token
        decoder_input_ids = torch.zeros([1, 1], dtype=torch.int32)
        token = 65000
        decoder_input_ids[0, 0] = token

        # Initialize past key values for decoder
        past_key_values = []
        num_layers = self.decoder.num_layers

        # Initialize kv cache
        for layer_idx in range(num_layers):
            # past_self_key_states and past_self_value_states (empty for first step)
            past_key_values.append(
                torch.zeros(
                    [
                        encoder_outputs[0].shape[0],
                        encoder_outputs[0].shape[1],
                        255,
                        encoder_outputs[0].shape[3],
                    ],
                    dtype=torch.float32,
                )
            )
            past_key_values.append(
                torch.zeros(
                    [
                        encoder_outputs[1].shape[0],
                        encoder_outputs[1].shape[1],
                        255,
                        encoder_outputs[1].shape[3],
                    ],
                    dtype=torch.float32,
                )
            )
            # cross_key_states and cross_value_states from encoder
            past_key_values.append(encoder_outputs[2 * layer_idx])
            past_key_values.append(encoder_outputs[2 * layer_idx + 1])

        # Generate tokens
        generated_tokens = []

        for step in range(max_length):
            position = torch.tensor([step], dtype=torch.int32)

            # Run decoder
            with torch.no_grad():
                decoder_outputs = self.decoder(
                    decoder_input_ids,
                    attention_mask,
                    position,
                    *past_key_values,
                )

            logits = decoder_outputs[0]
            token = int(torch.argmax(logits, -1)[0, 0].item())

            # Check for end of sequence - use token == 0
            if token == 0:
                break

            generated_tokens.append(token)

            # Update decoder input for next step
            decoder_input_ids[0, 0] = token

            # Update past key values
            present_key_values = decoder_outputs[1:]  # Get present key values
            for layer_idx in range(num_layers):
                # Update self-attention past key values
                # The present_key_values from our decoder have the right shape already
                present_self_key = present_key_values[2 * layer_idx]
                present_self_value = present_key_values[2 * layer_idx + 1]

                # Check if the present key/value have the expected shape
                if present_self_key.dim() == 4 and present_self_key.shape[2] == 1:
                    # Shape is [batch, heads, 1, head_dim] - this is what we expect
                    past_key_values[4 * layer_idx][:, :, step : step + 1, :] = (
                        present_self_key
                    )
                    past_key_values[4 * layer_idx + 1][:, :, step : step + 1, :] = (
                        present_self_value
                    )
                elif present_self_key.dim() == 3:
                    # Shape is [heads, seq_len, head_dim] - need to reshape
                    # Add batch dimension and transpose to [batch, heads, seq_len, head_dim]
                    present_self_key = present_self_key.unsqueeze(0).transpose(1, 2)
                    present_self_value = present_self_value.unsqueeze(0).transpose(1, 2)
                    past_key_values[4 * layer_idx][:, :, step : step + 1, :] = (
                        present_self_key
                    )
                    past_key_values[4 * layer_idx + 1][:, :, step : step + 1, :] = (
                        present_self_value
                    )
                # Fallback: just use the tensor as is, but only take the last position
                elif present_self_key.shape[-2] == 1:  # Already has seq_len=1
                    past_key_values[4 * layer_idx][:, :, step : step + 1, :] = (
                        present_self_key
                    )
                    past_key_values[4 * layer_idx + 1][:, :, step : step + 1, :] = (
                        present_self_value
                    )
                else:
                    # Take the last position
                    past_key_values[4 * layer_idx][:, :, step : step + 1, :] = (
                        present_self_key[:, :, -1:, :]
                    )
                    past_key_values[4 * layer_idx + 1][:, :, step : step + 1, :] = (
                        present_self_value[:, :, -1:, :]
                    )

        # Decode generated tokens
        if generated_tokens:
            translated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
        else:
            translated_text = ""

        return translated_text.strip()

    def batch_translate(self, texts: list[str], batch_size: int = 1) -> list[str]:
        """
        Translate multiple texts in batches.

        Parameters
        ----------
        texts
            List of input texts to translate
        batch_size
            Number of texts to process in each batch

        Returns
        -------
        translated_texts : list[str]
            List of translated texts
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = []

            for text in batch_texts:
                translated = self._translate_single(text)
                batch_results.append(translated)

            results.extend(batch_results)

        return results

    def interactive_translate(self) -> None:
        """Start an interactive translation session."""
        print(f"OpusMT Interactive Translation ({self.hf_model_id})")
        print("Type 'quit' or 'exit' to stop.")
        print("-" * 50)

        while True:
            try:
                user_input = input("Enter text to translate: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                print("Translating...")
                translated = self._translate_single(user_input)
                print(f"Translation: {translated}")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during translation: {e}")
                print("Please try again.")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns
        -------
        model_info : dict[str, Any]
            Model information including tokenizer details
        """
        return {
            "model_id": self.hf_model_id,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "vocab_size": len(self.tokenizer),
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
            "special_tokens": {
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            },
        }


def create_opus_mt_app(
    encoder: OpusMTEncoder,
    decoder: OpusMTDecoder,
    hf_model_id: str,
    **kwargs: Any,
) -> OpusMTApp:
    """
    Factory function to create an OpusMT application.

    Parameters
    ----------
    encoder
        The OpusMT encoder model
    decoder
        The OpusMT decoder model
    hf_model_id
        HuggingFace model identifier
    **kwargs
        Additional arguments passed to OpusMTApp

    Returns
    -------
    OpusMTApp
        Configured OpusMT application
    """
    return OpusMTApp(encoder, decoder, hf_model_id, **kwargs)
