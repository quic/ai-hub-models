# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to evaluate PPL of Llama 3.2 model.
Packages to install: llama_v3_2_3b_chat_quantized
"""
from torch.utils.data import DataLoader

from qai_hub_models.datasets.wikitext import WikiText, collate_fn
from qai_hub_models.evaluators.ppl_evaluator import PerplexityEvaluator
from qai_hub_models.models._shared.llama3.model import RopeEmbedding
from qai_hub_models.models.llama_v3_2_3b_chat_quantized.model import Llama3_2_Quantized

if __name__ == "__main__":
    sequence_length = 2048
    context_length = 4096

    # Load model
    model = Llama3_2_Quantized.from_pretrained(
        sequence_length=sequence_length, context_length=context_length
    )
    rope_embeddings = RopeEmbedding(
        max_length=model.context_length, config=model.llm_config
    )
    input_specs = model.get_input_spec(
        sequence_length=model.sequence_length,
        context_length=model.context_length,
    )

    # Pass KV cache shape
    past_key_values = []
    for k, (shape, _) in input_specs.items():
        if k.startswith("past_"):
            past_key_values.append(shape)

    # Load dataset.
    dataset = WikiText(
        model.tokenizer,
        rope_embeddings,
        block_size=sequence_length,
        context_length=context_length,
    )

    # Pass it to data loader
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

    # Instantiate the evaluator
    evaluator = PerplexityEvaluator(past_key_values, context_length, sequence_length)

    # Pass batches of data through the model.
    evaluator.add_from_dataset(model=model, data=dataloader, eval_iterations=4)
    print(evaluator.formatted_accuracy())

    evaluator.plot_ppl_per_batch(output_dir=".")
