# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
import tensorflow as tf

from transformers import TensorType
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import sys

bs = 1
SEQ_LEN = 384
MODEL_NAME = "mrm8488/electra-small-finetuned-squadv2"

# Allocate tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, from_pt=True)

def model_fn(input_ids, attention_mask, token_type_ids):
    output = model(input_ids, attention_mask, token_type_ids)
    return (output.start_logits, output.end_logits)

model_fn = tf.function(
    model_fn,
    input_signature=[
        tf.TensorSpec(shape=[bs, SEQ_LEN], dtype=tf.int32) ,
        tf.TensorSpec(shape=[bs, SEQ_LEN], dtype=tf.int32), 
        tf.TensorSpec(shape=[bs, SEQ_LEN], dtype=tf.int32)
    ]
)

# Sample input
context = "I'm on highway to Paradip"
question = "Where am I ?"

input_encodings = tokenizer(
            question,
            context,
            return_tensors=TensorType.TENSORFLOW,
            # return_tensors="np",
            padding='max_length',
            return_length=True,
            max_length=SEQ_LEN,
            return_special_tokens_mask=True
        )
# print(input_encodings)

print(f"\nContext = \n{context}")
print(f"\nQ. > {question}")
start_logits, end_logits = model_fn(input_encodings.input_ids, input_encodings.attention_mask , input_encodings.token_type_ids)

answer_start_index = int(tf.math.argmax(start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(end_logits, axis=-1)[0])

predict_answer_tokens = input_encodings.input_ids[0, answer_start_index : answer_end_index + 1]
ans = tokenizer.decode(predict_answer_tokens)
print(f"Prediction: {ans}\n")

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(model_fn.get_concrete_function())

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("NO. of Frozen model layers: {}".format(len(layers)))

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

graph_def = frozen_func.graph.as_graph_def()

graph_def = tf.compat.v1.graph_util.remove_training_nodes(graph_def)

tf.io.write_graph(graph_or_graph_def=graph_def,
                  logdir="./frozen_models",
                  name="electra_small_squad2.pb",
                  as_text=False)

