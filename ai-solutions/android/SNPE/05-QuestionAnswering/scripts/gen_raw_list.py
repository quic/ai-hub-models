# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
# <input_layer_name>:=<input_layer_path>[<space><input_layer_name>:=<input_layer_path>]

import sys

if len(sys.argv) != 2:
    print("Usage : python gen_raw_list.py <no. of iterations>")
    sys.exit()

total_iter = int(sys.argv[1])
print("Generating input_list \"small_raw_list.txt\" with {} iterations".format(total_iter))

with open("tf_raw_list.txt",'w') as f:
    for i in range(total_iter):
        f.write("input_ids:=input_ids/input_ids__{0:03}_.raw attention_mask:=attention_mask/attention_mask__{0:003}_.raw token_type_ids:=token_type_ids/token_type_ids__{0:003}_.raw\n".format(i,i,i))

with open("snpe_raw_list.txt",'w') as f:
    f.write("%Identity:0 Identity_1:0\n")
    for i in range(total_iter):
        f.write("input_ids:0:=input_ids/input_ids__{0:03}_.raw attention_mask:0:=attention_mask/attention_mask__{0:003}_.raw token_type_ids:0:=token_type_ids/token_type_ids__{0:003}_.raw\n".format(i,i,i))
