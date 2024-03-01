# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import os
import shutil
#print(torch.hub.list('facebookresearch/detr'))
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
dummy_input=torch.randn(1, 3, 800, 1066)
if os.path.exists('models')==False:
    os.mkdir('models')
else:
    shutil.rmtree('models')
    os.mkdir('models')
    print("Folder Already exists")
torch.onnx.export(model, dummy_input, "models/detr_resnet101.onnx", opset_version=11, verbose=False)
print("ONNX Model saved Successfully")
command1="snpe-onnx-to-dlc --input_network models/detr_resnet101.onnx --output_path models/detr_resnet101_fp32.dlc"
os.system(command1)
command2="snpe-dlc-quantize --input_dlc models/detr_resnet101_fp32.dlc --input_list list.txt  --output_dlc models/detr_resnet101_w8a8.dlc --enable_htp"
os.system(command2)

