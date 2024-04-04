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
from main import Model
from config import cfg

#you can download the pretrained models from here
#https://github.com/ymmshi/MBLLEN.pytorch/tree/master/pretrained_models

model = Model(cfg['model'])
model.load_state_dict(torch.load('pretrained_models/lowlight.ckpt',map_location=torch.device('cpu'))['state_dict'])
#model = model.cuda()
model.eval()
dummy_input=torch.randn(1, 3, 480,640)
torch.onnx.export(model, dummy_input, "../mbllen.onnx", opset_version=11, verbose=False)
print("ONNX model saved Successfully")