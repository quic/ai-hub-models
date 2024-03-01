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
import urllib, os
from model import Network


#You can Download from this below link, weight folder(ckpt)
# https://github.com/dut-media-lab/RUAS/tree/main/ckpt

model = Network()
model_dict = torch.load('ckpt/lol.pt',map_location=torch.device('cpu'))
model.load_state_dict(model_dict)
dummy_input=torch.randn(1, 3,480,640)
torch.onnx.export(model, dummy_input, "../ruas.onnx", opset_version=11, verbose=False)

print("ONNX model saved successfully")

