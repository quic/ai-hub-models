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
import model


#You can download the weights from this below link
#https://github.com/Li-Chongyi/Zero-DCE_extension/tree/main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B
#You can put this weight to your appropiate path
scale_factor = 12

DCE_net = model.enhance_net_nopool(scale_factor)
DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth',map_location=torch.device('cpu')))

dummy_input=torch.randn(1, 3, 2160,3840)


torch.onnx.export(DCE_net, dummy_input, "../../zero_dce++.onnx", opset_version=11, verbose=False)

print("ONNX model saved Successfully")
