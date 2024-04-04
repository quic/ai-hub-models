
import torch
from model import Finetunemodel


#You can download the weights from the below link
#https://github.com/vis-opt-group/SCI/tree/main/weights
model = Finetunemodel('weights/difficult.pt')
dummy_input=torch.randn(1, 3, 480,640)

torch.onnx.export(model, dummy_input, "../sci_difficult.onnx", opset_version=11, verbose=False)

print("ONNX Model Saved Successfully")