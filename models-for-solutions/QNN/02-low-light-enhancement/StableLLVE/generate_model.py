import torch
from model import UNet


#You can download the pretrained weight from here
#https://github.com/zkawfanx/StableLLVE/blob/main/checkpoint.pth

model = UNet(n_channels=3, bilinear=True)
model.load_state_dict(torch.load('./checkpoint.pth',map_location=torch.device('cpu')))

dummy_input=torch.randn(1, 3,480,640)

torch.onnx.export(model, dummy_input, "../StableLLVE.onnx", opset_version=11, verbose=False)
print("ONNX model saved Successfully")
