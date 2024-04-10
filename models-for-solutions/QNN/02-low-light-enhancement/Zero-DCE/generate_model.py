import torch
import model


#You can download the weights from here
# https://github.com/Li-Chongyi/Zero-DCE/tree/master/Zero-DCE_code/snapshots
 

DCE_net = model.enhance_net_nopool()
DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth',map_location=torch.device('cpu')))
DCE_net.eval()

dummy_input=torch.randn(1, 3, 480,640)
torch.onnx.export(DCE_net, dummy_input, "../../zero_dce.onnx", opset_version=11, verbose=False)
print("ONNX model saved Successfully")




		

