# Super Image Resolution with SRGAN

| Field | Description |
| --- | --- |
| Model Name | SRGAN |
| DNN Framwork | PyTorch |
| Public Repo  | https://github.com/quic/aimet-model-zoo/ |
| Paper        | NA |
| Accuracy Metric | PSNR |
| Input Resolution | 128 x 128 |
| Output Resolution | 512 x 512 |
| Pre-Process | <ol><li>Read the input [BGR ,LQ(128 x 128)] as a numpy array</li><li>Expand input dimensions i.e., add Channel(C)</li><li>BGR -> RGB</li><li>HWC -> CWH</li><li>numpy -> tensor</li><li>Export the tensor to the model</li></ol> |
| post-Process| <ol><li>Read the input tensor [ RGB, 512 x 512 x 3] , i.e., flattened array</li><li>Reshape the Input to 512 x 512 x 3 array</li><li>numpy -> tensor</li><li>Do Min(0),Max(1) Clamping</li><li>tensor -> numpy</li><li>Multiply the the array with 255.0 and round it off to the nearest Integer</li><li>RGB -> BGR</li><li>Export the array as UINT8</li></ol> |

## Pre-requisites

- Setup AI SDK <a href="https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct"> Qualcomm® AI Engine Direct (Linux)</a>. 

- Follow the insturctions given in SDK to setup the SDK 

-  Please follow the instructions for setting up  Qualcomm® AI Engine Direct using the [link] (https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/setup.html) provided. 

- Install onnx v1.6.0 ```pip install onnx==1.6.0```.

- Install onnxsim ```pip install onnxsim``` and onnxruntime ```pip install onnxruntime```.

- Install OpenCV ```pip install cv2```

- Install mxnet ```pip install mxnet```



## Modify srgan_quanteval.py to get model ?

1. In mmsr/codes folder create a ```__init__.py ``` file.

2. Copy the ```srgan_quanteval.py``` from ```aimet_zoo_torch/srgan/evaluators/``` into your ```mmsr``` directory
3. In the ```mmsr``` repo , go to ```srgan_quanteval.py``` file and do the following changes:
4. Comment imports
```python
# from aimet_torch import quantsim

# from aimet_zoo_torch.common.utils import utils


```
5. Replace the main function of srgan_quanteval.py to this code
```Python
def main(args):
    """Evaluation main script"""

    # Adding hardcoded values to config on top of args
    config = ModelConfig(args)

    download_weights()
    print("download complete!")

    print("configuration complete!")

    print(f"Parsing file {config.yml}...")
    opt = option.parse(config.yml, is_train=False)
    opt = option.dict_to_nonedict(opt)

    print("Loading test images...") # comment

    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

    model = create_model(opt)
    generator = model.netG.module
    generator.eval()
    dummy_input = torch.randn(1,3, 128, 128).type(torch.FloatTensor).to('cpu')
    torch.onnx.export(generator, dummy_input, "srgan.onnx",opset_version=11)


```

6. Replace the ModelConfig  function of srgan_quanteval.py to this code
```python
   class ModelConfig:
    """Adding hardcoded values into args from parseargs() and return config object"""

    def __init__(self, args):
        self.yml = "./codes/options/test/test_SRGAN.yml"
        self.quant_scheme = "tf_enhanced"
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
```
 

## follow notebook for further steps


## Convert Model to library file / serialized binary

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary

## Quantization of model

Please refer to notebook for detailed steps to converting pre-trained model to library file / serialized binary


## Make Inference, Verify output. 

Please refer to notebook for detailed steps to making inference, verifying model output

###### *Snapdragon and Qualcomm AI Engine Direct are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*

