[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# Qualcomm® AI Hub Models

The [Qualcomm® AI Hub Models](https://aihub.qualcomm.com/) are a collection of
state-of-the-art machine learning models optimized for performance (latency,
memory etc.) and ready to deploy on Qualcomm® devices.

* Explore models optimized for on-device deployment of vision, speech, text, and genenrative AI.
* View open-source recipes to quantize, optimize, and deploy these models on-device.
* Browse through [performance metrics](https://aihub.qualcomm.com/models) captured for these models on several devices.
* Access the models through [Hugging Face](https://huggingface.co/qualcomm).
* Check out [sample apps](https://github.com/quic/ai-hub-apps) for on-device deployment of AI Hub models.
* [Sign up](https://myaccount.qualcomm.com/signup) to run these models on hosted Qualcomm® devices.

Supported **python package host machine** Operating Systems:
- Linux (x86, ARM)
- Windows (x86)
- Windows (ARM-- ONLY via x86 Python, not ARM Python)
- MacOS (x86, ARM)

Supported runtimes
* [TensorFlow Lite](https://www.tensorflow.org/lite)
* [Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/artificial-intelligence#overview)
* [ONNX](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)

Models can be deployed on:
* Android
* Windows
* Linux

Supported compute units
* CPU, GPU, NPU (includes [Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor), [HTP](https://developer.qualcomm.com/hardware/qualcomm-innovators-development-kit/ai-resources-overview/ai-hardware-cores-accelerators))

Supported precision
* Floating Points: FP16
* Integer: INT8 (8-bit weight and activation on select models), INT4 (4-bit weight, 16-bit activation on select models)

Supported chipsets
* [Snapdragon 845](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-845-mobile-platform), [Snapdragon 855/855+](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-855-mobile-platform), [Snapdragon 865/865+](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-865-plus-5g-mobile-platform), [Snapdragon 888/888+](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-888-5g-mobile-platform)
* [Snapdragon 8 Gen 1](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-1-mobile-platform), [Snapdragon 8 Gen 2](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-2-mobile-platform), [Snapdragon 8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform), [Snapdragon X Elite](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite)

Select supported devices
* Samsung Galaxy S21 Series, Galaxy S22 Series, Galaxy S23 Series, Galaxy S24 Series
* Xiaomi 12, 13
* Google Pixel 3, 4, 5
* Snapdragon X Elite CRD (Compute Reference Device)

and many more.

## Installation

We currently support **Python >=3.8 and <= 3.10.** We recommend using a Python
virtual environment
([miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) or
[virtualenv](https://virtualenv.pypa.io/en/latest/)).

You can setup a virtualenv using:
```
python -m venv qai_hub_models_env && source qai_hub_models_env/bin/activate
```

Once the environment is setup, you can install the base package using:

```shell
pip install qai_hub_models
```

Some models (e.g. [YOLOv7](https://github.com/WongKinYiu/yolov7)) require
additional dependencies. You can install those dependencies automatically
using:

```shell
pip install "qai_hub_models[yolov7]"
```

## Getting Started

Each model comes with the following set of CLI demos:
* Locally runnable PyTorch based CLI demo to validate the model off device.
* On-device CLI demo that produces a model ready for on-device deployment and runs the model on a hosted Qualcomm® device (needs [sign up](https://myaccount.qualcomm.com/signup)).

All the models produced by these demos are freely available on [Hugging
Face](https://huggingface.co/qualcomm) or through our
[website](https://aihub.qualcomm.com/models). See the individual model readme
files (e.g. [YOLOv7](qai_hub_models/models/yolov7/README.md)) for more
details.

### Local CLI Demo with PyTorch

[All models](#model-directory) contain CLI demos that run the model in
**PyTorch** locally with sample input. Demos are optimized for code clarity
rather than latency, and run exclusively in PyTorch. Optimal model latency can
be achieved with model export via [Qualcomm® AI
Hub](https://www.aihub.qualcomm.com).

```shell
python -m qai_hub_models.models.yolov7.demo
```

For additional details on how to use the demo CLI, use the `--help` option
```shell
python -m qai_hub_models.models.yolov7.demo --help
```

See the [model directory](#model-directory) below to explore all other models.

---

Note that most ML use cases require some pre and post-processing that are not
part of the model itself. A python reference implementation of this is provided
for each model in `app.py`. Apps load & pre-process model input, run model
inference, and post-process model output before returning it to you.

Here is an example of how the PyTorch CLI works for [YOLOv7](https://github.com/WongKinYiu/yolov7):

```python
from PIL import Image
from qai_hub_models.models.yolov7 import Model as YOLOv7Model
from qai_hub_models.models.yolov7 import App as YOLOv7App
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS

# Load pre-trained model
torch_model = YOLOv7Model.from_pretrained()

# Load a simple PyTorch based application
app = YOLOv7App(torch_model)
image = load_image(IMAGE_ADDRESS, "yolov7")

# Perform prediction on a sample image
pred_image = app.predict(image)[0]
Image.fromarray(pred_image).show()

```

### CLI demo to run on hosted Qualcomm® devices

[Some models](#model-directory) contain CLI demos that run the model on a hosted
Qualcomm® device using [Qualcomm® AI Hub](https://aihub.qualcomm.com).

To run the model on a hosted device, [sign up for access to Qualcomm® AI
Hub](https://myaccount.qualcomm.com/signup). Sign-in to Qualcomm® AI Hub with your
Qualcomm® ID. Once signed in navigate to Account -> Settings -> API Token.

With this API token, you can configure your client to run models on the cloud
hosted devices.

```shell
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://app.aihub.qualcomm.com/docs/) for more information.

The on-device CLI demo performs the following:
* Exports the model for on-device execution.
* Profiles the model on-device on a cloud hosted Qualcomm® device.
* Runs the model on-device on a cloud hosted Qualcomm® device and compares accuracy between a local CPU based PyTorch run and the on-device run.
* Downloads models (and other required assets) that can be deployed on-device in an Android application.

```shell
python -m qai_hub_models.models.yolov7.export
```

Many models may have initialization parameters that allow loading custom
weights and checkpoints. See `--help` for more details

```shell
python -m qai_hub_models.models.yolov7.export --help
```

#### How does this export script work?

As described above, the script above compiles, optimizes, and runs the model on
a cloud hosted Qualcomm® device. The demo uses [Qualcomm® AI Hub's Python
APIs](https://app.aihub.qualcomm.com/docs/).

<img src="https://app.aihub.qualcomm.com/docs/_images/How-It-Works.png" alt="Qualcomm® AI Hub explained" width="90%">

Here is a simplified example of code that can be used to run the entire model
on a cloud hosted device:

```python
from typing import Tuple
import torch
import qai_hub as hub
from qai_hub_models.models.yolov7 import Model as YOLOv7Model

# Load YOLOv7 in PyTorch
torch_model = YOLOv7Model.from_pretrained()
torch_model.eval()

# Trace the PyTorch model using one data point of provided sample inputs to
# torch tensor to trace the model.
example_input = [torch.tensor(data[0]) for name, data in torch_model.sample_inputs().items()]
pt_model = torch.jit.trace(torch_model, example_input)

# Select a device
device = hub.Device("Samsung Galaxy S23")

# Compile model for a specific device
compile_job = hub.submit_compile_job(
    model=pt_model,
    device=device,
    input_specs=torch_model.get_input_spec(),
)

# Get target model to run on a cloud hosted device
target_model = compile_job.get_target_model()

# Profile the previously compiled model on a cloud hosted device
profile_job = hub.submit_profile_job(
    model=target_model,
    device=device,
)

# Perform on-device inference on a cloud hosted device
input_data = torch_model.sample_inputs()
inference_job = hub.submit_inference_job(
    model=target_model,
    device=device,
    inputs=input_data,
)

# Returns the output as dict{name: numpy}
on_device_output = inference_job.download_output_data()
```

---

### Working with source code

You can clone the repository using:

```shell
git clone https://github.com/quic/ai-hub-models/blob/main
cd main
pip install -e .
```

Install additional dependencies to prepare a model before using the following:
```shell
cd main
pip install -e ".[yolov7]"
```

All models have accuracy and end-to-end tests when applicable. These tests as
designed to be run locally and verify that the PyTorch code produces correct
results.  To run the tests for a model:
```shell
python -m pytest --pyargs qai_hub_models.models.yolov7.test
```
---

For any issues, please contact us at ai-hub-support@qti.qualcomm.com.


---

### LICENSE

Qualcomm® AI Hub Models is licensed under BSD-3. See the [LICENSE file](../LICENSE).


---

## Model Directory

### Computer Vision

| Model | README | Torch App | Device Export | CLI Demo
| -- | -- | -- | -- | --
| | | | |
| **Image Classification**
| [ConvNext-Tiny](https://aihub.qualcomm.com/models/convnext_tiny) | [qai_hub_models.models.convnext_tiny](qai_hub_models/models/convnext_tiny/README.md) | ✔️ | ✔️ | ✔️
| [ConvNext-Tiny-w8a16-Quantized](https://aihub.qualcomm.com/models/convnext_tiny_w8a16_quantized) | [qai_hub_models.models.convnext_tiny_w8a16_quantized](qai_hub_models/models/convnext_tiny_w8a16_quantized/README.md) | ✔️ | ✔️ | ✔️
| [ConvNext-Tiny-w8a8-Quantized](https://aihub.qualcomm.com/models/convnext_tiny_w8a8_quantized) | [qai_hub_models.models.convnext_tiny_w8a8_quantized](qai_hub_models/models/convnext_tiny_w8a8_quantized/README.md) | ✔️ | ✔️ | ✔️
| [DenseNet-121](https://aihub.qualcomm.com/models/densenet121) | [qai_hub_models.models.densenet121](qai_hub_models/models/densenet121/README.md) | ✔️ | ✔️ | ✔️
| [EfficientNet-B0](https://aihub.qualcomm.com/models/efficientnet_b0) | [qai_hub_models.models.efficientnet_b0](qai_hub_models/models/efficientnet_b0/README.md) | ✔️ | ✔️ | ✔️
| [GoogLeNet](https://aihub.qualcomm.com/models/googlenet) | [qai_hub_models.models.googlenet](qai_hub_models/models/googlenet/README.md) | ✔️ | ✔️ | ✔️
| [GoogLeNetQuantized](https://aihub.qualcomm.com/models/googlenet_quantized) | [qai_hub_models.models.googlenet_quantized](qai_hub_models/models/googlenet_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Inception-v3](https://aihub.qualcomm.com/models/inception_v3) | [qai_hub_models.models.inception_v3](qai_hub_models/models/inception_v3/README.md) | ✔️ | ✔️ | ✔️
| [Inception-v3-Quantized](https://aihub.qualcomm.com/models/inception_v3_quantized) | [qai_hub_models.models.inception_v3_quantized](qai_hub_models/models/inception_v3_quantized/README.md) | ✔️ | ✔️ | ✔️
| [MNASNet05](https://aihub.qualcomm.com/models/mnasnet05) | [qai_hub_models.models.mnasnet05](qai_hub_models/models/mnasnet05/README.md) | ✔️ | ✔️ | ✔️
| [MobileNet-v2](https://aihub.qualcomm.com/models/mobilenet_v2) | [qai_hub_models.models.mobilenet_v2](qai_hub_models/models/mobilenet_v2/README.md) | ✔️ | ✔️ | ✔️
| [MobileNet-v2-Quantized](https://aihub.qualcomm.com/models/mobilenet_v2_quantized) | [qai_hub_models.models.mobilenet_v2_quantized](qai_hub_models/models/mobilenet_v2_quantized/README.md) | ✔️ | ✔️ | ✔️
| [MobileNet-v3-Large](https://aihub.qualcomm.com/models/mobilenet_v3_large) | [qai_hub_models.models.mobilenet_v3_large](qai_hub_models/models/mobilenet_v3_large/README.md) | ✔️ | ✔️ | ✔️
| [MobileNet-v3-Large-Quantized](https://aihub.qualcomm.com/models/mobilenet_v3_large_quantized) | [qai_hub_models.models.mobilenet_v3_large_quantized](qai_hub_models/models/mobilenet_v3_large_quantized/README.md) | ✔️ | ✔️ | ✔️
| [MobileNet-v3-Small](https://aihub.qualcomm.com/models/mobilenet_v3_small) | [qai_hub_models.models.mobilenet_v3_small](qai_hub_models/models/mobilenet_v3_small/README.md) | ✔️ | ✔️ | ✔️
| [RegNet](https://aihub.qualcomm.com/models/regnet) | [qai_hub_models.models.regnet](qai_hub_models/models/regnet/README.md) | ✔️ | ✔️ | ✔️
| [RegNetQuantized](https://aihub.qualcomm.com/models/regnet_quantized) | [qai_hub_models.models.regnet_quantized](qai_hub_models/models/regnet_quantized/README.md) | ✔️ | ✔️ | ✔️
| [ResNeXt101](https://aihub.qualcomm.com/models/resnext101) | [qai_hub_models.models.resnext101](qai_hub_models/models/resnext101/README.md) | ✔️ | ✔️ | ✔️
| [ResNeXt101Quantized](https://aihub.qualcomm.com/models/resnext101_quantized) | [qai_hub_models.models.resnext101_quantized](qai_hub_models/models/resnext101_quantized/README.md) | ✔️ | ✔️ | ✔️
| [ResNeXt50](https://aihub.qualcomm.com/models/resnext50) | [qai_hub_models.models.resnext50](qai_hub_models/models/resnext50/README.md) | ✔️ | ✔️ | ✔️
| [ResNeXt50Quantized](https://aihub.qualcomm.com/models/resnext50_quantized) | [qai_hub_models.models.resnext50_quantized](qai_hub_models/models/resnext50_quantized/README.md) | ✔️ | ✔️ | ✔️
| [ResNet101](https://aihub.qualcomm.com/models/resnet101) | [qai_hub_models.models.resnet101](qai_hub_models/models/resnet101/README.md) | ✔️ | ✔️ | ✔️
| [ResNet101Quantized](https://aihub.qualcomm.com/models/resnet101_quantized) | [qai_hub_models.models.resnet101_quantized](qai_hub_models/models/resnet101_quantized/README.md) | ✔️ | ✔️ | ✔️
| [ResNet18](https://aihub.qualcomm.com/models/resnet18) | [qai_hub_models.models.resnet18](qai_hub_models/models/resnet18/README.md) | ✔️ | ✔️ | ✔️
| [ResNet18Quantized](https://aihub.qualcomm.com/models/resnet18_quantized) | [qai_hub_models.models.resnet18_quantized](qai_hub_models/models/resnet18_quantized/README.md) | ✔️ | ✔️ | ✔️
| [ResNet50](https://aihub.qualcomm.com/models/resnet50) | [qai_hub_models.models.resnet50](qai_hub_models/models/resnet50/README.md) | ✔️ | ✔️ | ✔️
| [ResNet50Quantized](https://aihub.qualcomm.com/models/resnet50_quantized) | [qai_hub_models.models.resnet50_quantized](qai_hub_models/models/resnet50_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Shufflenet-v2](https://aihub.qualcomm.com/models/shufflenet_v2) | [qai_hub_models.models.shufflenet_v2](qai_hub_models/models/shufflenet_v2/README.md) | ✔️ | ✔️ | ✔️
| [Shufflenet-v2Quantized](https://aihub.qualcomm.com/models/shufflenet_v2_quantized) | [qai_hub_models.models.shufflenet_v2_quantized](qai_hub_models/models/shufflenet_v2_quantized/README.md) | ✔️ | ✔️ | ✔️
| [SqueezeNet-1_1](https://aihub.qualcomm.com/models/squeezenet1_1) | [qai_hub_models.models.squeezenet1_1](qai_hub_models/models/squeezenet1_1/README.md) | ✔️ | ✔️ | ✔️
| [SqueezeNet-1_1Quantized](https://aihub.qualcomm.com/models/squeezenet1_1_quantized) | [qai_hub_models.models.squeezenet1_1_quantized](qai_hub_models/models/squeezenet1_1_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Swin-Base](https://aihub.qualcomm.com/models/swin_base) | [qai_hub_models.models.swin_base](qai_hub_models/models/swin_base/README.md) | ✔️ | ✔️ | ✔️
| [Swin-Small](https://aihub.qualcomm.com/models/swin_small) | [qai_hub_models.models.swin_small](qai_hub_models/models/swin_small/README.md) | ✔️ | ✔️ | ✔️
| [Swin-Tiny](https://aihub.qualcomm.com/models/swin_tiny) | [qai_hub_models.models.swin_tiny](qai_hub_models/models/swin_tiny/README.md) | ✔️ | ✔️ | ✔️
| [VIT](https://aihub.qualcomm.com/models/vit) | [qai_hub_models.models.vit](qai_hub_models/models/vit/README.md) | ✔️ | ✔️ | ✔️
| [WideResNet50](https://aihub.qualcomm.com/models/wideresnet50) | [qai_hub_models.models.wideresnet50](qai_hub_models/models/wideresnet50/README.md) | ✔️ | ✔️ | ✔️
| [WideResNet50-Quantized](https://aihub.qualcomm.com/models/wideresnet50_quantized) | [qai_hub_models.models.wideresnet50_quantized](qai_hub_models/models/wideresnet50_quantized/README.md) | ✔️ | ✔️ | ✔️
| | | | |
| **Image Editing**
| [AOT-GAN](https://aihub.qualcomm.com/models/aotgan) | [qai_hub_models.models.aotgan](qai_hub_models/models/aotgan/README.md) | ✔️ | ✔️ | ✔️
| [LaMa-Dilated](https://aihub.qualcomm.com/models/lama_dilated) | [qai_hub_models.models.lama_dilated](qai_hub_models/models/lama_dilated/README.md) | ✔️ | ✔️ | ✔️
| | | | |
| **Super Resolution**
| [ESRGAN](https://aihub.qualcomm.com/models/esrgan) | [qai_hub_models.models.esrgan](qai_hub_models/models/esrgan/README.md) | ✔️ | ✔️ | ✔️
| [QuickSRNetLarge](https://aihub.qualcomm.com/models/quicksrnetlarge) | [qai_hub_models.models.quicksrnetlarge](qai_hub_models/models/quicksrnetlarge/README.md) | ✔️ | ✔️ | ✔️
| [QuickSRNetLarge-Quantized](https://aihub.qualcomm.com/models/quicksrnetlarge_quantized) | [qai_hub_models.models.quicksrnetlarge_quantized](qai_hub_models/models/quicksrnetlarge_quantized/README.md) | ✔️ | ✔️ | ✔️
| [QuickSRNetMedium](https://aihub.qualcomm.com/models/quicksrnetmedium) | [qai_hub_models.models.quicksrnetmedium](qai_hub_models/models/quicksrnetmedium/README.md) | ✔️ | ✔️ | ✔️
| [QuickSRNetMedium-Quantized](https://aihub.qualcomm.com/models/quicksrnetmedium_quantized) | [qai_hub_models.models.quicksrnetmedium_quantized](qai_hub_models/models/quicksrnetmedium_quantized/README.md) | ✔️ | ✔️ | ✔️
| [QuickSRNetSmall](https://aihub.qualcomm.com/models/quicksrnetsmall) | [qai_hub_models.models.quicksrnetsmall](qai_hub_models/models/quicksrnetsmall/README.md) | ✔️ | ✔️ | ✔️
| [QuickSRNetSmall-Quantized](https://aihub.qualcomm.com/models/quicksrnetsmall_quantized) | [qai_hub_models.models.quicksrnetsmall_quantized](qai_hub_models/models/quicksrnetsmall_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Real-ESRGAN-General-x4v3](https://aihub.qualcomm.com/models/real_esrgan_general_x4v3) | [qai_hub_models.models.real_esrgan_general_x4v3](qai_hub_models/models/real_esrgan_general_x4v3/README.md) | ✔️ | ✔️ | ✔️
| [Real-ESRGAN-x4plus](https://aihub.qualcomm.com/models/real_esrgan_x4plus) | [qai_hub_models.models.real_esrgan_x4plus](qai_hub_models/models/real_esrgan_x4plus/README.md) | ✔️ | ✔️ | ✔️
| [SESR-M5](https://aihub.qualcomm.com/models/sesr_m5) | [qai_hub_models.models.sesr_m5](qai_hub_models/models/sesr_m5/README.md) | ✔️ | ✔️ | ✔️
| [SESR-M5-Quantized](https://aihub.qualcomm.com/models/sesr_m5_quantized) | [qai_hub_models.models.sesr_m5_quantized](qai_hub_models/models/sesr_m5_quantized/README.md) | ✔️ | ✔️ | ✔️
| [XLSR](https://aihub.qualcomm.com/models/xlsr) | [qai_hub_models.models.xlsr](qai_hub_models/models/xlsr/README.md) | ✔️ | ✔️ | ✔️
| [XLSR-Quantized](https://aihub.qualcomm.com/models/xlsr_quantized) | [qai_hub_models.models.xlsr_quantized](qai_hub_models/models/xlsr_quantized/README.md) | ✔️ | ✔️ | ✔️
| | | | |
| **Semantic Segmentation**
| [DDRNet23-Slim](https://aihub.qualcomm.com/models/ddrnet23_slim) | [qai_hub_models.models.ddrnet23_slim](qai_hub_models/models/ddrnet23_slim/README.md) | ✔️ | ✔️ | ✔️
| [DeepLabV3-Plus-MobileNet](https://aihub.qualcomm.com/models/deeplabv3_plus_mobilenet) | [qai_hub_models.models.deeplabv3_plus_mobilenet](qai_hub_models/models/deeplabv3_plus_mobilenet/README.md) | ✔️ | ✔️ | ✔️
| [DeepLabV3-Plus-MobileNet-Quantized](https://aihub.qualcomm.com/models/deeplabv3_plus_mobilenet_quantized) | [qai_hub_models.models.deeplabv3_plus_mobilenet_quantized](qai_hub_models/models/deeplabv3_plus_mobilenet_quantized/README.md) | ✔️ | ✔️ | ✔️
| [DeepLabV3-ResNet50](https://aihub.qualcomm.com/models/deeplabv3_resnet50) | [qai_hub_models.models.deeplabv3_resnet50](qai_hub_models/models/deeplabv3_resnet50/README.md) | ✔️ | ✔️ | ✔️
| [FCN-ResNet50](https://aihub.qualcomm.com/models/fcn_resnet50) | [qai_hub_models.models.fcn_resnet50](qai_hub_models/models/fcn_resnet50/README.md) | ✔️ | ✔️ | ✔️
| [FCN-ResNet50-Quantized](https://aihub.qualcomm.com/models/fcn_resnet50_quantized) | [qai_hub_models.models.fcn_resnet50_quantized](qai_hub_models/models/fcn_resnet50_quantized/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-122NS-LowRes](https://aihub.qualcomm.com/models/ffnet_122ns_lowres) | [qai_hub_models.models.ffnet_122ns_lowres](qai_hub_models/models/ffnet_122ns_lowres/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-40S](https://aihub.qualcomm.com/models/ffnet_40s) | [qai_hub_models.models.ffnet_40s](qai_hub_models/models/ffnet_40s/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-40S-Quantized](https://aihub.qualcomm.com/models/ffnet_40s_quantized) | [qai_hub_models.models.ffnet_40s_quantized](qai_hub_models/models/ffnet_40s_quantized/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-54S](https://aihub.qualcomm.com/models/ffnet_54s) | [qai_hub_models.models.ffnet_54s](qai_hub_models/models/ffnet_54s/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-54S-Quantized](https://aihub.qualcomm.com/models/ffnet_54s_quantized) | [qai_hub_models.models.ffnet_54s_quantized](qai_hub_models/models/ffnet_54s_quantized/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-78S](https://aihub.qualcomm.com/models/ffnet_78s) | [qai_hub_models.models.ffnet_78s](qai_hub_models/models/ffnet_78s/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-78S-LowRes](https://aihub.qualcomm.com/models/ffnet_78s_lowres) | [qai_hub_models.models.ffnet_78s_lowres](qai_hub_models/models/ffnet_78s_lowres/README.md) | ✔️ | ✔️ | ✔️
| [FFNet-78S-Quantized](https://aihub.qualcomm.com/models/ffnet_78s_quantized) | [qai_hub_models.models.ffnet_78s_quantized](qai_hub_models/models/ffnet_78s_quantized/README.md) | ✔️ | ✔️ | ✔️
| [FastSam-S](https://aihub.qualcomm.com/models/fastsam_s) | [qai_hub_models.models.fastsam_s](qai_hub_models/models/fastsam_s/README.md) | ✔️ | ✔️ | ✔️
| [FastSam-X](https://aihub.qualcomm.com/models/fastsam_x) | [qai_hub_models.models.fastsam_x](qai_hub_models/models/fastsam_x/README.md) | ✔️ | ✔️ | ✔️
| [MediaPipe-Selfie-Segmentation](https://aihub.qualcomm.com/models/mediapipe_selfie) | [qai_hub_models.models.mediapipe_selfie](qai_hub_models/models/mediapipe_selfie/README.md) | ✔️ | ✔️ | ✔️
| [SINet](https://aihub.qualcomm.com/models/sinet) | [qai_hub_models.models.sinet](qai_hub_models/models/sinet/README.md) | ✔️ | ✔️ | ✔️
| [Segment-Anything-Model](https://aihub.qualcomm.com/models/sam) | [qai_hub_models.models.sam](qai_hub_models/models/sam/README.md) | ✔️ | ✔️ | ✔️
| [Unet-Segmentation](https://aihub.qualcomm.com/models/unet_segmentation) | [qai_hub_models.models.unet_segmentation](qai_hub_models/models/unet_segmentation/README.md) | ✔️ | ✔️ | ✔️
| [YOLOv8-Segmentation](https://aihub.qualcomm.com/models/yolov8_seg) | [qai_hub_models.models.yolov8_seg](qai_hub_models/models/yolov8_seg/README.md) | ✔️ | ✔️ | ✔️
| | | | |
| **Object Detection**
| [DETR-ResNet101](https://aihub.qualcomm.com/models/detr_resnet101) | [qai_hub_models.models.detr_resnet101](qai_hub_models/models/detr_resnet101/README.md) | ✔️ | ✔️ | ✔️
| [DETR-ResNet101-DC5](https://aihub.qualcomm.com/models/detr_resnet101_dc5) | [qai_hub_models.models.detr_resnet101_dc5](qai_hub_models/models/detr_resnet101_dc5/README.md) | ✔️ | ✔️ | ✔️
| [DETR-ResNet50](https://aihub.qualcomm.com/models/detr_resnet50) | [qai_hub_models.models.detr_resnet50](qai_hub_models/models/detr_resnet50/README.md) | ✔️ | ✔️ | ✔️
| [DETR-ResNet50-DC5](https://aihub.qualcomm.com/models/detr_resnet50_dc5) | [qai_hub_models.models.detr_resnet50_dc5](qai_hub_models/models/detr_resnet50_dc5/README.md) | ✔️ | ✔️ | ✔️
| [MediaPipe-Face-Detection](https://aihub.qualcomm.com/models/mediapipe_face) | [qai_hub_models.models.mediapipe_face](qai_hub_models/models/mediapipe_face/README.md) | ✔️ | ✔️ | ✔️
| [MediaPipe-Face-Detection-Quantized](https://aihub.qualcomm.com/models/mediapipe_face_quantized) | [qai_hub_models.models.mediapipe_face_quantized](qai_hub_models/models/mediapipe_face_quantized/README.md) | ✔️ | ✔️ | ✔️
| [MediaPipe-Hand-Detection](https://aihub.qualcomm.com/models/mediapipe_hand) | [qai_hub_models.models.mediapipe_hand](qai_hub_models/models/mediapipe_hand/README.md) | ✔️ | ✔️ | ✔️
| [YOLOv11-Detection](qai_hub_models/models/yolov11_det/README.md) | [qai_hub_models.models.yolov11_det](qai_hub_models/models/yolov11_det/README.md) | ✔️ | ✔️ | ✔️
| [YOLOv8-Detection](https://aihub.qualcomm.com/models/yolov8_det) | [qai_hub_models.models.yolov8_det](qai_hub_models/models/yolov8_det/README.md) | ✔️ | ✔️ | ✔️
| [YOLOv8-Detection-Quantized](https://aihub.qualcomm.com/models/yolov8_det_quantized) | [qai_hub_models.models.yolov8_det_quantized](qai_hub_models/models/yolov8_det_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Yolo-NAS](https://aihub.qualcomm.com/models/yolonas) | [qai_hub_models.models.yolonas](qai_hub_models/models/yolonas/README.md) | ✔️ | ✔️ | ✔️
| [Yolo-NAS-Quantized](https://aihub.qualcomm.com/models/yolonas_quantized) | [qai_hub_models.models.yolonas_quantized](qai_hub_models/models/yolonas_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Yolo-v6](https://aihub.qualcomm.com/models/yolov6) | [qai_hub_models.models.yolov6](qai_hub_models/models/yolov6/README.md) | ✔️ | ✔️ | ✔️
| [Yolo-v7](https://aihub.qualcomm.com/models/yolov7) | [qai_hub_models.models.yolov7](qai_hub_models/models/yolov7/README.md) | ✔️ | ✔️ | ✔️
| [Yolo-v7-Quantized](https://aihub.qualcomm.com/models/yolov7_quantized) | [qai_hub_models.models.yolov7_quantized](qai_hub_models/models/yolov7_quantized/README.md) | ✔️ | ✔️ | ✔️
| | | | |
| **Pose Estimation**
| [FaceMap_3DMM](qai_hub_models/models/facemap_3dmm/README.md) | [qai_hub_models.models.facemap_3dmm](qai_hub_models/models/facemap_3dmm/README.md) | ✔️ | ✔️ | ✔️
| [HRNetPose](https://aihub.qualcomm.com/models/hrnet_pose) | [qai_hub_models.models.hrnet_pose](qai_hub_models/models/hrnet_pose/README.md) | ✔️ | ✔️ | ✔️
| [HRNetPoseQuantized](https://aihub.qualcomm.com/models/hrnet_pose_quantized) | [qai_hub_models.models.hrnet_pose_quantized](qai_hub_models/models/hrnet_pose_quantized/README.md) | ✔️ | ✔️ | ✔️
| [LiteHRNet](https://aihub.qualcomm.com/models/litehrnet) | [qai_hub_models.models.litehrnet](qai_hub_models/models/litehrnet/README.md) | ✔️ | ✔️ | ✔️
| [MediaPipe-Pose-Estimation](https://aihub.qualcomm.com/models/mediapipe_pose) | [qai_hub_models.models.mediapipe_pose](qai_hub_models/models/mediapipe_pose/README.md) | ✔️ | ✔️ | ✔️
| [OpenPose](https://aihub.qualcomm.com/models/openpose) | [qai_hub_models.models.openpose](qai_hub_models/models/openpose/README.md) | ✔️ | ✔️ | ✔️
| [Posenet-Mobilenet](https://aihub.qualcomm.com/models/posenet_mobilenet) | [qai_hub_models.models.posenet_mobilenet](qai_hub_models/models/posenet_mobilenet/README.md) | ✔️ | ✔️ | ✔️
| [Posenet-Mobilenet-Quantized](https://aihub.qualcomm.com/models/posenet_mobilenet_quantized) | [qai_hub_models.models.posenet_mobilenet_quantized](qai_hub_models/models/posenet_mobilenet_quantized/README.md) | ✔️ | ✔️ | ✔️
| | | | |
| **Depth Estimation**
| [Midas-V2](https://aihub.qualcomm.com/models/midas) | [qai_hub_models.models.midas](qai_hub_models/models/midas/README.md) | ✔️ | ✔️ | ✔️
| [Midas-V2-Quantized](https://aihub.qualcomm.com/models/midas_quantized) | [qai_hub_models.models.midas_quantized](qai_hub_models/models/midas_quantized/README.md) | ✔️ | ✔️ | ✔️

### Audio

| Model | README | Torch App | Device Export | CLI Demo
| -- | -- | -- | -- | --
| | | | |
| **Speech Recognition**
| [HuggingFace-WavLM-Base-Plus](https://aihub.qualcomm.com/models/huggingface_wavlm_base_plus) | [qai_hub_models.models.huggingface_wavlm_base_plus](qai_hub_models/models/huggingface_wavlm_base_plus/README.md) | ✔️ | ✔️ | ✔️
| [Whisper-Base-En](https://aihub.qualcomm.com/models/whisper_base_en) | [qai_hub_models.models.whisper_base_en](qai_hub_models/models/whisper_base_en/README.md) | ✔️ | ✔️ | ✔️
| [Whisper-Small-En](https://aihub.qualcomm.com/models/whisper_small_en) | [qai_hub_models.models.whisper_small_en](qai_hub_models/models/whisper_small_en/README.md) | ✔️ | ✔️ | ✔️
| [Whisper-Tiny-En](https://aihub.qualcomm.com/models/whisper_tiny_en) | [qai_hub_models.models.whisper_tiny_en](qai_hub_models/models/whisper_tiny_en/README.md) | ✔️ | ✔️ | ✔️

### Multimodal

| Model | README | Torch App | Device Export | CLI Demo
| -- | -- | -- | -- | --
| | | | |
| [TrOCR](https://aihub.qualcomm.com/models/trocr) | [qai_hub_models.models.trocr](qai_hub_models/models/trocr/README.md) | ✔️ | ✔️ | ✔️
| [OpenAI-Clip](https://aihub.qualcomm.com/models/openai_clip) | [qai_hub_models.models.openai_clip](qai_hub_models/models/openai_clip/README.md) | ✔️ | ✔️ | ✔️

### Generative Ai

| Model | README | Torch App | Device Export | CLI Demo
| -- | -- | -- | -- | --
| | | | |
| **Image Generation**
| [ControlNet](https://aihub.qualcomm.com/models/controlnet_quantized) | [qai_hub_models.models.controlnet_quantized](qai_hub_models/models/controlnet_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Riffusion](https://aihub.qualcomm.com/models/riffusion_quantized) | [qai_hub_models.models.riffusion_quantized](qai_hub_models/models/riffusion_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Stable-Diffusion-v1.5](https://aihub.qualcomm.com/models/stable_diffusion_v1_5_quantized) | [qai_hub_models.models.stable_diffusion_v1_5_quantized](qai_hub_models/models/stable_diffusion_v1_5_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Stable-Diffusion-v2.1](https://aihub.qualcomm.com/models/stable_diffusion_v2_1_quantized) | [qai_hub_models.models.stable_diffusion_v2_1_quantized](qai_hub_models/models/stable_diffusion_v2_1_quantized/README.md) | ✔️ | ✔️ | ✔️
| | | | |
| **Text Generation**
| [Baichuan-7B](https://aihub.qualcomm.com/models/baichuan_7b_quantized) | [qai_hub_models.models.baichuan_7b_quantized](qai_hub_models/models/baichuan_7b_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Llama-v2-7B-Chat](https://aihub.qualcomm.com/models/llama_v2_7b_chat_quantized) | [qai_hub_models.models.llama_v2_7b_chat_quantized](qai_hub_models/models/llama_v2_7b_chat_quantized/README.md) | ✔️ | ✔️ | ✔️
| [Llama-v3-8B-Chat](https://aihub.qualcomm.com/models/llama_v3_8b_chat_quantized) | [qai_hub_models.models.llama_v3_8b_chat_quantized](qai_hub_models/models/llama_v3_8b_chat_quantized/README.md) | ✔️ | ✔️ | ✔️
