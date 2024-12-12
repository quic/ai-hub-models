[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](https://aihub.qualcomm.com)

# [Qualcomm® AI Hub Models](https://aihub.qualcomm.com/)

[![Release](https://img.shields.io/github/v/release/quic/ai-hub-models)](https://github.com/quic/ai-hub-models/releases/latest)
[![Tag](https://img.shields.io/github/v/tag/quic/ai-hub-models)](https://github.com/quic/ai-hub-models/releases/latest)
[![PyPi](https://img.shields.io/pypi/v/qai-hub-models)](https://pypi.org/project/qai-hub-models/)
![Python 3.9, 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.9%2C%203.10%20(Recommended)%2C%203.11%2C%203.12-yellow)

The Qualcomm® AI Hub Models are a collection of
state-of-the-art machine learning models optimized for deployment on Qualcomm® devices.

* [List of Models by Category](#model-directory)
* [On-Device Performance Data](https://aihub.qualcomm.com/models)
* [Device-Native Sample Apps](https://github.com/quic/ai-hub-apps)

See supported: [On-Device Runtimes](#on-device-runtimes), [Hardware Targets & Precision](#device-hardware--precision), [Chipsets](#chipsets), [Devices](#devices)

&nbsp;

## Setup

### 1. Install Python Package

The package is available via pip:

```shell
# NOTE for Snapdragon X Elite users:
# Only AMDx64 (64-bit) Python in supported on Windows.
# Installation will fail when using Windows ARM64 Python.

pip install qai_hub_models
```

Some models (e.g. [YOLOv7](https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/yolov7)) require
additional dependencies that can be installed as follows:

```shell
pip install "qai_hub_models[yolov7]"
```

&nbsp;

### 2. Configure AI Hub Access

Many features of AI Hub Models _(such as model compilation, on-device profiling, etc.)_ require access to Qualcomm® AI Hub:

-  [Create a Qualcomm® ID](https://myaccount.qualcomm.com/signup), and use it to [login to Qualcomm® AI Hub](https://app.aihub.qualcomm.com/).
-  Configure your [API token](https://app.aihub.qualcomm.com/account/): `qai-hub configure --api_token API_TOKEN`

&nbsp;

## Getting Started

### Export and Run A Model on a Physical Device

All [models in our directory](#model-directory) can be compiled and profiled on a hosted
Qualcomm® device:

```shell
pip install "qai_hub_models[yolov7]"

python -m qai_hub_models.models.yolov7.export [--target-runtime ...] [--device ...] [--help]
```

_Using Qualcomm® AI Hub_, the export script will:

1. **Compile** the model for the chosen device and target runtime (see: [Compiling Models on AI Hub](https://app.aihub.qualcomm.com/docs/hub/compile_examples.html)).
2. If applicable, **Quantize** the model (see: [Quantization on AI Hub](https://app.aihub.qualcomm.com/docs/hub/quantize_examples.html))
3. **Profile** the compiled model on a real device in the cloud (see: [Profiling Models on AI Hub](https://app.aihub.qualcomm.com/docs/hub/profile_examples.html)).
4. **Run inference** with a sample input data on a real device in the cloud, and compare on-device model output with PyTorch output (see: [Running Inference on AI Hub](https://app.aihub.qualcomm.com/docs/hub/inference_examples.html))
5. **Download** the compiled model to disk.

&nbsp;

### End-To-End Model Demos

Most [models in our directory](#model-directory) contain CLI demos that run the model _end-to-end_:

```shell
pip install "qai_hub_models[yolov7]"
# Predict and draw bounding boxes on the provided image
python -m qai_hub_models.models.yolov7.demo [--image ...] [--on-device] [--help]
```

_End-to-end_ demos:
1. **Preprocess** human-readable input into model input
2. Run **model inference**
3. **Postprocess** model output to a human-readable format

**Many end-to-end demos use AI Hub to run inference on a real cloud-hosted device** _(if the `--on-device` flag is set)_. All end-to-end demos also run locally via PyTorch.

&nbsp;

### Sample Applications

**Native** applications that can run our models (with pre- and post-processing) on physical devices are published in the [AI Hub Apps repository](https://github.com/quic/ai-hub-apps/).

**Python** applications are defined for all models [(from qai_hub_models.models.\<model_name> import App)](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/yolov7/app.py). These apps wrap model inference with pre- and post-processing steps written using torch & numpy. **These apps are optimized to be an easy-to-follow example, rather than to minimize prediction time.**

&nbsp;

## Model Support Data

### On-Device Runtimes

| Runtime | Supported OS |
| -- | -- |
| [Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/artificial-intelligence#overview) | Android, Linux, Windows
| [LiteRT (TensorFlow Lite)](https://www.tensorflow.org/lite) | Android, Linux
| [ONNX](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) | Android, Linux, Windows

### Device Hardware & Precision

| Device Compute Unit | Supported Precision |
| -- | -- |
| CPU | FP32, INT16, INT8
| GPU | FP32, FP16
| NPU (includes [Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor), [HTP](https://developer.qualcomm.com/hardware/qualcomm-innovators-development-kit/ai-resources-overview/ai-hardware-cores-accelerators)) | FP16*, INT16, INT8

*Some older chipsets do not support fp16 inference on their NPU.

### Chipsets
* Snapdragon [8 Elite](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-elite-mobile-platform), [8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform), [8 Gen 2](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-2-mobile-platform), and [8 Gen 1](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-1-mobile-platform) Mobile Platforms
* [Snapdragon X Elite](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite) Compute Platform
* SA8255P, SA8295P, SA8650P, and SA8775P Automotive Platforms
* [QCS 6490](https://www.qualcomm.com/products/internet-of-things/industrial/building-enterprise/qcs6490),  [QCS 8250](https://www.qualcomm.com/products/internet-of-things/consumer/cameras/qcs8250), and [QCS 8550](https://www.qualcomm.com/products/technology/processors/qcs8550) IoT Platforms
* QCS8450 XR Platform

and many more.

### Devices
* Samsung Galaxy S21, S22, S23, and S24 Series
* Xiaomi 12 and 13
* Snapdragon X Elite CRD (Compute Reference Device)
* Qualcomm RB3 Gen 2, RB5

and many more.

&nbsp;

## Model Directory

### Computer Vision

| Model | README |
| -- | -- |
| | |
| **Image Classification**
| [Beit](https://aihub.qualcomm.com/models/beit) | [qai_hub_models.models.beit](qai_hub_models/models/beit/README.md) |
| [ConvNext-Base](https://aihub.qualcomm.com/models/convnext_base) | [qai_hub_models.models.convnext_base](qai_hub_models/models/convnext_base/README.md) |
| [ConvNext-Tiny](https://aihub.qualcomm.com/models/convnext_tiny) | [qai_hub_models.models.convnext_tiny](qai_hub_models/models/convnext_tiny/README.md) |
| [ConvNext-Tiny-w8a16-Quantized](https://aihub.qualcomm.com/models/convnext_tiny_w8a16_quantized) | [qai_hub_models.models.convnext_tiny_w8a16_quantized](qai_hub_models/models/convnext_tiny_w8a16_quantized/README.md) |
| [ConvNext-Tiny-w8a8-Quantized](https://aihub.qualcomm.com/models/convnext_tiny_w8a8_quantized) | [qai_hub_models.models.convnext_tiny_w8a8_quantized](qai_hub_models/models/convnext_tiny_w8a8_quantized/README.md) |
| [DenseNet-121](https://aihub.qualcomm.com/models/densenet121) | [qai_hub_models.models.densenet121](qai_hub_models/models/densenet121/README.md) |
| [DenseNet-121-Quantized](https://aihub.qualcomm.com/models/densenet121_quantized) | [qai_hub_models.models.densenet121_quantized](qai_hub_models/models/densenet121_quantized/README.md) |
| [EfficientNet-B0](https://aihub.qualcomm.com/models/efficientnet_b0) | [qai_hub_models.models.efficientnet_b0](qai_hub_models/models/efficientnet_b0/README.md) |
| [EfficientNet-B4](https://aihub.qualcomm.com/models/efficientnet_b4) | [qai_hub_models.models.efficientnet_b4](qai_hub_models/models/efficientnet_b4/README.md) |
| [EfficientNet-V2-s](https://aihub.qualcomm.com/models/efficientnet_v2_s) | [qai_hub_models.models.efficientnet_v2_s](qai_hub_models/models/efficientnet_v2_s/README.md) |
| [EfficientViT-b2-cls](https://aihub.qualcomm.com/models/efficientvit_b2_cls) | [qai_hub_models.models.efficientvit_b2_cls](qai_hub_models/models/efficientvit_b2_cls/README.md) |
| [EfficientViT-l2-cls](https://aihub.qualcomm.com/models/efficientvit_l2_cls) | [qai_hub_models.models.efficientvit_l2_cls](qai_hub_models/models/efficientvit_l2_cls/README.md) |
| [GoogLeNet](https://aihub.qualcomm.com/models/googlenet) | [qai_hub_models.models.googlenet](qai_hub_models/models/googlenet/README.md) |
| [GoogLeNetQuantized](https://aihub.qualcomm.com/models/googlenet_quantized) | [qai_hub_models.models.googlenet_quantized](qai_hub_models/models/googlenet_quantized/README.md) |
| [Inception-v3](https://aihub.qualcomm.com/models/inception_v3) | [qai_hub_models.models.inception_v3](qai_hub_models/models/inception_v3/README.md) |
| [Inception-v3-Quantized](https://aihub.qualcomm.com/models/inception_v3_quantized) | [qai_hub_models.models.inception_v3_quantized](qai_hub_models/models/inception_v3_quantized/README.md) |
| [MNASNet05](https://aihub.qualcomm.com/models/mnasnet05) | [qai_hub_models.models.mnasnet05](qai_hub_models/models/mnasnet05/README.md) |
| [MobileNet-v2](https://aihub.qualcomm.com/models/mobilenet_v2) | [qai_hub_models.models.mobilenet_v2](qai_hub_models/models/mobilenet_v2/README.md) |
| [MobileNet-v2-Quantized](https://aihub.qualcomm.com/models/mobilenet_v2_quantized) | [qai_hub_models.models.mobilenet_v2_quantized](qai_hub_models/models/mobilenet_v2_quantized/README.md) |
| [MobileNet-v3-Large](https://aihub.qualcomm.com/models/mobilenet_v3_large) | [qai_hub_models.models.mobilenet_v3_large](qai_hub_models/models/mobilenet_v3_large/README.md) |
| [MobileNet-v3-Large-Quantized](https://aihub.qualcomm.com/models/mobilenet_v3_large_quantized) | [qai_hub_models.models.mobilenet_v3_large_quantized](qai_hub_models/models/mobilenet_v3_large_quantized/README.md) |
| [MobileNet-v3-Small](https://aihub.qualcomm.com/models/mobilenet_v3_small) | [qai_hub_models.models.mobilenet_v3_small](qai_hub_models/models/mobilenet_v3_small/README.md) |
| [Mobile_Vit](https://aihub.qualcomm.com/models/mobile_vit) | [qai_hub_models.models.mobile_vit](qai_hub_models/models/mobile_vit/README.md) |
| [RegNet](https://aihub.qualcomm.com/models/regnet) | [qai_hub_models.models.regnet](qai_hub_models/models/regnet/README.md) |
| [RegNetQuantized](https://aihub.qualcomm.com/models/regnet_quantized) | [qai_hub_models.models.regnet_quantized](qai_hub_models/models/regnet_quantized/README.md) |
| [ResNeXt101](https://aihub.qualcomm.com/models/resnext101) | [qai_hub_models.models.resnext101](qai_hub_models/models/resnext101/README.md) |
| [ResNeXt101Quantized](https://aihub.qualcomm.com/models/resnext101_quantized) | [qai_hub_models.models.resnext101_quantized](qai_hub_models/models/resnext101_quantized/README.md) |
| [ResNeXt50](https://aihub.qualcomm.com/models/resnext50) | [qai_hub_models.models.resnext50](qai_hub_models/models/resnext50/README.md) |
| [ResNeXt50Quantized](https://aihub.qualcomm.com/models/resnext50_quantized) | [qai_hub_models.models.resnext50_quantized](qai_hub_models/models/resnext50_quantized/README.md) |
| [ResNet101](https://aihub.qualcomm.com/models/resnet101) | [qai_hub_models.models.resnet101](qai_hub_models/models/resnet101/README.md) |
| [ResNet101Quantized](https://aihub.qualcomm.com/models/resnet101_quantized) | [qai_hub_models.models.resnet101_quantized](qai_hub_models/models/resnet101_quantized/README.md) |
| [ResNet18](https://aihub.qualcomm.com/models/resnet18) | [qai_hub_models.models.resnet18](qai_hub_models/models/resnet18/README.md) |
| [ResNet18Quantized](https://aihub.qualcomm.com/models/resnet18_quantized) | [qai_hub_models.models.resnet18_quantized](qai_hub_models/models/resnet18_quantized/README.md) |
| [ResNet50](https://aihub.qualcomm.com/models/resnet50) | [qai_hub_models.models.resnet50](qai_hub_models/models/resnet50/README.md) |
| [ResNet50Quantized](https://aihub.qualcomm.com/models/resnet50_quantized) | [qai_hub_models.models.resnet50_quantized](qai_hub_models/models/resnet50_quantized/README.md) |
| [Shufflenet-v2](https://aihub.qualcomm.com/models/shufflenet_v2) | [qai_hub_models.models.shufflenet_v2](qai_hub_models/models/shufflenet_v2/README.md) |
| [Shufflenet-v2Quantized](https://aihub.qualcomm.com/models/shufflenet_v2_quantized) | [qai_hub_models.models.shufflenet_v2_quantized](qai_hub_models/models/shufflenet_v2_quantized/README.md) |
| [SqueezeNet-1_1](https://aihub.qualcomm.com/models/squeezenet1_1) | [qai_hub_models.models.squeezenet1_1](qai_hub_models/models/squeezenet1_1/README.md) |
| [SqueezeNet-1_1Quantized](https://aihub.qualcomm.com/models/squeezenet1_1_quantized) | [qai_hub_models.models.squeezenet1_1_quantized](qai_hub_models/models/squeezenet1_1_quantized/README.md) |
| [Swin-Base](https://aihub.qualcomm.com/models/swin_base) | [qai_hub_models.models.swin_base](qai_hub_models/models/swin_base/README.md) |
| [Swin-Small](https://aihub.qualcomm.com/models/swin_small) | [qai_hub_models.models.swin_small](qai_hub_models/models/swin_small/README.md) |
| [Swin-Tiny](https://aihub.qualcomm.com/models/swin_tiny) | [qai_hub_models.models.swin_tiny](qai_hub_models/models/swin_tiny/README.md) |
| [VIT](https://aihub.qualcomm.com/models/vit) | [qai_hub_models.models.vit](qai_hub_models/models/vit/README.md) |
| [VITQuantized](https://aihub.qualcomm.com/models/vit_quantized) | [qai_hub_models.models.vit_quantized](qai_hub_models/models/vit_quantized/README.md) |
| [WideResNet50](https://aihub.qualcomm.com/models/wideresnet50) | [qai_hub_models.models.wideresnet50](qai_hub_models/models/wideresnet50/README.md) |
| [WideResNet50-Quantized](https://aihub.qualcomm.com/models/wideresnet50_quantized) | [qai_hub_models.models.wideresnet50_quantized](qai_hub_models/models/wideresnet50_quantized/README.md) |
| | |
| **Image Editing**
| [AOT-GAN](https://aihub.qualcomm.com/models/aotgan) | [qai_hub_models.models.aotgan](qai_hub_models/models/aotgan/README.md) |
| [LaMa-Dilated](https://aihub.qualcomm.com/models/lama_dilated) | [qai_hub_models.models.lama_dilated](qai_hub_models/models/lama_dilated/README.md) |
| | |
| **Super Resolution**
| [ESRGAN](https://aihub.qualcomm.com/models/esrgan) | [qai_hub_models.models.esrgan](qai_hub_models/models/esrgan/README.md) |
| [QuickSRNetLarge](https://aihub.qualcomm.com/models/quicksrnetlarge) | [qai_hub_models.models.quicksrnetlarge](qai_hub_models/models/quicksrnetlarge/README.md) |
| [QuickSRNetLarge-Quantized](https://aihub.qualcomm.com/models/quicksrnetlarge_quantized) | [qai_hub_models.models.quicksrnetlarge_quantized](qai_hub_models/models/quicksrnetlarge_quantized/README.md) |
| [QuickSRNetMedium](https://aihub.qualcomm.com/models/quicksrnetmedium) | [qai_hub_models.models.quicksrnetmedium](qai_hub_models/models/quicksrnetmedium/README.md) |
| [QuickSRNetMedium-Quantized](https://aihub.qualcomm.com/models/quicksrnetmedium_quantized) | [qai_hub_models.models.quicksrnetmedium_quantized](qai_hub_models/models/quicksrnetmedium_quantized/README.md) |
| [QuickSRNetSmall](https://aihub.qualcomm.com/models/quicksrnetsmall) | [qai_hub_models.models.quicksrnetsmall](qai_hub_models/models/quicksrnetsmall/README.md) |
| [QuickSRNetSmall-Quantized](https://aihub.qualcomm.com/models/quicksrnetsmall_quantized) | [qai_hub_models.models.quicksrnetsmall_quantized](qai_hub_models/models/quicksrnetsmall_quantized/README.md) |
| [Real-ESRGAN-General-x4v3](https://aihub.qualcomm.com/models/real_esrgan_general_x4v3) | [qai_hub_models.models.real_esrgan_general_x4v3](qai_hub_models/models/real_esrgan_general_x4v3/README.md) |
| [Real-ESRGAN-x4plus](https://aihub.qualcomm.com/models/real_esrgan_x4plus) | [qai_hub_models.models.real_esrgan_x4plus](qai_hub_models/models/real_esrgan_x4plus/README.md) |
| [SESR-M5](https://aihub.qualcomm.com/models/sesr_m5) | [qai_hub_models.models.sesr_m5](qai_hub_models/models/sesr_m5/README.md) |
| [SESR-M5-Quantized](https://aihub.qualcomm.com/models/sesr_m5_quantized) | [qai_hub_models.models.sesr_m5_quantized](qai_hub_models/models/sesr_m5_quantized/README.md) |
| [XLSR](https://aihub.qualcomm.com/models/xlsr) | [qai_hub_models.models.xlsr](qai_hub_models/models/xlsr/README.md) |
| [XLSR-Quantized](https://aihub.qualcomm.com/models/xlsr_quantized) | [qai_hub_models.models.xlsr_quantized](qai_hub_models/models/xlsr_quantized/README.md) |
| | |
| **Semantic Segmentation**
| [DDRNet23-Slim](https://aihub.qualcomm.com/models/ddrnet23_slim) | [qai_hub_models.models.ddrnet23_slim](qai_hub_models/models/ddrnet23_slim/README.md) |
| [DeepLabV3-Plus-MobileNet](https://aihub.qualcomm.com/models/deeplabv3_plus_mobilenet) | [qai_hub_models.models.deeplabv3_plus_mobilenet](qai_hub_models/models/deeplabv3_plus_mobilenet/README.md) |
| [DeepLabV3-Plus-MobileNet-Quantized](https://aihub.qualcomm.com/models/deeplabv3_plus_mobilenet_quantized) | [qai_hub_models.models.deeplabv3_plus_mobilenet_quantized](qai_hub_models/models/deeplabv3_plus_mobilenet_quantized/README.md) |
| [DeepLabV3-ResNet50](https://aihub.qualcomm.com/models/deeplabv3_resnet50) | [qai_hub_models.models.deeplabv3_resnet50](qai_hub_models/models/deeplabv3_resnet50/README.md) |
| [EfficientViT-l2-seg](https://aihub.qualcomm.com/models/efficientvit_l2_seg) | [qai_hub_models.models.efficientvit_l2_seg](qai_hub_models/models/efficientvit_l2_seg/README.md) |
| [FCN-ResNet50](https://aihub.qualcomm.com/models/fcn_resnet50) | [qai_hub_models.models.fcn_resnet50](qai_hub_models/models/fcn_resnet50/README.md) |
| [FCN-ResNet50-Quantized](https://aihub.qualcomm.com/models/fcn_resnet50_quantized) | [qai_hub_models.models.fcn_resnet50_quantized](qai_hub_models/models/fcn_resnet50_quantized/README.md) |
| [FFNet-122NS-LowRes](https://aihub.qualcomm.com/models/ffnet_122ns_lowres) | [qai_hub_models.models.ffnet_122ns_lowres](qai_hub_models/models/ffnet_122ns_lowres/README.md) |
| [FFNet-40S](https://aihub.qualcomm.com/models/ffnet_40s) | [qai_hub_models.models.ffnet_40s](qai_hub_models/models/ffnet_40s/README.md) |
| [FFNet-40S-Quantized](https://aihub.qualcomm.com/models/ffnet_40s_quantized) | [qai_hub_models.models.ffnet_40s_quantized](qai_hub_models/models/ffnet_40s_quantized/README.md) |
| [FFNet-54S](https://aihub.qualcomm.com/models/ffnet_54s) | [qai_hub_models.models.ffnet_54s](qai_hub_models/models/ffnet_54s/README.md) |
| [FFNet-54S-Quantized](https://aihub.qualcomm.com/models/ffnet_54s_quantized) | [qai_hub_models.models.ffnet_54s_quantized](qai_hub_models/models/ffnet_54s_quantized/README.md) |
| [FFNet-78S](https://aihub.qualcomm.com/models/ffnet_78s) | [qai_hub_models.models.ffnet_78s](qai_hub_models/models/ffnet_78s/README.md) |
| [FFNet-78S-LowRes](https://aihub.qualcomm.com/models/ffnet_78s_lowres) | [qai_hub_models.models.ffnet_78s_lowres](qai_hub_models/models/ffnet_78s_lowres/README.md) |
| [FFNet-78S-Quantized](https://aihub.qualcomm.com/models/ffnet_78s_quantized) | [qai_hub_models.models.ffnet_78s_quantized](qai_hub_models/models/ffnet_78s_quantized/README.md) |
| [FastSam-S](https://aihub.qualcomm.com/models/fastsam_s) | [qai_hub_models.models.fastsam_s](qai_hub_models/models/fastsam_s/README.md) |
| [FastSam-X](https://aihub.qualcomm.com/models/fastsam_x) | [qai_hub_models.models.fastsam_x](qai_hub_models/models/fastsam_x/README.md) |
| [MediaPipe-Selfie-Segmentation](https://aihub.qualcomm.com/models/mediapipe_selfie) | [qai_hub_models.models.mediapipe_selfie](qai_hub_models/models/mediapipe_selfie/README.md) |
| [SINet](https://aihub.qualcomm.com/models/sinet) | [qai_hub_models.models.sinet](qai_hub_models/models/sinet/README.md) |
| [Segment-Anything-Model](https://aihub.qualcomm.com/models/sam) | [qai_hub_models.models.sam](qai_hub_models/models/sam/README.md) |
| [Unet-Segmentation](https://aihub.qualcomm.com/models/unet_segmentation) | [qai_hub_models.models.unet_segmentation](qai_hub_models/models/unet_segmentation/README.md) |
| [YOLOv11-Segmentation](https://aihub.qualcomm.com/models/yolov11_seg) | [qai_hub_models.models.yolov11_seg](qai_hub_models/models/yolov11_seg/README.md) |
| [YOLOv8-Segmentation](https://aihub.qualcomm.com/models/yolov8_seg) | [qai_hub_models.models.yolov8_seg](qai_hub_models/models/yolov8_seg/README.md) |
| | |
| **Object Detection**
| [Conditional-DETR-ResNet50](https://aihub.qualcomm.com/models/conditional_detr_resnet50) | [qai_hub_models.models.conditional_detr_resnet50](qai_hub_models/models/conditional_detr_resnet50/README.md) |
| [DETR-ResNet101](https://aihub.qualcomm.com/models/detr_resnet101) | [qai_hub_models.models.detr_resnet101](qai_hub_models/models/detr_resnet101/README.md) |
| [DETR-ResNet101-DC5](https://aihub.qualcomm.com/models/detr_resnet101_dc5) | [qai_hub_models.models.detr_resnet101_dc5](qai_hub_models/models/detr_resnet101_dc5/README.md) |
| [DETR-ResNet50](https://aihub.qualcomm.com/models/detr_resnet50) | [qai_hub_models.models.detr_resnet50](qai_hub_models/models/detr_resnet50/README.md) |
| [DETR-ResNet50-DC5](https://aihub.qualcomm.com/models/detr_resnet50_dc5) | [qai_hub_models.models.detr_resnet50_dc5](qai_hub_models/models/detr_resnet50_dc5/README.md) |
| [Facial-Attribute-Detection](https://aihub.qualcomm.com/models/face_attrib_net) | [qai_hub_models.models.face_attrib_net](qai_hub_models/models/face_attrib_net/README.md) |
| [Facial-Attribute-Detection-Quantized](https://aihub.qualcomm.com/models/face_attrib_net_quantized) | [qai_hub_models.models.face_attrib_net_quantized](qai_hub_models/models/face_attrib_net_quantized/README.md) |
| [Lightweight-Face-Detection](https://aihub.qualcomm.com/models/face_det_lite) | [qai_hub_models.models.face_det_lite](qai_hub_models/models/face_det_lite/README.md) |
| [Lightweight-Face-Detection-Quantized](https://aihub.qualcomm.com/models/face_det_lite_quantized) | [qai_hub_models.models.face_det_lite_quantized](qai_hub_models/models/face_det_lite_quantized/README.md) |
| [MediaPipe-Face-Detection](https://aihub.qualcomm.com/models/mediapipe_face) | [qai_hub_models.models.mediapipe_face](qai_hub_models/models/mediapipe_face/README.md) |
| [MediaPipe-Face-Detection-Quantized](https://aihub.qualcomm.com/models/mediapipe_face_quantized) | [qai_hub_models.models.mediapipe_face_quantized](qai_hub_models/models/mediapipe_face_quantized/README.md) |
| [MediaPipe-Hand-Detection](https://aihub.qualcomm.com/models/mediapipe_hand) | [qai_hub_models.models.mediapipe_hand](qai_hub_models/models/mediapipe_hand/README.md) |
| [PPE-Detection](https://aihub.qualcomm.com/models/gear_guard_net) | [qai_hub_models.models.gear_guard_net](qai_hub_models/models/gear_guard_net/README.md) |
| [PPE-Detection-Quantized](https://aihub.qualcomm.com/models/gear_guard_net_quantized) | [qai_hub_models.models.gear_guard_net_quantized](qai_hub_models/models/gear_guard_net_quantized/README.md) |
| [Person-Foot-Detection](https://aihub.qualcomm.com/models/foot_track_net) | [qai_hub_models.models.foot_track_net](qai_hub_models/models/foot_track_net/README.md) |
| [Person-Foot-Detection-Quantized](https://aihub.qualcomm.com/models/foot_track_net_quantized) | [qai_hub_models.models.foot_track_net_quantized](qai_hub_models/models/foot_track_net_quantized/README.md) |
| [YOLOv11-Detection](https://aihub.qualcomm.com/models/yolov11_det) | [qai_hub_models.models.yolov11_det](qai_hub_models/models/yolov11_det/README.md) |
| [YOLOv8-Detection](https://aihub.qualcomm.com/models/yolov8_det) | [qai_hub_models.models.yolov8_det](qai_hub_models/models/yolov8_det/README.md) |
| [YOLOv8-Detection-Quantized](https://aihub.qualcomm.com/models/yolov8_det_quantized) | [qai_hub_models.models.yolov8_det_quantized](qai_hub_models/models/yolov8_det_quantized/README.md) |
| [Yolo-NAS](https://aihub.qualcomm.com/models/yolonas) | [qai_hub_models.models.yolonas](qai_hub_models/models/yolonas/README.md) |
| [Yolo-NAS-Quantized](https://aihub.qualcomm.com/models/yolonas_quantized) | [qai_hub_models.models.yolonas_quantized](qai_hub_models/models/yolonas_quantized/README.md) |
| [Yolo-v3](https://aihub.qualcomm.com/models/yolov3) | [qai_hub_models.models.yolov3](qai_hub_models/models/yolov3/README.md) |
| [Yolo-v6](https://aihub.qualcomm.com/models/yolov6) | [qai_hub_models.models.yolov6](qai_hub_models/models/yolov6/README.md) |
| [Yolo-v7](https://aihub.qualcomm.com/models/yolov7) | [qai_hub_models.models.yolov7](qai_hub_models/models/yolov7/README.md) |
| [Yolo-v7-Quantized](https://aihub.qualcomm.com/models/yolov7_quantized) | [qai_hub_models.models.yolov7_quantized](qai_hub_models/models/yolov7_quantized/README.md) |
| | |
| **Pose Estimation**
| [Facial-Landmark-Detection](https://aihub.qualcomm.com/models/facemap_3dmm) | [qai_hub_models.models.facemap_3dmm](qai_hub_models/models/facemap_3dmm/README.md) |
| [Facial-Landmark-Detection-Quantized](https://aihub.qualcomm.com/models/facemap_3dmm_quantized) | [qai_hub_models.models.facemap_3dmm_quantized](qai_hub_models/models/facemap_3dmm_quantized/README.md) |
| [HRNetPose](https://aihub.qualcomm.com/models/hrnet_pose) | [qai_hub_models.models.hrnet_pose](qai_hub_models/models/hrnet_pose/README.md) |
| [HRNetPoseQuantized](https://aihub.qualcomm.com/models/hrnet_pose_quantized) | [qai_hub_models.models.hrnet_pose_quantized](qai_hub_models/models/hrnet_pose_quantized/README.md) |
| [LiteHRNet](https://aihub.qualcomm.com/models/litehrnet) | [qai_hub_models.models.litehrnet](qai_hub_models/models/litehrnet/README.md) |
| [MediaPipe-Pose-Estimation](https://aihub.qualcomm.com/models/mediapipe_pose) | [qai_hub_models.models.mediapipe_pose](qai_hub_models/models/mediapipe_pose/README.md) |
| [OpenPose](https://aihub.qualcomm.com/models/openpose) | [qai_hub_models.models.openpose](qai_hub_models/models/openpose/README.md) |
| [Posenet-Mobilenet](https://aihub.qualcomm.com/models/posenet_mobilenet) | [qai_hub_models.models.posenet_mobilenet](qai_hub_models/models/posenet_mobilenet/README.md) |
| [Posenet-Mobilenet-Quantized](https://aihub.qualcomm.com/models/posenet_mobilenet_quantized) | [qai_hub_models.models.posenet_mobilenet_quantized](qai_hub_models/models/posenet_mobilenet_quantized/README.md) |
| | |
| **Depth Estimation**
| [Depth-Anything](https://aihub.qualcomm.com/models/depth_anything) | [qai_hub_models.models.depth_anything](qai_hub_models/models/depth_anything/README.md) |
| [Depth-Anything-V2](https://aihub.qualcomm.com/models/depth_anything_v2) | [qai_hub_models.models.depth_anything_v2](qai_hub_models/models/depth_anything_v2/README.md) |
| [Midas-V2](https://aihub.qualcomm.com/models/midas) | [qai_hub_models.models.midas](qai_hub_models/models/midas/README.md) |
| [Midas-V2-Quantized](https://aihub.qualcomm.com/models/midas_quantized) | [qai_hub_models.models.midas_quantized](qai_hub_models/models/midas_quantized/README.md) |

### Audio

| Model | README |
| -- | -- |
| | |
| **Speech Recognition**
| [HuggingFace-WavLM-Base-Plus](https://aihub.qualcomm.com/models/huggingface_wavlm_base_plus) | [qai_hub_models.models.huggingface_wavlm_base_plus](qai_hub_models/models/huggingface_wavlm_base_plus/README.md) |
| [Whisper-Base-En](https://aihub.qualcomm.com/models/whisper_base_en) | [qai_hub_models.models.whisper_base_en](qai_hub_models/models/whisper_base_en/README.md) |
| [Whisper-Tiny-En](https://aihub.qualcomm.com/models/whisper_tiny_en) | [qai_hub_models.models.whisper_tiny_en](qai_hub_models/models/whisper_tiny_en/README.md) |

### Multimodal

| Model | README |
| -- | -- |
| | |
| [OpenAI-Clip](https://aihub.qualcomm.com/models/openai_clip) | [qai_hub_models.models.openai_clip](qai_hub_models/models/openai_clip/README.md) |
| [TrOCR](https://aihub.qualcomm.com/models/trocr) | [qai_hub_models.models.trocr](qai_hub_models/models/trocr/README.md) |

### Generative Ai

| Model | README |
| -- | -- |
| | |
| **Image Generation**
| [ControlNet](https://aihub.qualcomm.com/models/controlnet_quantized) | [qai_hub_models.models.controlnet_quantized](qai_hub_models/models/controlnet_quantized/README.md) |
| [Riffusion](https://aihub.qualcomm.com/models/riffusion_quantized) | [qai_hub_models.models.riffusion_quantized](qai_hub_models/models/riffusion_quantized/README.md) |
| [Stable-Diffusion-v1.5](https://aihub.qualcomm.com/models/stable_diffusion_v1_5_quantized) | [qai_hub_models.models.stable_diffusion_v1_5_quantized](qai_hub_models/models/stable_diffusion_v1_5_quantized/README.md) |
| [Stable-Diffusion-v2.1](https://aihub.qualcomm.com/models/stable_diffusion_v2_1_quantized) | [qai_hub_models.models.stable_diffusion_v2_1_quantized](qai_hub_models/models/stable_diffusion_v2_1_quantized/README.md) |
| | |
| **Text Generation**
| [Baichuan2-7B](https://aihub.qualcomm.com/models/baichuan2_7b_quantized) | [qai_hub_models.models.baichuan2_7b_quantized](qai_hub_models/models/baichuan2_7b_quantized/README.md) |
| [IBM-Granite-3B-Code-Instruct](https://aihub.qualcomm.com/models/ibm_granite_3b_code_instruct) | [qai_hub_models.models.ibm_granite_3b_code_instruct](qai_hub_models/models/ibm_granite_3b_code_instruct/README.md) |
| [IndusQ-1.1B](https://aihub.qualcomm.com/models/indus_1b_quantized) | [qai_hub_models.models.indus_1b_quantized](qai_hub_models/models/indus_1b_quantized/README.md) |
| [JAIS-6p7b-Chat](https://aihub.qualcomm.com/models/jais_6p7b_chat_quantized) | [qai_hub_models.models.jais_6p7b_chat_quantized](qai_hub_models/models/jais_6p7b_chat_quantized/README.md) |
| [Llama-v2-7B-Chat](https://aihub.qualcomm.com/models/llama_v2_7b_chat_quantized) | [qai_hub_models.models.llama_v2_7b_chat_quantized](qai_hub_models/models/llama_v2_7b_chat_quantized/README.md) |
| [Llama-v3-8B-Chat](https://aihub.qualcomm.com/models/llama_v3_8b_chat_quantized) | [qai_hub_models.models.llama_v3_8b_chat_quantized](qai_hub_models/models/llama_v3_8b_chat_quantized/README.md) |
| [Llama-v3.1-8B-Chat](https://aihub.qualcomm.com/models/llama_v3_1_8b_chat_quantized) | [qai_hub_models.models.llama_v3_1_8b_chat_quantized](qai_hub_models/models/llama_v3_1_8b_chat_quantized/README.md) |
| [Llama-v3.2-3B-Chat](https://aihub.qualcomm.com/models/llama_v3_2_3b_chat_quantized) | [qai_hub_models.models.llama_v3_2_3b_chat_quantized](qai_hub_models/models/llama_v3_2_3b_chat_quantized/README.md) |
| [Mistral-3B](https://aihub.qualcomm.com/models/mistral_3b_quantized) | [qai_hub_models.models.mistral_3b_quantized](qai_hub_models/models/mistral_3b_quantized/README.md) |
| [Mistral-7B-Instruct-v0.3](https://aihub.qualcomm.com/models/mistral_7b_instruct_v0_3_quantized) | [qai_hub_models.models.mistral_7b_instruct_v0_3_quantized](qai_hub_models/models/mistral_7b_instruct_v0_3_quantized/README.md) |
| [PLaMo-1B](https://aihub.qualcomm.com/models/plamo_1b_quantized) | [qai_hub_models.models.plamo_1b_quantized](qai_hub_models/models/plamo_1b_quantized/README.md) |
| [Qwen2-7B-Instruct](https://aihub.qualcomm.com/models/qwen2_7b_instruct_quantized) | [qai_hub_models.models.qwen2_7b_instruct_quantized](qai_hub_models/models/qwen2_7b_instruct_quantized/README.md) |


## Need help?
Slack: https://aihub.qualcomm.com/community/slack

GitHub Issues: https://github.com/quic/ai-hub-models/issues

Email: ai-hub-support@qti.qualcomm.com.

## LICENSE

Qualcomm® AI Hub Models is licensed under BSD-3. See the [LICENSE file](LICENSE).
