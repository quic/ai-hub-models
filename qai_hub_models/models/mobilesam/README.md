# [MobileSam: Faster Segment Anything: Towards lightweight SAM for mobile applications](https://aihub.qualcomm.com/models/mobilesam)

Transformer based encoder-decoder where prompts specify what to segment in an image thereby allowing segmentation without the need for additional training. The image encoder generates embeddings and the lightweight decoder operates on the embeddings for point and mask based image segmentation.

This is based on the implementation of MobileSam found [here](https://github.com/facebookresearch/segment-anything).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/mobilesam).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[mobilesam]" git+https://github.com/ChaoningZhang/MobileSAM@34bbbfd --use-pep517
```

### 2. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.mobilesam.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

By default, the demo will run locally in PyTorch. Pass `--eval-mode on-device` to the demo script to run the model on a cloud-hosted target device.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.mobilesam.export
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of MobileSam can be found
  [here](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).

## References
* [Segment Anything](https://arxiv.org/abs/2306.14289)
* [Source Model Implementation](https://github.com/facebookresearch/segment-anything)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
