[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [DETR-ResNet50-DC5: Transformer based object detector with ResNet50 backbone (dilated C5 stage)](https://aihub.qualcomm.com/models/detr_resnet50_dc5)

DETR is a machine learning model that can detect objects (trained on COCO dataset).

This is based on the implementation of DETR-ResNet50-DC5 found
[here](https://github.com/facebookresearch/detr). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/detr_resnet50_dc5).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[detr_resnet50_dc5]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.detr_resnet50_dc5.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.detr_resnet50_dc5.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of DETR-ResNet50-DC5 can be found
  [here](https://github.com/facebookresearch/detr/blob/main/LICENSE).


## References
* [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
* [Source Model Implementation](https://github.com/facebookresearch/detr)
