# [DeformableDETR: Transformer based object detector with ResNet-50 backbone](https://aihub.qualcomm.com/models/deformable_detr)

Deformable DETR is a machine learning model that can detect objects (trained on COCO dataset).

This is based on the implementation of DeformableDETR found [here](https://github.com/fundamentalvision/Deformable-DETR). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/deformable_detr).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[deformable-detr]"
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.deformable_detr.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.deformable_detr.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of DeformableDETR can be found
  [here](https://github.com/facebookresearch/detr/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
* [Source Model Implementation](https://github.com/fundamentalvision/Deformable-DETR)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
