[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [FastSam-S: Generate high quality segmentation mask on device](https://aihub.qualcomm.com/models/fastsam_s)

The Fast Segment Anything Model (FastSAM) is a novel, real-time CNN-based solution for the Segment Anything task. This task is designed to segment any object within an image based on various possible user interaction prompts. The model performs competitively despite significantly reduced computation, making it a practical choice for a variety of vision tasks.

This is based on the implementation of FastSam-S found
[here](https://github.com/CASIA-IVA-Lab/FastSAM). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/fastsam_s).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[fastsam_s]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.fastsam_s.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.fastsam_s.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of FastSam-S can be found
  [here](https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/LICENSE).


## References
* [Fast Segment Anything](https://arxiv.org/abs/2306.12156)
* [Source Model Implementation](https://github.com/CASIA-IVA-Lab/FastSAM)
