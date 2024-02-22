[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [QuickSRNetSmall: Upscale images and remove image noise](https://aihub.qualcomm.com/models/quicksrnetsmall)

QuickSRNet Small is designed for upscaling images on mobile platforms to sharpen in real-time.

This is based on the implementation of QuickSRNetSmall found
[here](https://github.com/quic/aimet-model-zoo/tree/develop/aimet_zoo_torch/quicksrnet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/quicksrnetsmall).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.quicksrnetsmall.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.quicksrnetsmall.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of QuickSRNetSmall can be found
  [here](https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf).


## References
* [QuickSRNet: Plain Single-Image Super-Resolution Architecture for Faster Inference on Mobile Platforms](https://arxiv.org/abs/2303.04336)
* [Source Model Implementation](https://github.com/quic/aimet-model-zoo/tree/develop/aimet_zoo_torch/quicksrnet)
