[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [TrOCR: Transformer based model for state-of-the-art optical character recognition (OCR) on both printed and handwritten text](https://aihub.qualcomm.com/models/trocr)

End-to-end text recognition approach with pre-trained image transformer and text transformer models for both image understanding and wordpiece-level text generation.

This is based on the implementation of TrOCR found
[here](https://huggingface.co/microsoft/trocr-small-stage1). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/trocr).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[trocr]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.trocr.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.trocr.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of TrOCR can be found
  [here](https://github.com/microsoft/unilm/blob/master/LICENSE).


## References
* [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)
* [Source Model Implementation](https://huggingface.co/microsoft/trocr-small-stage1)
