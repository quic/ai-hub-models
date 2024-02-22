[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [StyleGAN2: Generate realistic, randomized images of real classes](https://aihub.qualcomm.com/models/stylegan2)

StyleGAN2 is a machine learning model that generates realistic images from random input state vectors.

This is based on the implementation of StyleGAN2 found
[here](https://github.com/NVlabs/stylegan3). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/stylegan2).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[stylegan2]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.stylegan2.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.stylegan2.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- Code in the Qualcomm® AI Hub Models repository is covered by the LICENSE
  file at the repository root.
- The license for the original implementation of StyleGAN2 can be found
  [here](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).


## References
* [Analyzing and Improving the Image Quality of StyleGAN](http://arxiv.org/abs/1912.04958)
* [Source Model Implementation](https://github.com/NVlabs/stylegan3)
