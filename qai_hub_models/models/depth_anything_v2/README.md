# [Depth-Anything-V2: Deep Convolutional Neural Network model for depth estimation](https://aihub.qualcomm.com/models/depth_anything_v2)

Depth Anything is designed for estimating depth at each point in an image.

This is based on the implementation of Depth-Anything-V2 found [here](https://github.com/huggingface/transformers/tree/main/src/transformers/models/depth_anything). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/depth_anything_v2).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[depth-anything-v2]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.depth_anything_v2.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.depth_anything_v2.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Depth-Anything-V2 can be found
  [here](https://github.com/huggingface/transformers/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Depth Anything V2](https://arxiv.org/abs/2406.09414)
* [Source Model Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/depth_anything)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
