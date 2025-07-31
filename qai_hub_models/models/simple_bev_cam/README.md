# [Simple-Bev: Construct a bird's eye view from sensors mounted on a vehicle](https://aihub.qualcomm.com/models/simple_bev_cam)

Simple-Bev is a machine learning model for generating a bird's eye view representation from the sensors (cameras) mounted on a vehicle. It uses ResNet-101 as the backbone and segnet as a segmentation model for specific use cases.

This is based on the implementation of Simple-Bev found [here](https://github.com/aharley/simple_bev/blob/main/nets/segnet.py). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/simple_bev_cam).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install qai-hub-models
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.simple_bev_cam.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.simple_bev_cam.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Simple-Bev can be found
  [here](https://github.com/aharley/simple_bev/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?](https://arxiv.org/abs/2206.07959)
* [Source Model Implementation](https://github.com/aharley/simple_bev/blob/main/nets/segnet.py)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
