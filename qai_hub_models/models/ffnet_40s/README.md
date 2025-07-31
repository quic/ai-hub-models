# [FFNet-40S: Semantic segmentation for automotive street scenes](https://aihub.qualcomm.com/models/ffnet_40s)

FFNet-40S is a "fuss-free network" that segments street scene images with per-pixel classes like road, sidewalk, and pedestrian. Trained on the Cityscapes dataset.

This is based on the implementation of FFNet-40S found [here](https://github.com/Qualcomm-AI-research/FFNet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/ffnet_40s).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[ffnet-40s]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.ffnet_40s.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.ffnet_40s.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of FFNet-40S can be found
  [here](https://github.com/Qualcomm-AI-research/FFNet/blob/master/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Simple and Efficient Architectures for Semantic Segmentation](https://arxiv.org/abs/2206.08236)
* [Source Model Implementation](https://github.com/Qualcomm-AI-research/FFNet)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
