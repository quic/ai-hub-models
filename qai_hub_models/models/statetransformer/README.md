# [StateTransformer: Multi-agent trajectory prediction model for autonomous driving](https://aihub.qualcomm.com/models/statetransformer)

StateTransformer is a transformer-based model designed for trajectory prediction in self-driving scenarios. It integrates rasterized map data, agent context, and temporal dynamics to generate accurate future trajectories.

This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/statetransformer).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[statetransformer]" git+https://github.com/motional/nuplan-devkit.git@d60b4cd2071de9bb041509c43f5226dd22f248c0#egg=nuplan_devkit
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.statetransformer.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.statetransformer.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of StateTransformer can be found
  [here](This model's original implementation does not provide a LICENSE.).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
