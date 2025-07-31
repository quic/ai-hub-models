# [Person-Foot-Detection: Multi-task Human detector](https://aihub.qualcomm.com/models/foot_track_net)

Real-time multiple person detection with accurate feet localization optimized for mobile and edge.

This is based on the implementation of Person-Foot-Detection found [here](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/foot_track_net/model.py). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/foot_track_net).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[foot-track-net]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.foot_track_net.demo { --quantize w8a8, w8a16 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.foot_track_net.export { --quantize w8a8, w8a16 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Person-Foot-Detection can be found
  [here](https://github.com/quic/ai-hub-models/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Source Model Implementation](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/foot_track_net/model.py)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
