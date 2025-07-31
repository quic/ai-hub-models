# [BGNet: Segment images in real-time on device](https://aihub.qualcomm.com/models/bgnet)

BGNet or Boundary-Guided Network, is a model designed for camouflaged object detection. It leverages edge semantics to enhance the representation learning process, making it more effective at identifying objects that blend into their surroundings

This is based on the implementation of BGNet found [here](https://github.com/thograce/bgnet). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/bgnet).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[bgnet]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.bgnet.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.bgnet.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of BGNet can be found
  [here](This model's original implementation does not provide a LICENSE.).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [BGNet: Boundary-Guided Camouflaged Object Detection (IJCAI 2022)](https://arxiv.org/abs/2207.00794)
* [Source Model Implementation](https://github.com/thograce/bgnet)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
