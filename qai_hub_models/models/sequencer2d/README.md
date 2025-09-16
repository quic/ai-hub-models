# [Sequencer2D: Imagenet classifier and general purpose backbone](https://aihub.qualcomm.com/models/sequencer2d)

sequencer2d is a vision transformer model that can classify images from the Imagenet dataset.

This is based on the implementation of Sequencer2D found [here](https://github.com/okojoalg/sequencer). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/sequencer2d).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[sequencer2d]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.sequencer2d.demo { --quantize w8a8 }
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.sequencer2d.export { --quantize w8a8 }
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Sequencer2D can be found
  [here](https://github.com/facebookresearch/LeViT?tab=Apache-2.0-1-ov-file).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Sequencer: Deep LSTM for Image Classification](https://arxiv.org/abs/2205.01972)
* [Source Model Implementation](https://github.com/okojoalg/sequencer)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
