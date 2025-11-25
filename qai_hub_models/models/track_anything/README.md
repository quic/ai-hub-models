# [Track-Anything: High-performance interactive tracking and segmentation in videos](https://aihub.qualcomm.com/models/track_anything)

Track-Anything is a video based machine learning model to track an object in a video.

This is based on the implementation of Track-Anything found [here](https://github.com/gaomingqi/Track-Anything). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/track_anything).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[track-anything]" git+https://github.com/ChaoningZhang/MobileSAM@34bbbfd --use-pep517
```


Once installed, run the following simple CLI demo on the host machine:

```bash
python -m qai_hub_models.models.track_anything.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This package contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.track_anything.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Track-Anything can be found
  [here](https://github.com/gaomingqi/Track-Anything/blob/master/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Track Anything: Segment Anything Meets Videos](https://arxiv.org/abs/2304.11968)
* [Source Model Implementation](https://github.com/gaomingqi/Track-Anything)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
