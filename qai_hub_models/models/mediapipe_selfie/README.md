# [MediaPipe-Selfie-Segmentation: Segments the person from background in a selfie image and realtime background segmentation in video conferencing](https://aihub.qualcomm.com/models/mediapipe_selfie)

Light-weight model that segments a person from the background in square or landscape selfie and video conference imagery.

This is based on the implementation of MediaPipe-Selfie-Segmentation found [here](https://github.com/google/mediapipe/tree/master/mediapipe/modules/selfie_segmentation). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/mediapipe_selfie).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[mediapipe-selfie]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.mediapipe_selfie.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.mediapipe_selfie.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of MediaPipe-Selfie-Segmentation can be found
  [here](https://github.com/google/mediapipe/blob/master/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Image segmentation guide](https://developers.google.com/mediapipe/solutions/vision/image_segmenter/)
* [Source Model Implementation](https://github.com/google/mediapipe/tree/master/mediapipe/modules/selfie_segmentation)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
