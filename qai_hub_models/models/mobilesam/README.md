# [MobileSam: Faster Segment Anything: Towards lightweight SAM for mobile applications](https://aihub.qualcomm.com/models/mobilesam)

Transformer based encoder-decoder where prompts specify what to segment in an image thereby allowing segmentation without the need for additional training. The image encoder generates embeddings and the lightweight decoder operates on the embeddings for point and mask based image segmentation.

This is based on the implementation of MobileSam found [here](https://github.com/facebookresearch/segment-anything). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/mobilesam).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[mobilesam]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.mobilesam.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.mobilesam.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of MobileSam can be found
  [here](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Segment Anything](https://arxiv.org/abs/2306.14289)
* [Source Model Implementation](https://github.com/facebookresearch/segment-anything)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
