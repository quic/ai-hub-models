# [HuggingFace-WavLM-Base-Plus: Real-time Speech processing](https://aihub.qualcomm.com/models/huggingface_wavlm_base_plus)

HuggingFaceWavLMBasePlus is a real time speech processing backbone based on Microsoft's WavLM model.

This is based on the implementation of HuggingFace-WavLM-Base-Plus found [here](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/huggingface_wavlm_base_plus).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[huggingface-wavlm-base-plus]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.huggingface_wavlm_base_plus.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.huggingface_wavlm_base_plus.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of HuggingFace-WavLM-Base-Plus can be found
  [here](https://github.com/microsoft/unilm/blob/master/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
* [Source Model Implementation](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus/tree/main)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
