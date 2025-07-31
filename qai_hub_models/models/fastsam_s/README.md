# [FastSam-S: Generate high quality segmentation mask on device](https://aihub.qualcomm.com/models/fastsam_s)

The Fast Segment Anything Model (FastSAM) is a novel, real-time CNN-based solution for the Segment Anything task. This task is designed to segment any object within an image based on various possible user interaction prompts. The model performs competitively despite significantly reduced computation, making it a practical choice for a variety of vision tasks.

This is based on the implementation of FastSam-S found [here](https://github.com/CASIA-IVA-Lab/FastSAM). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/fastsam_s).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[fastsam-s]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.fastsam_s.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.fastsam_s.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of FastSam-S can be found
  [here](https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/LICENSE)


## References
* [Fast Segment Anything](https://arxiv.org/abs/2306.12156)
* [Source Model Implementation](https://github.com/CASIA-IVA-Lab/FastSAM)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
