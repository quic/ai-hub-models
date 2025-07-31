# [Nomic-Embed-Text: Resizable Production Embeddings](https://aihub.qualcomm.com/models/nomic_embed_text)

A text encoder that surpasses OpenAI text-embedding-ada-002 and text-embedding-3-small performance on short and long context tasks.

This is based on the implementation of Nomic-Embed-Text found [here](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/nomic_embed_text).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.




## Example & Usage

Install the package via pip:
```bash
pip install "qai-hub-models[nomic-embed-text]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.nomic_embed_text.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.nomic_embed_text.export
```
Additional options are documented with the `--help` option.


## License
* The license for the original implementation of Nomic-Embed-Text can be found
  [here](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Introducing Nomic Embed: A Truly Open Embedding Model](https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
* [Source Model Implementation](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
