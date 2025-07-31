# [IBM-Granite-v3.1-8B-Instruct: State-of-the-art large language model useful on a variety of code understanding and generation tasks](https://aihub.qualcomm.com/models/ibm_granite_v3_1_8b_instruct)

Granite-3.1-8B-Instruct is a 8B parameter long-context instruct model finetuned from Granite-3.1-8B-Base using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets tailored for solving long context problems. This model is developed using a diverse set of techniques with a structured chat format, including supervised finetuning, model alignment using reinforcement learning, and model merging.

This is based on the implementation of IBM-Granite-v3.1-8B-Instruct found [here](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/ibm_granite_v3_1_8b_instruct).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying IBM Granite 3.1 on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.





## License
* The license for the original implementation of IBM-Granite-v3.1-8B-Instruct can be found
  [here](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).
* The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf)


## References
* [Granite Code Models: A Family of Open Foundation Models for Code Intelligence](https://arxiv.org/abs/2405.04324)
* [Source Model Implementation](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)



## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation
