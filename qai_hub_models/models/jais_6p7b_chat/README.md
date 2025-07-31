# [JAIS-6p7b-Chat: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/jais_6p7b_chat)

JAIS 6.7B is a bilingual large language model (LLM) for both Arabic and English developed by Inception, a G42 company in partnership with MBZUAI and Cerebras. This is a 6.7 billion parameter LLM, trained on a dataset containing 141 billion Arabic tokens and 339 billion English/code tokens. The model is based on transformer-based decoder-only (GPT-3) architecture and uses SwiGLU non-linearity. It implements ALiBi position embeddings, enabling the model to extrapolate to long sequence lengths, providing improved context handling and model precision. The JAIS family of models is a comprehensive series of bilingual English-Arabic LLMs. These models are optimized to excel in Arabic while having strong English capabilities.

This is based on the implementation of JAIS-6p7b-Chat found [here](https://huggingface.co/inceptionai/jais-family-6p7b). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/jais_6p7b_chat).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying JAIS-6p7b-Chat on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.





## References
* [Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models](https://arxiv.org/abs/2308.16149)
* [Source Model Implementation](https://huggingface.co/inceptionai/jais-family-6p7b)



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
