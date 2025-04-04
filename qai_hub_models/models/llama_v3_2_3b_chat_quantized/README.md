[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Llama-v3.2-3B-Chat: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/llama_v3_2_3b_chat_quantized)

Llama 3 is a family of LLMs. The "Chat" at the end indicates that the model is optimized for chatbot-like dialogue. The model is quantized to w4a16 (4-bit weights and 16-bit activations) and part of the model is quantized to w8a16 (8-bit weights and 16-bit activations) making it suitable for on-device deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-Quantized's latency.

This is based on the implementation of Llama-v3.2-3B-Chat found [here](https://github.com/meta-llama/llama3/tree/main). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v3_2_3b_chat_quantized).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying Llama 3.2 on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.





## License
* The license for the original implementation of Llama-v3.2-3B-Chat can be found
  [here](https://github.com/facebookresearch/llama/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/facebookresearch/llama/blob/main/LICENSE)


## References
* [LLaMA: Open and Efficient Foundation Language Models](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
* [Source Model Implementation](https://github.com/meta-llama/llama3/tree/main)



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
