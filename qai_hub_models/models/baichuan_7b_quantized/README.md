[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Baichuan-7B: Large language model achieving state-of-the-art performance on Chinese and English language benchmarks](https://aihub.qualcomm.com/models/baichuan_7b_quantized)

Baichuan-7B is a family of LLMs. It achieves the state-of-the-art performance of its size on standard Chinese and English authoritative benchmarks (C-EVAL/MMLU). 4-bit weights and 16-bit activations making it suitable for on-device The model is quantized to deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-KVCache-Quantized's latency.

This is based on the implementation of Baichuan-7B found
[here](https://github.com/baichuan-inc/Baichuan-7B/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/baichuan_7b_quantized).

[Sign up](https://myaccount.qualcomm.com/signup) for early access to run these models on
a hosted Qualcomm® device.





## License
- The license for the original implementation of Baichuan-7B can be found
  [here](https://github.com/baichuan-inc/Baichuan-7B/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://github.com/baichuan-inc/Baichuan-7B/blob/main/LICENSE)

## References
* [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)
* [Source Model Implementation](https://github.com/baichuan-inc/Baichuan-7B/)

## Community
* Join [our AI Hub Slack community](https://qualcomm-ai-hub.slack.com/join/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA#/shared-invite/email) to collaborate, post questions and learn more about on-device AI.
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


