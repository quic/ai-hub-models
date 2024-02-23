[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Baichuan-7B: Large language model achieving state-of-the-art performance on Chinese and English language benchmarks](https://aihub.qualcomm.com/models/baichuan_7b_quantized)

Baichuan-7B is a family of LLMs. It achieves the state-of-the-art performance of its size on standard Chinese and English authoritative benchmarks (C-EVAL/MMLU). 4-bit weights and 16-bit activations making it suitable for on-device The model is quantized to deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-KVCache-Quantized's latency.

This is based on the implementation of Baichuan-7B found
[here](https://github.com/baichuan-inc/Baichuan-7B/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/baichuan_7b_quantized).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.



## License
- The license for the original implementation of Baichuan-7B can be found
  [here](https://github.com/baichuan-inc/Baichuan-7B/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf).

## References
* [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)
* [Source Model Implementation](https://github.com/baichuan-inc/Baichuan-7B/)

## Community
* Join [our AI Hub Slack community](https://join.slack.com/t/qualcommaihub-nac3926/shared_invite/zt-2d5zsmas3-Sj0Q9TzslueCjS31eXG2UA) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

{usage_and_limitation}
