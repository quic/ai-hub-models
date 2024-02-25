[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [Llama-v2-7B-Chat: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/llama_v2_7b_chat_quantized)

Llama 2 is a family of LLMs. The "Chat" at the end indicates that the model is optimized for chatbot-like dialogue. The model is quantized to 4-bit weights and 16-bit activations making it suitable for on-device deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-KVCache-Quantized's latency.

This is based on the implementation of Llama-v2-7B-Chat found
[here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v2_7b_chat_quantized).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.



## License
- The license for the original implementation of Llama-v2-7B-Chat can be found
  [here](https://github.com/facebookresearch/llama/blob/main/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf).

## References
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Source Model Implementation](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

## Community
* Join [our AI Hub Slack community](https://join.slack.com/t/qualcomm-ai-hub/shared_invite/zt-2dgf95loi-CXHTDRR1rvPgQWPO~ZZZJg) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).


