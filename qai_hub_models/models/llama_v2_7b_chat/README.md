# [Llama-v2-7B-Chat: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/llama_v2_7b_chat)

Llama 2 is a family of LLMs. The "Chat" at the end indicates that the model is optimized for chatbot-like dialogue. The model is quantized to w4a16(4-bit weights and 16-bit activations) and part of the model is quantized to w8a16(8-bit weights and 16-bit activations) making it suitable for on-device deployment. For Prompt and output length specified below, the time to first token is Llama-PromptProcessor-Quantized's latency and average time per addition token is Llama-TokenGenerator-KVCache-Quantized's latency.

This is based on the implementation of Llama-v2-7B-Chat found [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/llama_v2_7b_chat).

[Sign up](https://myaccount.qualcomm.com/signup) to start using Qualcomm AI Hub and run these models on a hosted Qualcomm® device.

## Deploying Llama 2 on-device

Please follow the [LLM on-device deployment](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.

## Sample output prompts generated on-device
1. --prompt "what is gravity?" --max-output-tokens 30
~~~
-------- Response Summary --------
Prompt: what is gravity?
Response: Hello! I'm here to help you answer your question. Gravity is a fundamental force of nature that affects the behavior of objects with mass
~~~

2. --prompt "what is 2+3?" --max-output-tokens 30
~~~
-------- Response Summary --------
Prompt: what is 2+3?
Response: Of course! I'm happy to help! The answer to 2+3 is 5.
~~~

3. --prompt "could you please write code for fibonacci series in python?" --max-output-tokens 100
~~~
-------- Response Summary --------
Prompt: could you please write code for fibonacci series in python?
Response: Of course! Here is an example of how you could implement the Fibonacci sequence in Python:
```
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```
You can test the function by calling it with different values of `n`, like this:
```
print(fibonacci(5))
~~~





## License
* The license for the original implementation of Llama-v2-7B-Chat can be found
  [here](https://github.com/facebookresearch/llama/blob/main/LICENSE).
* The license for the compiled assets for on-device deployment can be found [here](https://github.com/facebookresearch/llama/blob/main/LICENSE)


## References
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Source Model Implementation](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)



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
