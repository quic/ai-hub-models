# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models._shared.opus_mt.app import OpusMTApp
from qai_hub_models.models._shared.opus_mt.model import OpusMT


def opus_mt_demo(
    model_cls: type[OpusMT], source_lang: str, target_lang: str, is_test: bool = False
) -> None:
    """
    Demo for OpusMT translation model.

    Parameters
    ----------
    model_cls
        Model class to use for the demo
    source_lang
        Source language name (e.g., "English", "Chinese")
    target_lang
        Target language name (e.g., "Chinese", "English")
    is_test
        Whether this is being run as a test
    """
    # Load model and tokenizer
    print(f"Loading OpusMT {source_lang} to {target_lang} model...")
    model = model_cls.from_pretrained()
    app = OpusMTApp(model.encoder, model.decoder, model.hf_source)

    # Example sentences to translate based on source language
    if is_test:
        # Use simple sentences for testing
        if "english" in source_lang.lower() or "en" in source_lang.lower():
            sentences = ["Hello, how are you?", "The weather is nice today."]
        elif "chinese" in source_lang.lower() or "zh" in source_lang.lower():
            sentences = ["你好,你好吗?", "今天天气很好。"]
        elif "spanish" in source_lang.lower() or "es" in source_lang.lower():
            sentences = ["Hola, ¿cómo estás?", "El clima es agradable hoy."]
        else:
            sentences = ["Hello, world!", "How are you?"]
    elif "english" in source_lang.lower() or "en" in source_lang.lower():
        sentences = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love learning new languages.",
            "Technology is changing the world.",
            "Machine translation helps people communicate.",
        ]
    elif "chinese" in source_lang.lower() or "zh" in source_lang.lower():
        sentences = [
            "你好,你好吗?",
            "今天天气很好。",
            "我喜欢学习新语言。",
            "技术正在改变世界。",
            "机器翻译帮助人们交流。",
        ]
    elif "spanish" in source_lang.lower() or "es" in source_lang.lower():
        sentences = [
            "Hola, ¿cómo estás?",
            "El clima es agradable hoy.",
            "Me encanta aprender nuevos idiomas.",
            "La tecnología está cambiando el mundo.",
            "La traducción automática ayuda a las personas a comunicarse.",
        ]
    else:
        sentences = [
            "Hello, world!",
            "How are you?",
            "This is a test sentence.",
            "Machine translation is useful.",
            "Thank you for using this demo.",
        ]

    print(f"\nTranslating {source_lang} sentences to {target_lang}:")
    print("=" * 60)

    max_length = 100 if not is_test else 20
    for sentence in sentences:
        print(f"\n{source_lang}: {sentence}")
        translation = app.translate(sentence, max_length=max_length)
        if translation:
            print(f"{target_lang}: {translation}")
        else:
            print(f"{target_lang}: [No translation generated]")

    print("\n" + "=" * 60)
    print("Demo completed!")
