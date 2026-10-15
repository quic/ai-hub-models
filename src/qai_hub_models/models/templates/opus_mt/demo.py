# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.templates.opus_mt.app import OpusMTApp
from qai_hub_models.models.templates.opus_mt.model import OpusMT


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
        if source_lang.lower() == "english":
            sentences = ["Hello, how are you?", "The weather is nice today."]
        elif source_lang.lower() == "chinese":
            sentences = ["你好,你好吗?", "今天天气很好。"]
        elif source_lang.lower() == "spanish":
            sentences = ["Hola, ¿cómo estás?", "El clima es agradable hoy."]
        elif source_lang.lower() == "japanese":
            sentences = ["こんにちは お元気ですか?", "今日はいい天気です。"]
        elif source_lang.lower() == "korean":
            sentences = ["안녕하세요. 어떻게 지내세요?", "오늘 날씨가 좋네요."]
        elif source_lang.lower() == "german":
            sentences = ["Hallo, wie geht es dir?", "Das Wetter ist heute schön."]
        elif source_lang.lower() == "italian":
            sentences = ["Ciao, come stai?", "Il tempo è bello oggi."]
        elif source_lang.lower() == "french":
            sentences = ["Bonjour, comment allez-vous ?", "Il fait beau aujourd'hui."]
        else:
            sentences = ["Hello, world!", "How are you?"]
    elif source_lang.lower() == "english":
        sentences = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love learning new languages.",
            "Technology is changing the world.",
            "Machine translation helps people communicate.",
        ]
    elif source_lang.lower() == "chinese":
        sentences = [
            "你好,你好吗?",
            "今天天气很好。",
            "我喜欢学习新语言。",
            "技术正在改变世界。",
            "机器翻译帮助人们交流。",
        ]
    elif source_lang.lower() == "spanish":
        sentences = [
            "Hola, ¿cómo estás?",
            "El clima es agradable hoy.",
            "Me encanta aprender nuevos idiomas.",
            "La tecnología está cambiando el mundo.",
            "La traducción automática ayuda a las personas a comunicarse.",
        ]
    elif source_lang.lower() == "japanese":
        sentences = [
            "こんにちは お元気ですか?",
            "今日はいい天気です。",
            "新しい言語を学ぶのが大好きです。",
            "テクノロジーは世界を変えています。",
            "機械翻訳は、人々のコミュニケーションを助けます。",
        ]
    elif source_lang.lower() == "korean":
        sentences = [
            "안녕하세요. 어떻게 지내세요?",
            "오늘 날씨가 좋네요.",
            "저는 새로운 언어를 배우는 것을 좋아합니다.",
            "기술이 세상을 변화시키고 있습니다.",
            "기계 번역은 사람들이 의사소통하는 데 도움을 줍니다.",
        ]
    elif source_lang.lower() == "german":
        sentences = [
            "Hallo, wie geht es dir?",
            "Das Wetter ist heute schön.",
            "Ich lerne gerne neue Sprachen.",
            "Die Technologie verändert die Welt.",
            "Maschinelle Übersetzung hilft Menschen bei der Kommunikation.",
        ]
    elif source_lang.lower() == "italian":
        sentences = [
            "Ciao, come stai?",
            "Il tempo è bello oggi.",
            "Mi piace imparare nuove lingue.",
            "La tecnologia sta cambiando il mondo.",
            "La traduzione automatica aiuta le persone a comunicare.",
        ]
    elif source_lang.lower() == "french":
        sentences = [
            "Bonjour, comment allez-vous ?",
            "Il fait beau aujourd'hui.",
            "J'adore apprendre de nouvelles langues.",
            "La technologie change le monde.",
            "La traduction automatique aide les gens à communiquer.",
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
