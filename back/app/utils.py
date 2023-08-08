"""utils.py

ÚTEIS
--------------------------------------------------------------------------------
Funções utilitárias, que podem ser usadas em várias partes do código.
"""

from transformers import pipeline

def sentiment_analysis(text):
    classifier = pipeline("sentiment-analysis")
    res = classifier(text)
    return res


def text_generation(text):
    generator = pipeline("text-generation")
    res = generator(text, max_length=100, num_return_sequences=2)
    return res


def translation(text):
    translator = pipeline("translation_en_to_fr")
    res = translator(text)
    return res


