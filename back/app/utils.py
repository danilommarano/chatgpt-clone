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

