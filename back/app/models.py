"""models.py

MODELOS
--------------------------------------------------------------------------------
Neste arquivo estão todos os modelos usados na aplicação. Todos os modelos 
definidos aqui são extraídos da plataforma HugginFace.
(https://huggingface.co/)

São definidas as seguintes tarefas de NLP neste arquivo:
- question-answering
- summarization [ ]
- text-classification [X]
- text-generation [ ]
- text-generation [ ]
- translation [X]

"""
from transformers import pipeline


MODEL_TRANSLATION_EN_PT = "unicamp-dl/translation-en-pt-t5"
MODEL_TRANSLATION_PT_EN = "unicamp-dl/translation-en-pt-t5"
MODEL_TEXT_CLASSIFICATION = "ProsusAI/finbert"
MODEL_ZERO_SHOT_CLASSIFICATION = "facebook/bart-large-mnli"


translation_en_pt = pipeline("translation", model=MODEL_TRANSLATION_EN_PT)
translation_pt_en = pipeline("translation", model=MODEL_TRANSLATION_PT_EN)

zero_shot_classification = pipeline("zero_shot-classification", model=MODEL_ZERO_SHOT_CLASSIFICATION)

text_classification = pipeline("text-classification", model=MODEL_TEXT_CLASSIFICATION)
