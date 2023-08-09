"""utils.py

ÚTEIS
--------------------------------------------------------------------------------
Funções utilitárias, que podem ser usadas em várias partes do código.
"""

from transformers import pipeline

def pipe(text, model):
    processor = pipeline(model)
    res = processor(text)
    return res


def audio_classification(text):
    res = pipe("audio-classification")
    return res


def automatic_speech_recognition(text):
    res = pipe("automatic-speech-recognition")
    return res


def conversational(text):
    res = pipe("conversational")
    return res


def depth_estimation(text):
    res = pipe("depth-estimation")
    return res


def document_question_answering(text):
    res = pipe("document-question-answering")
    return res


def feature_extraction(text):
    res = pipe("feature-extraction")
    return res


def fill_mask(text):
    res = pipe("fill-mask")
    return res


def image_classification(text):
    res = pipe("image-classification")
    return res


def image_segmentation(text):
    res = pipe("image-segmentation")
    return res


def image_to_text(text):
    res = pipe("image-to-text")
    return res


def mask_generation(text):
    res = pipe("mask-generation")
    return res


def object_detection(text):
    res = pipe("object-detection")
    return res


def question_answering(text):
    res = pipe("question-answering")
    return res


def summarization(text):
    res = pipe("summarization")
    return res


def table_question_answering(text):
    res = pipe("table-question-answering")
    return res


def text2text_generation(text):
    res = pipe("text2text-generation")
    return res


def text_classification(text):
    res = pipe("text-classification")
    return res


def text_generation(text):
    res = pipe("text-generation")
    return res


def token_classification(text):
    res = pipe("token-classification")
    return res


def translation(text):
    res = pipe("translation")
    return res


def translation_xx_to_yy(text):
    res = pipe("translation_xx_to_yy")
    return res


def video_classification(text):
    res = pipe("video-classification")
    return res


def visual_question_answering(text):
    res = pipe("visual-question-answering")
    return res


def zero_shot_classification(text):
    res = pipe("zero-shot-classification")
    return res


def zero_shot_image_classification(text):
    res = pipe("zero-shot-image-classification")
    return res


def zero_shot_audio_classification(text):
    res = pipe("zero-shot-audio-classification")
    return res


def zero_shot_object_detection(text):
    res = pipe("zero-shot-object-detection")
    return res




