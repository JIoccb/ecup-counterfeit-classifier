import numpy as np
import re
import unicodedata
import pymorphy3

def clean_text(text: str):
    text = re.sub(r'&.+;', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = unicodedata.normalize('NFKC', text).lower()
    return text

morph = pymorphy3.MorphAnalyzer()
def lemmatize_text(text: str):
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(lemmatized_tokens)


def predict(text: str):
    preprocessed_text = lemmatize_text(text)
    return 0