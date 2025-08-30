import numpy as np
import re
import unicodedata
import pymorphy3
import torch

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


def predict(text: str, model, tokenizer, max_length: int):
    preprocessed_text = lemmatize_text(text)
    inputs = tokenizer(preprocessed_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    probs = torch.nn.functional.softmax(output.logits, dim=-1)

    return probs.squeeze().tolist()[-1]