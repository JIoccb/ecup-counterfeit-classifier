# -*- coding: utf-8 -*-
"""
text_analyze.py
===============

Модуль для предобработки и инференса по тексту (название + описание).
"""

import re
import unicodedata
import torch

# Пытаемся импортировать pymorphy3; если его нет, работаем без лемматизации
try:
    import pymorphy3

    _morph = pymorphy3.MorphAnalyzer()
except Exception:
    _morph = None  # fallback


def clean_text(text: str) -> str:
    """
    Базовая (языко-независимая) очистка текста.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Удаляем HTML сущности и теги
    text = re.sub(r'&.+?;', '', text)  # чутка менее жадный вариант
    text = re.sub(r'<[^>]+>', '', text)

    # Разрешаем только буквы/цифры/пробелы (латиница+кириллица)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)

    # Схлопываем множественные пробелы
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # Нормализуем и приводим к нижнему регистру
    text = unicodedata.normalize('NFKC', text).lower()
    return text


def lemmatize_text(text: str) -> str:
    """
    Лемматизировать текст (если доступен pymorphy3). Иначе — вернёт clean_text.
    """
    if _morph is None:
        return clean_text(text)

    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    lemmatized_tokens = [_morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(lemmatized_tokens)


@torch.inference_mode()
def predict(
        text: str,
        model,
        tokenizer,
        max_length: int = 128,
        device: str = "cpu",
        lemmatize: bool = True,
        pos_index: int = -1
) -> float:
    """
    Получить вероятность положительного класса для текстового описания.
    """
    preprocessed_text = lemmatize_text(text) if lemmatize else clean_text(text)

    inputs = tokenizer(
        preprocessed_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = model.to(device).eval()
    output = model(**inputs)
    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    return probs.squeeze(0)[pos_index].item()
