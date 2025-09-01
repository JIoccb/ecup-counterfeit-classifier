# -*- coding: utf-8 -*-
"""
text_analyze.py
===============

Модуль для предобработки и инференса по тексту (название + описание).
Включает:
- базовую очистку текста (удаление HTML-тегов, служебных сущностей, знаков препинания и т.п.),
- лемматизацию русскоязычных токенов с помощью pymorphy3 (опционально),
- инференс модели-классификатора из HuggingFace Transformers.

Ожидания по входам
------------------
- model: AutoModelForSequenceClassification (или совместимая) с двумя классами.
- tokenizer: соответствующий токенизатор.
- text: сырой текст (str).
- max_length: максимальная длина токенов.
- device: "cpu" или "cuda".
- lemmatize: включать ли лемматизацию (True/False).
- pos_index: индекс положительного класса в логитах (по умолчанию -1).

Выход
-----
- float: вероятность положительного класса.
"""

import re
import unicodedata
import pymorphy3
import torch


def clean_text(text: str) -> str:
    """
    Базовая (языко-независимая) очистка текста.

    Шаги:
    - Приведение к строке и обработка None.
    - Удаление HTML-сущностей (&...;), HTML-тегов (<...>).
    - Удаление небуквенно-цифровых символов (кроме пробелов).
    - Сведение повторяющихся пробелов к одному.
    - Нормализация Unicode (NFKC).
    - Приведение к нижнему регистру.

    Параметры
    ---------
    text : str
        Входной текст.

    Возвращает
    ----------
    str
        Очищенный текст.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Удаляем HTML сущности и теги
    text = re.sub(r'&.+;', '', text)
    text = re.sub(r'<[^>]+>', '', text)

    # Разрешаем только буквы/цифры/пробелы (латиница+кириллица)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)

    # Схлопываем множественные пробелы
    text = re.sub(r'\s{2,}', ' ', text)

    # Нормализуем и приводим к нижнему регистру
    text = unicodedata.normalize('NFKC', text).lower()
    return text


# Инициализируем морф-анализатор один раз на модуль
_morph = pymorphy3.MorphAnalyzer()


def lemmatize_text(text: str) -> str:
    """
    Лемматизировать текст (в основном для русскоязычных токенов).

    Параметры
    ---------
    text : str
        Исходный текст.

    Возвращает
    ----------
    str
        Лемматизированная строка (токены разделены пробелами).
    """
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()

    # Для каждого токена берём нормальную форму первого (наиболее вероятного) разбора
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

    Параметры
    ---------
    text : str
        Сырой текст (например, "название + описание").
    model : transformers.PreTrainedModel
        Классификатор (2 класса).
    tokenizer : transformers.PreTrainedTokenizer
        Соответствующий токенизатор.
    max_length : int, optional
        Максимальная длина токенов (padding/truncation). По умолчанию 128.
    device : str, optional
        Устройство инференса ("cpu" / "cuda"). По умолчанию "cpu".
    lemmatize : bool, optional
        Применять ли лемматизацию русских токенов. По умолчанию True.
    pos_index : int, optional
        Индекс положительного класса в выходных логитах. По умолчанию -1.

    Возвращает
    ----------
    float
        Вероятность положительного класса.
    """
    # Предобработка: либо лемматизация, либо только очистка
    preprocessed_text = lemmatize_text(text) if lemmatize else clean_text(text)

    # Токенизация + приведение к тензорам
    inputs = tokenizer(
        preprocessed_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Инференс
    model = model.to(device).eval()
    output = model(**inputs)

    # Преобразуем логиты в вероятности
    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    return probs.squeeze(0)[pos_index].item()
