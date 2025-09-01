# -*- coding: utf-8 -*-
"""
image_analyze.py
================

Утилита для инференса по изображению: на вход подаётся объект PIL.Image,
а также модель и препроцессор (например, из CLIP). Возвращает вероятность
положительного класса.

Ожидания по входам
------------------
- img (PIL.Image.Image): загруженное изображение (RGB/RGBA — не важно, PIL сам конвертирует при необходимости).
- model (torch.nn.Module): модель, совместимая с интерфейсом forward(x) -> logits [B, C].
  В проекте это линейная «голова», обученная поверх визуальных фич CLIP.
- preprocess (callable): препроцессор, приводящий PIL.Image к тензору формата,
  ожидаемого моделью (нормализация, resize, center crop и т.п.).
- device (str): "cpu" или "cuda" (или "cuda:0" и т.п.). По умолчанию "cpu".
- pos_index (int): индекс положительного класса в логитах; по умолчанию -1 (последний столбец).

Выход
-----
- float: вероятность положительного класса (в интервале [0, 1]).

Пример
------
>>> from PIL import Image
>>> img = Image.open("some.jpg")
>>> prob = predict(img, model, preprocess, device="cuda", pos_index=-1)
>>> print(f"p(positive) = {prob:.3f}")
"""

from PIL import Image
import torch


@torch.inference_mode()
def predict(
    img: Image.Image,
    model,
    preprocess,
    device: str = "cpu",
    pos_index: int = -1
) -> float:
    """
    Выполнить предсказание вероятности положительного класса для одного изображения.

    Параметры
    ---------
    img : PIL.Image.Image
        Входное изображение.
    model : torch.nn.Module
        Модель классификации (возвращает логиты).
    preprocess : Callable
        Функция/трансформ, подготавливающая PIL.Image к тензору для модели.
    device : str, optional
        Устройство для инференса ("cpu" / "cuda"). По умолчанию "cpu".
    pos_index : int, optional
        Индекс положительного класса в логитах. По умолчанию -1.

    Возвращает
    ----------
    float
        Вероятность положительного класса.
    """
    # Преобразуем изображение в батч размера 1 и переносим на нужное устройство
    x = preprocess(img).unsqueeze(0).to(device)

    # Переводим модель на нужное устройство и в режим оценки
    model = model.to(device).eval()

    # Получаем логиты и переводим их в вероятности через softmax
    logits = model(x)
    probs = torch.nn.functional.softmax(logits, dim=1)

    # Забираем вероятность положительного класса
    return probs[0, pos_index].item()
