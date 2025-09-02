# -*- coding: utf-8 -*-
"""
image_analyze.py
================

Инференс по изображению для мультимодальной классификации.

Функции:
- predict(img, model, preprocess, device="cpu", pos_index=-1) -> float
    Применяет CLIP-препроцессор и линейную «голову»; возвращает вероятность
    положительного класса (softmax).
- perform_ocr(img_path) -> str
    Опционально: OCR через tesseract (ru+en), может пригодиться для фичей.
"""

from typing import Optional
import numpy as np
from PIL import Image
import torch
import shutil, subprocess


@torch.inference_mode()
def predict(
        img: Image.Image,
        model: torch.nn.Module,
        preprocess,
        device: str = "cpu",
        pos_index: int = -1
) -> Optional[float]:
    """
    Получить вероятность положительного класса по изображению.

    Параметры
    ---------
    img : PIL.Image.Image
        Входное изображение.
    model : torch.nn.Module
        Модель вида "CLIP + линейная голова" (см. load_clip_and_head в main.py).
    preprocess : Callable
        Препроцессор из `clip.load(...)`.
    device : {"cpu","cuda"}, optional
        Устройство инференса.
    pos_index : int, optional
        Индекс положительного класса в логитах (обычно -1 для второго класса).

    Возвращает
    ----------
    float | None
        Вероятность положительного класса в [0,1] или None, если что-то пошло не так.
    """
    try:
        x = preprocess(img).unsqueeze(0).to(device)
        model = model.to(device).eval()
        logits = model(x)  # ожидается shape (1, 2)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs.squeeze(0)[pos_index].item())
    except Exception:
        # На случай битых изображений или несовместимых тензоров
        return None


def perform_ocr(img_path: str) -> str:
    """
    Выполнить OCR (rus+eng) для изображения по пути `img_path`.
    Требуется установленный tesseract и языковые пакеты.

    Возвращает распознанный текст (str). Пустая строка при неудаче.
    """
    try:
        if shutil.which("tesseract") is None:
            return ""
        langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True).stdout.decode("utf-8", "ignore")
        lang = "rus+eng" if "rus" in langs and "eng" in langs else ("eng" if "eng" in langs else None)
        cmd = ["tesseract", img_path, "-"] + (["-l", lang] if lang else [])
        out = subprocess.run(cmd, capture_output=True, check=False)
        return out.stdout.decode("utf-8", "ignore")
    except Exception:
        return ""
