# -*- coding: utf-8 -*-
"""
main.py
=======

CLI-скрипт для мультимодальной (текст + изображение + табличные признаки)
классификации товаров на «контрафакт» (1) или «оригинал» (0).

Пайплайн инференса
------------------
1) Чтение CSV с данными и определение `id_col` (если нет — генерируем индексы).
2) Загрузка трёх моделей (по флагам можно отключать каждую модальность):
   - Текст: локальная директория HuggingFace (AutoModelForSequenceClassification + токенизатор).
   - Изображение: CLIP + линейная «голова» (чекпойнт классификатора).
   - Табличные признаки: sklearn-пайплайн (pickle).
3) Для каждой строки:
   - Сформировать текст (name_col + desc_col) → получить p_text.
   - Найти путь к картинке → получить p_image.
   - Вычислить p_tabular из табличных признаков.
   - Объединить вероятности (mean / max / weighted) → p_final.
4) Бинаризация по порогу:
   - По умолчанию используется фиксированный `--threshold`.
   - Если указан `--auto_threshold` и есть колонка `resolution`, подбирается порог,
     максимизирующий F1 на текущем CSV.
5) Сохранить `submission.csv` с колонками: `id,prediction`.
6) Дополнительно — сохранить отладочный CSV со всеми вероятностями.

Аргументы см. в парсере ниже (`--help`).
"""

import argparse
import os
from typing import Optional, List
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score

# Локальные модули инференса по модальностям
from . import text_analyze, image_analyze, tabular_analyze


def find_image_path(
    row,
    image_folder: str,
    id_col: str,
    img_col: Optional[str],
    default_ext: Optional[str]
) -> Optional[str]:
    """
    Найти путь к изображению для строки датасета.

    Логика:
    1) Если задан `img_col` и там есть валидное имя файла — склеиваем с `image_folder` и проверяем наличие.
    2) Иначе — пробуем по `id_col` + расширение(я):
       - Если `default_ext` задан и не "auto", используем только его.
       - Иначе — перебираем популярные: .png, .jpg, .jpeg, .webp.

    Параметры
    ---------
    row : pandas.Series
        Строка исходного DataFrame.
    image_folder : str
        Папка с изображениями.
    id_col : str
        Имя столбца с идентификатором.
    img_col : str | None
        Имя столбца с именем файла изображения (если есть).
    default_ext : str | None
        Явное расширение (например, ".jpg"). Если None — считаем, что "auto".

    Возвращает
    ----------
    str | None
        Полный путь к существующему файлу изображения или None, если не найден.
    """
    # Вариант 1: прямое имя файла в таблице
    if img_col and img_col in row and isinstance(row[img_col], str) and len(row[img_col]) > 0:
        p = os.path.join(image_folder, row[img_col])
        return p if os.path.isfile(p) else None

    # Вариант 2: подбор по id и расширению
    if id_col in row:
        base = str(row[id_col])
        exts = [default_ext] if (default_ext and default_ext != "auto") else [".png", ".jpg", ".jpeg", ".webp"]
        for ext in exts:
            if not ext:
                continue
            p = os.path.join(image_folder, base + ext)
            if os.path.isfile(p):
                return p
    return None


def aggregate(probs: List[Optional[float]], mode: str, weights: Optional[List[float]] = None) -> float:
    """
    Агрегация вероятностей из модальностей в единую оценку.

    Параметры
    ---------
    probs : list[float | None]
        [p_text, p_image, p_tabular], некоторые могут быть None (если модальность отключена/не сработала).
    mode : {"max", "mean", "weighted"}
        Способ агрегации:
        - "max": берём максимум из доступных;
        - "mean": простое среднее по доступным;
        - "weighted": взвешенное среднее (если веса не заданы или размер не совпадает — скатываемся в mean).
    weights : list[float] | None, optional
        Веса для "weighted". Порядок соответствует probs.

    Возвращает
    ----------
    float
        Финальная сводная вероятность.
    """
    # Отбрасываем отсутствующие значения
    vals = [v for v in probs if v is not None]
    if not vals:
        # Ничего не получилось посчитать — возвращаем «самый безопасный» ноль
        return 0.0

    if mode == "max":
        return max(vals)

    if mode == "mean":
        return float(np.mean(vals))

    if mode == "weighted":
        if weights is None or len(weights) != len(probs):
            # Если весов нет или они не согласованы — деградируем в простое среднее по имеющимся значениям
            return float(np.mean(vals))
        # Сопоставляем веса только для тех модальностей, где есть значение
        w = np.array([w for (w, v) in zip(weights, probs) if v is not None], dtype=float)
        v = np.array(vals, dtype=float)
        denom = w.sum() if w.sum() != 0 else 1.0
        return float((w * v).sum() / denom)

    raise ValueError(f"Unknown aggregation mode: {mode}")


def load_clip_and_head(clip_model_arg, classifier_path, device):
    """
    Загрузить базовый CLIP и линейную голову-классификатор.

    Поддерживаются два сценария:
    - clip_model_arg = "ViT-B/32" (или другое имя) → CLIP скачивается/берётся из кеша.
    - clip_model_arg = путь к .pt → в этом случае считаем, что это локальный чекпойнт CLIP,
      и используем стандартное имя "ViT-B/32" + download_root=директория файла.

    classifier_path — это обученная линейная «голова» (state_dict), совместимая
    с классом CLIPFineTuned ниже. Возможны два формата state_dict:
      1) полный state_dict только для classifier.*,
      2) state_dict с префиксом "module." (после DDP) — будет автоматически очищен.

    Возвращает
    ----------
    (model_ft, preprocess)
      model_ft : torch.nn.Module
          Обёртка над CLIP, которая извлекает визуальные фичи и прогоняет их через линейный слой.
      preprocess : Callable
          Препроцессор изображений CLIP.
    """
    download_root = None
    clip_name = None

    # Если передан путь к *.pt — предполагаем локальный чекпойнт
    if clip_model_arg.lower().endswith(".pt"):
        clip_name = "ViT-B/32"
        download_root = os.path.dirname(clip_model_arg)
    else:
        clip_name = clip_model_arg

    # Загружаем CLIP и его препроцессор
    clip_model, preprocess = clip.load(clip_name, device=device, download_root=download_root)

    class CLIPFineTuned(torch.nn.Module):
        """
        Обёртка над CLIP: freeze-им base.encode_image (через no_grad) и добавляем линейный классификатор.
        """
        def __init__(self, base_model, num_classes=2):
            super().__init__()
            self.base = base_model
            out_dim = getattr(self.base.visual, "output_dim", 512)
            self.classifier = torch.nn.Linear(out_dim, num_classes)

        def forward(self, x):
            # Важно: фичи CLIP не обучаем (no_grad), только используем
            with torch.no_grad():
                feats = self.base.encode_image(x).float()
            return self.classifier(feats)

    # Инициализируем голову и подгружаем веса классификатора
    model_ft = CLIPFineTuned(clip_model, num_classes=2)
    state = torch.load(classifier_path, map_location=device)

    # Снимаем префикс "module." при необходимости
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Поддержка случая, когда в стейте есть ключи "classifier.*"
    if "classifier.weight" in state:
        # Передан только head — загружаем из подмножества ключей
        model_ft.classifier.load_state_dict({k: v for k, v in state.items() if k.startswith("classifier.")}, strict=False)
    else:
        # На всякий случай допускаем и другой формат (загрузим что сможем)
        model_ft.classifier.load_state_dict(state, strict=False)

    model_ft.eval()
    return model_ft, preprocess


def maybe_compute_best_threshold(y_true, y_score):
    """
    Подобрать лучший порог для максимизации F1 на валидационных скорых.

    Параметры
    ---------
    y_true : array-like of shape (N,)
        Истинные бинарные метки {0,1}.
    y_score : array-like of shape (N,)
        Вероятности/оценки положительного класса.

    Возвращает
    ----------
    (best_thr, best_f1) : tuple[float, float]
        Лучший порог и соответствующий F1.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main():
    """
    Точка входа CLI. См. описание аргументов ниже или вызовите `python -m ... --help`.
    """
    parser = argparse.ArgumentParser(description="Multimodal (text+image+tabular) counterfeit classifier.")

    # --- Data & columns ---
    parser.add_argument("--csv_path", default="./data/data.csv", help="Путь к входному CSV с данными.")
    parser.add_argument("--image_folder", default="./data/imgs", help="Папка с изображениями.")
    parser.add_argument("--id_col", default="id", help="Имя столбца с идентификатором.")
    parser.add_argument("--name_col", default="name_rus", help="Имя столбца с названием товара.")
    parser.add_argument("--desc_col", default="description", help="Имя столбца с описанием товара.")
    parser.add_argument("--img_col", default=None, help="Имя столбца с именем файла изображения (если есть).")
    parser.add_argument("--img_ext", default="auto", help='Явное расширение изображения, например ".jpg"; "auto" — перебор популярных.')

    # --- Models ---
    parser.add_argument("--text_model_dir", default="./text_model", help="Директория локальной HF-модели текста.")
    parser.add_argument("--clip_model", default="./image_model/ViT-B-32.pt", help='Имя CLIP ("ViT-B/32") или путь к локальному .pt.')
    parser.add_argument("--classifier_path", default="./image_model/classifier_checkpoint3.pth", help="Путь к state_dict линейной головы.")
    parser.add_argument("--tabular_model_path", default="./f1_pipeline_final.pkl", help="Путь к pickle со sklearn-пайплайном для табличных признаков.")

    # --- Inference options ---
    parser.add_argument("--max_length", type=int, default=128, help="Макс. длина токенов для текста.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help='Устройство: "cpu"/"cuda".')
    parser.add_argument("--threshold", type=float, default=0.5, help="Фиксированный порог бинаризации (если не используется --auto_threshold).")
    parser.add_argument("--auto_threshold", action="store_true", help="Подобрать порог по F1, если в CSV есть колонка 'resolution'.")
    parser.add_argument("--agg", choices=["max", "mean", "weighted"], default="weighted", help="Метод агрегации модальностей.")
    parser.add_argument("--w_text", type=float, default=0.5, help="Вес текстовой модальности для weighted.")
    parser.add_argument("--w_img", type=float, default=0.5, help="Вес визуальной модальности для weighted.")
    parser.add_argument("--w_tab", type=float, default=0, help="Вес табличной модальности для weighted.")
    parser.add_argument("--no_lemmatize", action="store_true", help="Отключить лемматизацию текста (ускоряет инференс).")
    parser.add_argument("--disable_text", action="store_true", help="Отключить текстовую модальность.")
    parser.add_argument("--disable_image", action="store_true", help="Отключить визуальную модальность.")
    parser.add_argument("--disable_tabular", action="store_true", help="Отключить табличную модальность.")

    # --- Output ---
    parser.add_argument("--out_csv", default="./results/submission.csv", help="Куда сохранить итоговый submission.csv.")
    args = parser.parse_args()

    # Гарантируем существование папки для вывода
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # === 1) Загрузка данных ===
    df = pd.read_csv(args.csv_path)

    # Если явно заданного id-столбца нет — пытаемся использовать ItemID; иначе создаём индексы 0..N-1
    if args.id_col not in df.columns:
        if "ItemID" in df.columns:
            args.id_col = "ItemID"
        else:
            df[args.id_col] = np.arange(len(df))

    # === 2) Загрузка моделей ===
    # Текстовая модель
    text_model = None
    tokenizer = None
    if not args.disable_text:
        # Важно: local_files_only=True — модель должна быть заранее скачана/скопирована в text_model_dir
        text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_dir, local_files_only=True)
        text_model.eval()

    # Визуальная модель (CLIP + голова)
    img_model = None
    preprocess = None
    if not args.disable_image:
        img_model, preprocess = load_clip_and_head(args.clip_model, args.classifier_path, args.device)

    # Табличная модель
    tabular_pipeline = None
    if not args.disable_tabular and os.path.isfile(args.tabular_model_path):
        import joblib
        tabular_pipeline = joblib.load(args.tabular_model_path)

    # === 3) Инференс по строкам ===
    text_probs, image_probs, tab_probs, final_probs = [], [], [], []

    for _, row in df.iterrows():
        # --- TEXT ---
        tprob = None
        if text_model is not None:
            parts = []
            if args.name_col in row and isinstance(row[args.name_col], str):
                parts.append(row[args.name_col])
            if args.desc_col in row and isinstance(row[args.desc_col], str):
                parts.append(row[args.desc_col])
            text = " ".join(parts) if parts else ""

            # pos_index=-1 -> считаем, что второй класс (index 1) — «положительный», но оставляем гибкость
            tprob = text_analyze.predict(
                text=text,
                model=text_model,
                tokenizer=tokenizer,
                max_length=args.max_length,
                device=args.device,
                lemmatize=(not args.no_lemmatize),
                pos_index=-1,
            )

        # --- IMAGE ---
        iprob = None
        if img_model is not None:
            # Находим путь к картинке: либо по имени столбца, либо по id+расширение
            ipath = find_image_path(
                row,
                args.image_folder,
                args.id_col,
                args.img_col,
                args.img_ext if args.img_ext != "auto" else None
            )
            if ipath and os.path.isfile(ipath):
                try:
                    # Открываем изображение безопасно (контекстный менеджер гарантирует закрытие файла)
                    with Image.open(ipath) as img:
                        iprob = image_analyze.predict(img, img_model, preprocess, device=args.device, pos_index=-1)
                except Exception:
                    # На практике могут попадаться битые/нестандартные файлы изображений — игнорируем и продолжаем
                    iprob = None

        # --- TABULAR ---
        tabprob = None
        if tabular_pipeline is not None:
            tabprob = tabular_analyze.predict_tabular_prob(row, tabular_pipeline)

        # --- Агрегация трёх модальностей ---
        probs = [tprob, iprob, tabprob]
        weights = [args.w_text, args.w_img, args.w_tab]
        fprob = aggregate(probs, args.agg, weights)

        final_probs.append(fprob)
        text_probs.append(0.0 if tprob is None else tprob)
        image_probs.append(0.0 if iprob is None else iprob)
        tab_probs.append(0.0 if tabprob is None else tabprob)

    # === 4) Пороговая бинаризация ===
    thr = args.threshold
    if args.auto_threshold and "resolution" in df.columns:
        # Если есть разметка — подбираем порог, максимизирующий F1 на текущем наборе
        y_true = df["resolution"].astype(int).values
        thr, best_f1 = maybe_compute_best_threshold(y_true, np.array(final_probs))
        print(f"[auto_threshold] best_threshold={thr:.3f}, best_f1={best_f1:.4f}")

    preds = (np.array(final_probs) >= thr).astype(int)

    # === 5) Сохранение submission.csv ===
    out = pd.DataFrame({
        "id": df[args.id_col],
        "prediction": preds
    })
    out.to_csv(args.out_csv, index=False)
    print(f"Saved submission: {args.out_csv}  (N={len(out)})")

    # Доп. отчёт по F1, если есть колонка с истиной
    if "resolution" in df.columns:
        f1 = f1_score(df["resolution"].astype(int).values, preds)
        print(f"F1@thr={thr:.3f}: {f1:.4f}")

    # === 6) Сохранение отладочных вероятностей ===
    dbg = pd.DataFrame({
        args.id_col: df[args.id_col],
        "text_prob": text_probs,
        "image_prob": image_probs,
        "tabular_prob": tab_probs,
        "final_prob": final_probs,
        "pred": preds
    })
    debug_csv = os.path.splitext(args.out_csv)[0].replace("submission", "debug") + ".csv"
    dbg.to_csv(debug_csv, index=False)
    print(f"Saved debug scores: {debug_csv}")


if __name__ == "__main__":
    main()