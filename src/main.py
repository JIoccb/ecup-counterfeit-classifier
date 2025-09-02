# -*- coding: utf-8 -*-
"""
main.py
=======

CLI-скрипт для мультимодальной (текст + изображение + табличные признаки)
классификации товаров на «контрафакт» (1) или «оригинал» (0).
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


def find_image_path(row, image_folder: str, id_col: str, img_col: Optional[str], default_ext: Optional[str]) -> \
Optional[str]:
    if img_col and img_col in row and isinstance(row[img_col], str) and len(row[img_col]) > 0:
        p = os.path.join(image_folder, row[img_col])
        return p if os.path.isfile(p) else None
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
    vals = [v for v in probs if v is not None]
    if not vals:
        return 0.0
    if mode == "max":
        return max(vals)
    if mode == "mean":
        return float(np.mean(vals))
    if mode == "weighted":
        if weights is None or len(weights) != len(probs):
            return float(np.mean(vals))
        w = np.array([w for (w, v) in zip(weights, probs) if v is not None], dtype=float)
        v = np.array(vals, dtype=float)
        denom = w.sum() if w.sum() != 0 else 1.0
        return float((w * v).sum() / denom)
    raise ValueError(f"Unknown aggregation mode: {mode}")


def load_clip_and_head(clip_model_arg, classifier_path, device):
    """
    Загрузить базовый CLIP и линейную голову-классификатор.
    См. комментарии внутри. Возвращает (model_ft.eval(), preprocess).
    """
    download_root = None
    clip_name = None

    if clip_model_arg.lower().endswith(".pt"):
        clip_name = "ViT-B/32"
        download_root = os.path.dirname(clip_model_arg)
    else:
        clip_name = clip_model_arg

    clip_model, preprocess = clip.load(clip_name, device=device, download_root=download_root)

    class CLIPFineTuned(torch.nn.Module):
        def __init__(self, base_model, num_classes=2):
            super().__init__()
            self.base = base_model
            out_dim = getattr(self.base.visual, "output_dim", 512)
            self.classifier = torch.nn.Linear(out_dim, num_classes)

        def forward(self, x):
            with torch.no_grad():
                feats = self.base.encode_image(x).float()
            return self.classifier(feats)

    model_ft = CLIPFineTuned(clip_model, num_classes=2)
    state = torch.load(classifier_path, map_location=device)

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    if "classifier.weight" in state:
        model_ft.classifier.load_state_dict({k: v for k, v in state.items() if k.startswith("classifier.")},
                                            strict=False)
    else:
        model_ft.classifier.load_state_dict(state, strict=False)

    model_ft = model_ft.to(device).eval()  # <-- теперь переносим на устройство
    return model_ft, preprocess


def maybe_compute_best_threshold(y_true, y_score):
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main():
    parser = argparse.ArgumentParser(description="Multimodal (text+image+tabular) counterfeit classifier.")

    # --- Data & columns ---
    parser.add_argument("--csv_path", default="./data/data.csv", help="Путь к входному CSV с данными.")
    parser.add_argument("--image_folder", default="./data/imgs", help="Папка с изображениями.")
    parser.add_argument("--id_col", default="id", help="Имя столбца с идентификатором.")
    parser.add_argument("--name_col", default="name_rus", help="Имя столбца с названием товара.")
    parser.add_argument("--desc_col", default="description", help="Имя столбца с описанием товара.")
    parser.add_argument("--img_col", default=None, help="Имя столбца с именем файла изображения (если есть).")
    parser.add_argument("--img_ext", default="auto",
                        help='Явное расширение изображения, например ".jpg"; "auto" — перебор популярных.')

    # --- Models ---
    parser.add_argument("--text_model_dir", default="./text_model", help="Директория локальной HF-модели текста.")
    parser.add_argument("--clip_model", default="./image_model/ViT-B-32.pt",
                        help='Имя CLIP ("ViT-B/32") или путь к локальному .pt.')
    parser.add_argument("--classifier_path", default="./image_model/classifier_checkpoint3.pth",
                        help="Путь к state_dict линейной головы.")
    # фикс: путь по умолчанию соответствует структуре проекта
    parser.add_argument("--tabular_model_path", default="./table_model/f1_pipeline_final.pkl",
                        help="Путь к pickle со sklearn-пайплайном.")

    # --- Inference options ---
    parser.add_argument("--max_length", type=int, default=128, help="Макс. длина токенов для текста.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Устройство: "cpu"/"cuda".')
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Фиксированный порог бинаризации (если не используется --auto_threshold).")
    parser.add_argument("--auto_threshold", action="store_true",
                        help="Подобрать порог по F1, если в CSV есть колонка 'resolution'.")
    parser.add_argument("--agg", choices=["max", "mean", "weighted"], default="weighted",
                        help="Метод агрегации модальностей.")
    parser.add_argument("--w_text", type=float, default=0.5, help="Вес текстовой модальности для weighted.")
    parser.add_argument("--w_img", type=float, default=0.5, help="Вес визуальной модальности для weighted.")
    parser.add_argument("--w_tab", type=float, default=0.0, help="Вес табличной модальности для weighted.")
    parser.add_argument("--no_lemmatize", action="store_true",
                        help="Отключить лемматизацию текста (ускоряет инференс).")
    parser.add_argument("--disable_text", action="store_true", help="Отключить текстовую модальность.")
    parser.add_argument("--disable_image", action="store_true", help="Отключить визуальную модальность.")
    parser.add_argument("--disable_tabular", action="store_true", help="Отключить табличную модальность.")
    parser.add_argument(
        "--use_ocr",
        action="store_true",
        help="Подмешивать распознанный с изображения текст (tesseract rus+eng) к текстовой модальности."
    )
    # --- Output ---
    parser.add_argument("--out_csv", default="./results/submission.csv", help="Куда сохранить итоговый submission.csv.")
    args = parser.parse_args()
    use_ocr = getattr(args, "use_ocr", False)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # === 1) Загрузка данных ===
    df = pd.read_csv(args.csv_path)
    if args.id_col not in df.columns:
        if "ItemID" in df.columns:
            args.id_col = "ItemID"
        else:
            df[args.id_col] = np.arange(len(df))

    # === 2) Загрузка моделей ===
    text_model = None
    tokenizer = None
    if not args.disable_text:
        text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_dir, local_files_only=True)
        text_model.eval()

    img_model = None
    preprocess = None
    if not args.disable_image:
        img_model, preprocess = load_clip_and_head(args.clip_model, args.classifier_path, args.device)

    tabular_pipeline = None
    if not args.disable_tabular and os.path.isfile(args.tabular_model_path):
        import joblib
        tabular_pipeline = joblib.load(args.tabular_model_path)

    # === 3) Инференс по строкам ===
    text_probs, image_probs, tab_probs, final_probs = [], [], [], []

    for _, row in df.iterrows():
        # общий путь к картинке на итерацию
        ipath = find_image_path(
            row, args.image_folder, args.id_col, args.img_col, args.img_ext if args.img_ext != "auto" else None
        )

        # --- OCR (опционально) ---
        ocr_text = ""
        if use_ocr and ipath and os.path.isfile(ipath):
            ocr_text = image_analyze.perform_ocr(ipath) or ""

        # --- TEXT ---
        tprob = None
        if text_model is not None:
            parts = []
            if args.name_col in row and isinstance(row[args.name_col], str): parts.append(row[args.name_col])
            if args.desc_col in row and isinstance(row[args.desc_col], str): parts.append(row[args.desc_col])
            if ocr_text: parts.append(ocr_text)  # <-- подмешиваем OCR
            text = " ".join(parts) if parts else ""
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
        if img_model is not None and ipath and os.path.isfile(ipath):
            try:
                with Image.open(ipath) as img:
                    iprob = image_analyze.predict(img, img_model, preprocess, device=args.device, pos_index=-1)
            except Exception:
                iprob = None

        # --- TABULAR ---
        tabprob = None
        if tabular_pipeline is not None:
            tabprob = tabular_analyze.predict_tabular_prob(row, tabular_pipeline)

        # --- AGG ---
        probs = [tprob, iprob, tabprob]
        weights = [args.w_text, args.w_img, args.w_tab]
        fprob = aggregate(probs, args.agg, weights)

        final_probs.append(fprob)
        text_probs.append(0.0 if tprob is None else tprob)
        image_probs.append(0.0 if iprob is None else iprob)
        tab_probs.append(0.0 if tabprob is None else tabprob)

    # === 4) Бинаризация ===
    thr = args.threshold
    if args.auto_threshold and "resolution" in df.columns:
        y_true = df["resolution"].astype(int).values
        thr, best_f1 = maybe_compute_best_threshold(y_true, np.array(final_probs))
        print(f"[auto_threshold] best_threshold={thr:.3f}, best_f1={best_f1:.4f}")

    preds = (np.array(final_probs) >= thr).astype(int)

    # === 5) submission.csv ===
    out = pd.DataFrame({"id": df[args.id_col], "prediction": preds})
    out.to_csv(args.out_csv, index=False)
    print(f"Saved submission: {args.out_csv}  (N={len(out)})")

    if "resolution" in df.columns:
        f1 = f1_score(df["resolution"].astype(int).values, preds)
        print(f"F1@thr={thr:.3f}: {f1:.4f}")

    # === 6) debug-оценки ===
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
