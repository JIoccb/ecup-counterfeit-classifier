
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

from . import text_analyze, image_analyze, tabular_analyze

def find_image_path(row, image_folder: str, id_col: str, img_col: Optional[str], default_ext: Optional[str]) -> Optional[str]:
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

def aggregate(probs: List[float], mode: str, weights: Optional[List[float]] = None) -> float:
    vals = [v for v in probs if v is not None]
    if not vals:
        return 0.0
    if mode == "max":
        return max(vals)
    if mode == "mean":
        return float(np.mean(vals))
    if mode == "weighted":
        if weights is None or len(weights) != len(probs):
            # equal weights for provided values
            return float(np.mean(vals))
        w = np.array([w for (w, v) in zip(weights, probs) if v is not None], dtype=float)
        v = np.array(vals, dtype=float)
        denom = w.sum() if w.sum() != 0 else 1.0
        return float((w * v).sum() / denom)
    raise ValueError(f"Unknown aggregation mode: {mode}")

def load_clip_and_head(clip_model_arg, classifier_path, device):
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
        # Only head passed
        model_ft.classifier.load_state_dict({k: v for k, v in state.items() if k.startswith("classifier.")}, strict=False)
    else:
        model_ft.classifier.load_state_dict(state, strict=False)
    model_ft.eval()
    return model_ft, preprocess

def maybe_compute_best_threshold(y_true, y_score):
    # compute best threshold to maximize F1 on provided scores
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])

def main():
    parser = argparse.ArgumentParser(description="Multimodal (text+image+tabular) counterfeit classifier.")
    # Data & columns
    parser.add_argument("--csv_path", default="./data/data.csv")
    parser.add_argument("--image_folder", default="./data/imgs")
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--name_col", default="name_rus")
    parser.add_argument("--desc_col", default="description")
    parser.add_argument("--img_col", default=None)
    parser.add_argument("--img_ext", default="auto")
    # Models
    parser.add_argument("--text_model_dir", default="./text_model")
    parser.add_argument("--clip_model", default="./image_model/ViT-B-32.pt")
    parser.add_argument("--classifier_path", default="./image_model/classifier_checkpoint3.pth")
    parser.add_argument("--tabular_model_path", default="./f1_pipeline_final.pkl")
    # Inference
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Base threshold if --auto_threshold not used.")
    parser.add_argument("--auto_threshold", action="store_true", help="If labels present in CSV (resolution), fit threshold for best F1.")
    parser.add_argument("--agg", choices=["max", "mean", "weighted"], default="weighted")
    parser.add_argument("--w_text", type=float, default=0.33)
    parser.add_argument("--w_img", type=float, default=0.34)
    parser.add_argument("--w_tab", type=float, default=0.33)
    parser.add_argument("--no_lemmatize", action="store_true")
    parser.add_argument("--disable_text", action="store_true")
    parser.add_argument("--disable_image", action="store_true")
    parser.add_argument("--disable_tabular", action="store_true")
    # Output
    parser.add_argument("--out_csv", default="./results/submission.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv_path)
    if args.id_col not in df.columns:
        # try ItemID
        if "ItemID" in df.columns:
            args.id_col = "ItemID"
        else:
            df[args.id_col] = np.arange(len(df))

    # Models
    # Text
    text_model = None; tokenizer = None
    if not args.disable_text:
        text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_dir, local_files_only=True)
        text_model.eval()

    # Image
    img_model = None; preprocess = None
    if not args.disable_image:
        img_model, preprocess = load_clip_and_head(args.clip_model, args.classifier_path, args.device)

    # Tabular
    tabular_pipeline = None
    if not args.disable_tabular and os.path.isfile(args.tabular_model_path):
        import joblib
        tabular_pipeline = joblib.load(args.tabular_model_path)

    # Predictions
    text_probs, image_probs, tab_probs, final_probs, preds = [], [], [], [], []

    for _, row in df.iterrows():
        # TEXT
        tprob = None
        if text_model is not None:
            parts = []
            if args.name_col in row and isinstance(row[args.name_col], str):
                parts.append(row[args.name_col])
            if args.desc_col in row and isinstance(row[args.desc_col], str):
                parts.append(row[args.desc_col])
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

        # IMAGE
        iprob = None
        if img_model is not None:
            ipath = find_image_path(row, args.image_folder, args.id_col, args.img_col, args.img_ext if args.img_ext != "auto" else None)
            if ipath and os.path.isfile(ipath):
                try:
                    with Image.open(ipath) as img:
                        iprob = image_analyze.predict(img, img_model, preprocess, device=args.device, pos_index=-1)
                except Exception:
                    iprob = None

        # TABULAR
        tabprob = None
        if tabular_pipeline is not None:
            tabprob = tabular_analyze.predict_tabular_prob(row, tabular_pipeline)

        # Combine
        probs = [tprob, iprob, tabprob]
        weights = [args.w_text, args.w_img, args.w_tab]
        fprob = aggregate(probs, args.agg, weights)
        final_probs.append(fprob)
        text_probs.append(0.0 if tprob is None else tprob)
        image_probs.append(0.0 if iprob is None else iprob)
        tab_probs.append(0.0 if tabprob is None else tabprob)

    # Thresholding
    thr = args.threshold
    if args.auto_threshold and "resolution" in df.columns:
        y_true = df["resolution"].astype(int).values
        thr, best_f1 = maybe_compute_best_threshold(y_true, np.array(final_probs))
        print(f"[auto_threshold] best_threshold={thr:.3f}, best_f1={best_f1:.4f}")

    preds = (np.array(final_probs) >= thr).astype(int)

    # Save submission
    out = pd.DataFrame({
        "id": df[args.id_col],
        "prediction": preds
    })
    out.to_csv(args.out_csv, index=False)
    print(f"Saved submission: {args.out_csv}  (N={len(out)})")

    # Report (optional, if ground truth available)
    if "resolution" in df.columns:
        f1 = f1_score(df["resolution"].astype(int).values, preds)
        print(f"F1@thr={thr:.3f}: {f1:.4f}")

    # Also dump modality scores for debugging (next to submission)
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
