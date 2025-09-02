#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ./results

python -m src.main \
  --csv_path ./data/data.csv \
  --image_folder ./data/imgs \
  --text_model_dir ./text_model \
  --clip_model ./image_model/ViT-B-32.pt \
  --classifier_path ./image_model/classifier_checkpoint3.pth \
  --tabular_model_path ./f1_pipeline_final.pkl \
  --id_col id --name_col name_rus --desc_col description \
  --agg weighted --w_text 0.33 --w_img 0.34 --w_tab 0.33 \
  --out_csv ./results/submission.csv \
  "$@"
