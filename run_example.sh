#!/usr/bin/env bash
set -e
python -m src.main       --csv_path ./data/data.csv       --image_folder ./data/imgs       --text_model_dir ./text_model       --clip_model ./image_model/ViT-B-32.pt       --classifier_path ./image_model/classifier_checkpoint3.pth       --tabular_model_path ./table_model/f1_pipeline_final.pkl       --id_col id --name_col name_rus --desc_col description       --out_csv ./results/submission.csv       --agg weighted --w_text 0.33 --w_img 0.34 --w_tab 0.33
