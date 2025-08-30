import numpy as np
import pandas as pd
from PIL import Image
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import text_analyze
import image_analyze

filename = "./data/data.csv"

image_folder = "./data/imgs"

text_model_folder = "./text_model"

model = AutoModelForSequenceClassification.from_pretrained(text_model_folder, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(text_model_folder, local_files_only=True)
model.eval()

raw = pd.read_csv(filename)
result_arr = []

for index, row in raw.iterrows():
    description = row["description"]
    name = row["name_rus"]
    text = str(name) + " " + str(description)

    # image = Image.open(image_folder + "/" + str(row["ItemID"]))
    image = None

    text_prob = text_analyze.predict(text, model, tokenizer, 128)
    image_prob = image_analyze.predict(image)

    # random forest
    result_prob = max(text_prob, image_prob)

    
    if result_prob > 0.5:
        result_arr.append(1)
    else:
        result_arr.append(0)

result = pd.DataFrame({
    "id": raw["id"],
    "resolution": result_arr
})

result.to_csv("./results/result.csv", index=False)