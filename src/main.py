import numpy as np
import pandas as pd
from PIL import Image
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import clip

import text_analyze
import image_analyze

filename = "./data/data.csv"
image_folder = "./data/imgs"

enable_ocr = True

### TEXT MODEL

text_model_folder = "./text_model"

model = AutoModelForSequenceClassification.from_pretrained(text_model_folder, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(text_model_folder, local_files_only=True)
model.eval()

### IMAGES MODEL

class CLIPFineTuned(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuned, self).__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        return self.classifier(features)
    
model_clip, preprocess = clip.load("./image_model/ViT-B-32.pt", device="cpu")
model_ft = CLIPFineTuned(model_clip, 2)
model_ft.classifier.load_state_dict(torch.load("./image_model/classifier_checkpoint3.pth", weights_only=True, map_location=torch.device("cpu")))
model_ft.eval()

###

raw = pd.read_csv(filename)
result_arr = []

for index, row in raw.iterrows():
    description = row["description"]
    name = row["name_rus"]
    text = str(name) + " " + str(description)
    if enable_ocr:
        text += " " + image_analyze.perform_ocr(os.path.join(image_folder, str(row["ItemID"])) + ".png")

    # image = Image.open(os.path.join(image_folder, str(row["ItemID"])))
    image = Image.open(os.path.join(image_folder, "100481.png")) # CHANGE THISSSSSS

    text_prob = text_analyze.predict(text, model, tokenizer, 128)
    image_prob = image_analyze.predict(image, model_ft, preprocess).max()

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