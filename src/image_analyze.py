import numpy as np
from PIL import Image
import torch
import os
import subprocess


def predict(img: Image, model, preprocess):
    processed = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        return model(processed)
    

def perform_ocr(img_path: str):
    return subprocess.run(["tesseract", img_path, "-", "-l", "rus+eng"], capture_output=True).stdout.decode("utf-8")