import numpy as np
from PIL import Image
import torch


def predict(img: Image, model, preprocess):
    processed = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        return model(processed)