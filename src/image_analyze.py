
from PIL import Image
import torch

@torch.inference_mode()
def predict(img: Image.Image, model, preprocess, device: str = "cpu", pos_index: int = -1) -> float:
    x = preprocess(img).unsqueeze(0).to(device)
    model = model.to(device).eval()
    logits = model(x)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs[0, pos_index].item()
