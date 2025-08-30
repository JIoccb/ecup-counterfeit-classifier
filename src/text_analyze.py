
import re
import unicodedata
import pymorphy3
import torch

def clean_text(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = re.sub(r'&.+;', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = unicodedata.normalize('NFKC', text).lower()
    return text

_morph = pymorphy3.MorphAnalyzer()
def lemmatize_text(text: str):
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    lemmatized_tokens = [_morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(lemmatized_tokens)

@torch.inference_mode()
def predict(text: str, model, tokenizer, max_length: int = 128, device: str = "cpu", lemmatize: bool = True, pos_index: int = -1) -> float:
    preprocessed_text = lemmatize_text(text) if lemmatize else clean_text(text)
    inputs = tokenizer(preprocessed_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device).eval()
    output = model(**inputs)
    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    return probs.squeeze(0)[pos_index].item()
