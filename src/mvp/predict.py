import torch
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.settings import TRAINING_ARGS, ID2LABEL, PREDICTION_PROB_THRESHOLD
from src.tokenizer.bpe_tokenizer import BPETokenizer
from transformers import AutoModelForSequenceClassification

def predict_emotion(text: str, model, bpe_tokenizer, id2label: dict, threshold: float):
    input_ids = bpe_tokenizer.encode(text)
    input_ids_tensor = torch.tensor([input_ids])
    attention_mask = torch.ones_like(input_ids_tensor)
    inputs = {
        "input_ids": input_ids_tensor.to(device),
        "attention_mask": attention_mask.to(device)
    }
    model.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]
    preds = [(id2label[i], p.item()) for i, p in enumerate(probs) if p.item()>threshold]
    return sorted(preds, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    model_path = f"{TRAINING_ARGS['output_dir']}/final_model"
    if not os.path.exists(model_path):
        print("…") ; sys.exit(1)

    # Charge BPE tokenizer et modèle UNE seule fois
    bpe_tokenizer = BPETokenizer(vocab_size=5000)
    bpe_tokenizer.load("data/tokenizer.json")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Prêt ! Entrez du texte…")

    while True:
        text = input("Votre texte: ")
        if text.lower()=="quitter": break
        preds = predict_emotion(text, model, bpe_tokenizer, ID2LABEL, PREDICTION_PROB_THRESHOLD)
        print(preds or "Aucune émotion prédite")
