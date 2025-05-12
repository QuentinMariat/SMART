# src/evaluate.py

import os
import sys
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.data.data_handler import load_and_preprocess_data
from src.evaluation.metrics import compute_metrics
from src.config.settings import EMOTION_LABELS, NUM_LABELS, PREDICTION_PROB_THRESHOLD

# --- Import et wrapper de ton BPE tokenizer ---
from src.tokenizer.bpe_tokenizer import BPETokenizer

class WrappedBPETokenizer(PreTrainedTokenizerBase):
    def __init__(self, bpe_tokenizer: BPETokenizer):
        super().__init__()
        self.tokenizer = bpe_tokenizer
        # Définir un pad_token_id unique à la taille du vocab
        self.pad_token = "<pad>"
        self.pad_token_id = len(self.tokenizer.vocab)

    def __call__(self, texts, padding=True, truncation=True, max_length=512, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Encodage BPE
        input_ids = [self.tokenizer.encode(t)[:max_length] for t in texts]
        # padding manuel
        if padding:
            max_len = max(len(ids) for ids in input_ids)
            input_ids = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids)

# Répertoire où on stocke les résultats
OUTPUT_DIR = "./eval_results"

def main(model_path):
    print(f"Démarrage de l'évaluation du modèle depuis {model_path}")
    print(f"Labels ({NUM_LABELS} totaux) : {EMOTION_LABELS}")
    print(f"Seuil multi-label : {PREDICTION_PROB_THRESHOLD}")
    print(f"Dossier de sortie : {OUTPUT_DIR}")
    print("-" * 30)

    # 1) Charger et wrapper ton BPE tokenizer
    bpe = BPETokenizer(vocab_size=5000)
    bpe.load("data/tokenizer_files/tokenizer.json")
    tokenizer = WrappedBPETokenizer(bpe)

    # 2) Charger et prétraiter les données en passant TON tokenizer
    try:
        train_ds, val_ds, test_ds = load_and_preprocess_data(tokenizer=tokenizer)
        print(f"Données et tokenizer chargés. Nombre d'exemples test : {len(test_ds)}")
        print("-" * 30)
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        sys.exit(1)

    # 3) Charger le modèle fine-tuné
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)
        print("Modèle chargé avec succès.")
        print("-" * 30)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # 4) Configuration du Trainer pour évaluation
    eval_args = TrainingArguments(
        output_dir="./evaluation_output",
        per_device_eval_batch_size=64,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("Data collator configuré.")
    print("-" * 30)

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print("Trainer initialisé pour l'évaluation.")
    print("-" * 30)

    # 5) Lancer l’évaluation
    print("Lancement de l'évaluation...")
    try:
        results = trainer.evaluate()
        print("Évaluation terminée.")
        print(results)
        print("-" * 30)
    except Exception as e:
        print(f"Erreur pendant l'évaluation : {e}")
        sys.exit(1)

    # 6) Sauvegarder les résultats
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    serializable = {k: (v.item() if hasattr(v, "item") else v) for k, v in results.items()}
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Résultats enregistrés dans {results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Évaluer un modèle fine-tuné")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Chemin vers le dossier du modèle (e.g., ./results/final_model)")
    args = parser.parse_args()
    main(args.model_path)
