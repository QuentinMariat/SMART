# src/evaluate.py

import sys
import os
import json
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from src.data.data_handler import load_and_preprocess_data
from src.evaluation.metrics import compute_metrics

try:
    from src.config.settings import (
        EMOTION_LABELS,
        NUM_LABELS,
        PREDICTION_PROB_THRESHOLD
    )
    ALL_LABELS = EMOTION_LABELS

except ImportError as e:
    print(f"Erreur: Impossible d'importer une ou plusieurs variables depuis src.config.settings. Détails: {e}")
    sys.exit(1)

OUTPUT_DIR = "./eval_results"


def main(model_path):
    """
    charge le modèle sauvegardé et évalue ses performances sur le dataset de test.
    """
    print(f"Démarrage de l'évaluation du modèle depuis {model_path}")
    print(f"Labels d'émotion utilisés ({NUM_LABELS} au total): {ALL_LABELS}")
    print(f"Seuil de prédiction pour multi-label: {PREDICTION_PROB_THRESHOLD}")
    print(f"Dossier de sortie des résultats: {OUTPUT_DIR}")
    print("-" * 30)

    # chargement des données et du tokenizer en utilisant la fonction existante
    print("Chargement et prétraitement des données et du tokenizer...")
    try:
        train_dataset, val_dataset, test_dataset, tokenizer = load_and_preprocess_data()
        print("Données et tokenizer chargés et prétraités avec succès.")
        print(f"Nombre d'exemples dans le dataset de test: {len(test_dataset)}")
        print("-" * 30)

        if test_dataset and "labels" in test_dataset[0]:
             sample_labels = test_dataset[0]["labels"]
             print(f"Vérification du type des labels dans le dataset de test prétraité: {type(sample_labels)}")
             if isinstance(sample_labels, torch.Tensor):
                  print(f"Dtype du tenseur de labels: {sample_labels.dtype}")
             print("-" * 30)


    except Exception as e:
        print(f"Erreur lors du chargement ou du prétraitement des données: {e}")
        sys.exit(1)

    # chargement du modèle
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_LABELS)
        print(f"Modèle chargé avec succès depuis {model_path}")
        print("-" * 30)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle depuis {model_path}: {e}")
        sys.exit(1)

    # arguments du trainer pour l'évaluation
    evaluation_args = TrainingArguments(
        output_dir="./evaluation_output",
        per_device_eval_batch_size=64,
    )

    # config du data collector
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("Data collator configuré.")
    print("-" * 30)


    # init trainer
    trainer = Trainer(
        model=model,
        args=evaluation_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print("Trainer initialisé pour l'évaluation.")
    print("-" * 30)


    # évaluation
    print("Lancement de l'évaluation...")
    try:
        evaluation_results = trainer.evaluate()
        print("Évaluation terminée.")
        print("Résultats de l'évaluation:")
        print(evaluation_results)
        print("-" * 30)

    except Exception as e:
        print(f"Erreur lors de l'exécution de l'évaluation: {e}")
        sys.exit(1)


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")

    try:
        serializable_results = {k: (v.item() if hasattr(v, 'item') else v) for k, v in evaluation_results.items()}
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Résultats de l'évaluation sauvegardés dans {results_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Évaluer un modèle fine-tuné.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le répertoire contenant le modèle sauvegardé (e.g., ./results/final_model)')
    args = parser.parse_args()

    main(args.model_path)