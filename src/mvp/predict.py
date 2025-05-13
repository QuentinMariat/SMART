# src/predict.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
import sys
import numpy as np

from src.config.settings import TRAINING_ARGS, ID2LABEL, PREDICTION_PROB_THRESHOLD, BASE_MODEL_NAME

def predict_emotion(text: str, model, tokenizer, id2label: dict, threshold: float):
    """
    Prédit les émotions pour un texte donné en utilisant le modèle chargé.
    """
    # tokenization : utiliser les mêmes paramètres de padding et truncation que pendant l'entraînement
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # envoyer les entrées sur le même appareil que le modèle 
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # obtenir les logits et les convertir en proba
    logits = outputs.logits
    probabilities = torch.sigmoid(logits) 

    # appliquer le seuil pour obtenir prédictions binaires
    predicted_emotions_with_probs = []
    for i, prob in enumerate(probabilities[0]):
        if prob.item() > threshold:
            predicted_emotions_with_probs.append((id2label[i], prob.item()))

    # triez par proba décroissante
    predicted_emotions_with_probs.sort(key=lambda item: item[1], reverse=True)


    return predicted_emotions_with_probs

if __name__ == "__main__":
    model_path = f"{TRAINING_ARGS['output_dir']}/final_model"

    if not os.path.exists(model_path):
        print(f"Erreur: Le dossier du modèle '{model_path}' n'existe pas.")
        print("Veuillez d'abord exécuter le script d'entraînement pour sauvegarder le modèle.")
        sys.exit(1)

    print(f"Chargement du modèle et du tokenizer depuis {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        print("Modèle et tokenizer chargés avec succès.")
        print(f"Seuil de prédiction utilisé: {PREDICTION_PROB_THRESHOLD}")
        print("-" * 30)

        print("Entrez un texte pour prédire les émotions. Tapez 'quitter' pour sortir.")
        while True:
            user_input = input("Votre texte: ")
            if user_input.lower() == 'quitter':
                break

            if user_input.strip() == "":
                print("Veuillez entrer un texte non vide.")
                continue

            predicted_emotions = predict_emotion(user_input, model, tokenizer, ID2LABEL, PREDICTION_PROB_THRESHOLD)

            if predicted_emotions:
                print("Émotions prédites (nom, probabilité):")
                for emotion, prob in predicted_emotions:
                    print(f"- {emotion}: {prob:.4f}")
            else:
                print("Aucune émotion prédite au seuil spécifié.")
            print("-" * 30)

    except Exception as e:
        print(f"Une erreur est survenue lors du chargement ou de la prédiction: {e}")
        sys.exit(1)

    print("Fin du programme de prédiction.")