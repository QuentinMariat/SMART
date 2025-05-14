# src/mvp/predict.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
import sys
import numpy as np
import re

try:
    from src.config.settings import TRAINING_ARGS, ID2LABEL, PREDICTION_PROB_THRESHOLD, BASE_MODEL_NAME
except ImportError:
    # Fallback pour les imports relatifs dans le contexte du backend
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    sys.path.append(project_root)
    from src.config.settings import TRAINING_ARGS, ID2LABEL, PREDICTION_PROB_THRESHOLD, BASE_MODEL_NAME

def detect_gratitude_patterns(text):
    """
    Détecte les patterns implicites de gratitude et autres expressions positives.
    
    Args:
        text (str): Le texte à analyser
        
    Returns:
        bool: True si un pattern est détecté, False sinon
    """
    text = text.lower()
    
    # Patterns explicites de gratitude
    gratitude_patterns = [
        r"thank(s|\syou)", r"helped( me)?", r"(was|very) helpful",
        r"save[sd]( me| my)?", r"(great|good) job", r"well done",
        r"love (this|it|how|the way)", r"appreciate", r"grateful",
        r"(this|that) help(s|ed)", r"(this|that) (is|was) (great|good|helpful|useful)"
    ]
    
    for pattern in gratitude_patterns:
        if re.search(pattern, text):
            return True
            
    # Détection du contexte "help with exam/study"
    study_patterns = [
        r"(help(ed)?|assist(ed)?) (me )?(with )?(my )?(exam|study|test|homework)",
        r"exam.{1,15}help", r"study.{1,15}help",
        r"(got|received|needed) help",
        r"help.{1,15}(study|exam|test)",
        r"thanks to (this|you|him|her|them)"
    ]
    
    for pattern in study_patterns:
        if re.search(pattern, text):
            return True
            
    return False

def predict_emotion(text: str, model, tokenizer, id2label: dict, threshold: float):
    """
    Prédit l'émotion dominante pour un texte donné en utilisant le modèle chargé.
    
    Args:
        text (str): Le texte à analyser
        model: Le modèle de classification d'émotions
        tokenizer: Le tokenizer correspondant au modèle
        id2label (dict): Dictionnaire de correspondance ID -> label d'émotion
        threshold (float): Seuil de prédiction - toute émotion avec une probabilité supérieure sera considérée
    
    Returns:
        str: L'étiquette de l'émotion la plus probable qui dépasse le seuil, 
             ou "neutral" si aucune émotion ne dépasse le seuil
    """
    # Vérifier d'abord les patterns implicites de gratitude
    if detect_gratitude_patterns(text):
        # Si un pattern de gratitude est détecté, retourner une émotion positive
        # comme "gratitude" si disponible dans id2label, sinon "joy"
        for emotion_id, emotion in id2label.items():
            if emotion == "gratitude":
                return "gratitude"
        return "joy"  # Fallback si gratitude n'est pas disponible
    
    # Tokenization : utiliser les mêmes paramètres de padding et truncation que pendant l'entraînement
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Envoyer les entrées sur le même appareil que le modèle 
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Obtenir les logits et les convertir en probabilités
    logits = outputs.logits
    probabilities = torch.sigmoid(logits[0])  # Prendre le premier élément du batch

    # Trouver l'émotion avec la plus haute probabilité
    max_prob_idx = torch.argmax(probabilities).item()
    max_prob = probabilities[max_prob_idx].item()
    max_emotion = id2label[max_prob_idx]
    
    # Si la probabilité maximale dépasse le seuil, retourner cette émotion
    if max_prob > threshold:
        return max_emotion
    else:
        # Si aucune émotion ne dépasse le seuil, retourner "neutral"
        return "neutral"

if __name__ == "__main__":
    # Obtenir le chemin racine du projet pour les chemins absolus
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    
    # Chemin direct vers le modèle
    model_path = os.path.join(project_root, 'results/mvp_model')
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle n'existe pas à l'emplacement: {model_path}")
        print("Veuillez d'abord exécuter le script d'entraînement pour sauvegarder le modèle.")
        sys.exit(1)

    print(f"Chargement du modèle et du tokenizer depuis {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        print("Modèle et tokenizer chargés avec succès.")
        print(f"Seuil de prédiction utilisé: {PREDICTION_PROB_THRESHOLD}")
        print("-" * 30)

        print("Entrez un texte pour prédire l'émotion dominante. Tapez 'quitter' pour sortir.")
        while True:
            user_input = input("Votre texte: ")
            if user_input.lower() == 'quitter':
                break

            if user_input.strip() == "":
                print("Veuillez entrer un texte non vide.")
                continue

            # Prédire l'émotion dominante
            predicted_emotion = predict_emotion(user_input, model, tokenizer, ID2LABEL, PREDICTION_PROB_THRESHOLD)

            # Afficher le résultat
            if predicted_emotion != "neutral":
                print(f"Émotion prédite: {predicted_emotion}")
            else:
                print("Aucune émotion prédite avec confiance suffisante (résultat: neutral)")
            print("-" * 30)

    except Exception as e:
        print(f"Une erreur est survenue lors du chargement ou de la prédiction: {e}")
        sys.exit(1)

    print("Fin du programme de prédiction.")