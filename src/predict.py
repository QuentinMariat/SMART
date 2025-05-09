# src/predict.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
import sys
import numpy as np
import csv
import argparse
from tqdm import tqdm

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

def read_comments_from_csv(csv_file_path):
    """
    Lit les commentaires à partir d'un fichier CSV.
    """
    comments = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # Skip header
            next(reader, None)
            # Read each row
            for row in reader:
                if row and len(row) > 0:
                    comments.append(row[0])
        return comments
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV: {e}")
        return []

def save_predictions_to_csv(csv_input_path, predictions, threshold):
    """
    Sauvegarde les prédictions dans un fichier CSV.
    """
    # Créer un nom de fichier pour les résultats basé sur le fichier d'entrée
    output_filename = os.path.splitext(os.path.basename(csv_input_path))[0] + f"_predictions_{threshold}.csv"
    output_path = os.path.join(os.path.dirname(csv_input_path), output_filename)
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # En-tête: Commentaire, Émotion1, Probabilité1, Émotion2, Probabilité2, ...
            writer.writerow(['Comment', 'Top_Emotion', 'Top_Probability', 'All_Emotions'])
            
            for comment, emotions in predictions:
                if emotions:
                    top_emotion, top_prob = emotions[0]
                    # Convertir les émotions en chaîne formatée
                    all_emotions = "; ".join([f"{emotion}: {prob:.4f}" for emotion, prob in emotions])
                    writer.writerow([comment, top_emotion, f"{top_prob:.4f}", all_emotions])
                else:
                    writer.writerow([comment, "Aucune", "0.0000", ""])
        
        print(f"Prédictions sauvegardées dans: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des prédictions: {e}")
        return None

def main():
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Prédit les émotions pour tous les commentaires d'un fichier CSV.")
    parser.add_argument("csv_file", help="Chemin vers le fichier CSV contenant les commentaires")
    parser.add_argument("--threshold", type=float, default=PREDICTION_PROB_THRESHOLD, 
                        help=f"Seuil de prédiction (défaut: {PREDICTION_PROB_THRESHOLD})")
    parser.add_argument("--interactive", action="store_true", 
                        help="Mode interactif pour entrer du texte manuellement après le traitement du fichier")
    
    args = parser.parse_args()
    
    # Vérifier si le fichier CSV existe
    if not os.path.isfile(args.csv_file):
        print(f"Erreur: Le fichier CSV '{args.csv_file}' n'existe pas.")
        sys.exit(1)
    
    model_path = f"{TRAINING_ARGS['output_dir']}/final_model"

    if not os.path.exists(model_path):
        print(f"Erreur: Le dossier du modèle '{model_path}' n'existe pas.")
        print("Veuillez d'abord exécuter le script d'entraînement pour sauvegarder le modèle.")
        sys.exit(1)

    print(f"Chargement du modèle et du tokenizer depuis {model_path}...")

    try:
        # Charger le modèle et le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Modèle et tokenizer chargés avec succès.")
        
        # Afficher des informations
        print(f"Fichier CSV d'entrée: {args.csv_file}")
        print(f"Seuil de prédiction utilisé: {args.threshold}")
        print("-" * 50)
        
        # Lire les commentaires
        comments = read_comments_from_csv(args.csv_file)
        if not comments:
            print("Aucun commentaire trouvé dans le fichier CSV.")
            sys.exit(1)
        
        print(f"Nombre de commentaires à analyser: {len(comments)}")
        print("Analyse des commentaires en cours...")
        
        # Prédire les émotions pour chaque commentaire
        predictions = []
        for comment in tqdm(comments, desc="Prédictions"):
            if comment.strip():  # Ignorer les commentaires vides
                emotions = predict_emotion(comment, model, tokenizer, ID2LABEL, args.threshold)
                predictions.append((comment, emotions))
        
        # Sauvegarder les prédictions
        output_file = save_predictions_to_csv(args.csv_file, predictions, args.threshold)
        
        # Statistics summary
        emotion_counts = {}
        no_emotion_count = 0
        
        for _, emotions in predictions:
            if emotions:
                top_emotion = emotions[0][0]
                emotion_counts[top_emotion] = emotion_counts.get(top_emotion, 0) + 1
            else:
                no_emotion_count += 1
        
        print(f"Analyse terminée. {len(predictions)} commentaires analysés.")
        print("-" * 50)
        
        # Calculer le total pour vérification
        total_counted = sum(emotion_counts.values()) + no_emotion_count
        
        if emotion_counts or no_emotion_count > 0:
            print("Répartition des émotions dominantes:")
            
            # Afficher d'abord les émotions détectées, triées par fréquence
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(predictions)) * 100
                print(f"- {emotion}: {count} ({percentage:.1f}%)")
            
            # Afficher les commentaires sans émotion détectée
            if no_emotion_count > 0:
                percentage = (no_emotion_count / len(predictions)) * 100
                print(f"- Aucune émotion détectée: {no_emotion_count} ({percentage:.1f}%)")
            
            # Vérifier que le total est bien 100%
            total_percentage = sum([(count / len(predictions)) * 100 for count in emotion_counts.values()]) + (no_emotion_count / len(predictions)) * 100
            print(f"\nTotal: {total_counted} commentaires ({total_percentage:.1f}%)")
        
        # Mode interactif optionnel
        if args.interactive:
            print("\nMode interactif activé. Entrez un texte pour prédire les émotions. Tapez 'quitter' pour sortir.")
            while True:
                user_input = input("Votre texte: ")
                if user_input.lower() == 'quitter':
                    break

                if user_input.strip() == "":
                    print("Veuillez entrer un texte non vide.")
                    continue

                predicted_emotions = predict_emotion(user_input, model, tokenizer, ID2LABEL, args.threshold)

                if predicted_emotions:
                    print("Émotions prédites (nom, probabilité):")
                    for emotion, prob in predicted_emotions:
                        print(f"- {emotion}: {prob:.4f}")
                else:
                    print("Aucune émotion prédite au seuil spécifié.")
                print("-" * 30)

    except Exception as e:
        print(f"Une erreur est survenue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Fin du programme de prédiction.")

if __name__ == "__main__":
    main()
