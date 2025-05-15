import os
import sys
import csv
import logging
import torch
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def analyze_youtube_comments_with_model(comments_file):
    """
    Analyse les commentaires YouTube avec le modèle de prédiction d'émotions
    et génère un nouveau CSV avec les labels prédits.
    
    Args:
        comments_file (str): Chemin vers le fichier CSV contenant les commentaires
        
    Returns:
        dict: Statistiques d'analyse et chemin vers le nouveau fichier CSV généré
    """
    try:
        # Obtenir le chemin racine du projet
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
        
        # Vérifier si le modèle de prédiction est disponible
        sys.path.append(project_root)
        
        # Import predict_emotion depuis le module predict.py
        try:
            from src.mvp.predict import predict_emotion
            from src.config.settings import ID2LABEL, PREDICTION_PROB_THRESHOLD, TRAINING_ARGS
        except ImportError:
            logger.error("Impossible d'importer les modules nécessaires pour l'analyse des sentiments")
            return {"error": "Modules de prédiction non disponibles"}
        
        # Chemin direct vers le modèle à la racine du projet
        model_path = os.path.join(project_root, 'results/mvp_model')
        
        if not os.path.exists(model_path):
            logger.warning(f"Le modèle n'existe pas à l'emplacement: {model_path}")
            return {"error": "Modèle non disponible"}
        
        # Charger le modèle et le tokenizer
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"Chargement du modèle depuis {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logger.info("Modèle chargé avec succès")
        
        # Vérifier l'existence du fichier CSV d'entrée
        if not os.path.exists(comments_file):
            logger.error(f"Fichier de commentaires non trouvé: {comments_file}")
            return {"error": f"Fichier non trouvé: {comments_file}"}
        
        # Lire le CSV d'entrée avec pandas
        logger.info(f"Lecture du fichier CSV: {comments_file}")
        try:
            df = pd.read_csv(comments_file)
            total_comments = len(df)
            logger.info(f"Total de {total_comments} commentaires lus depuis le CSV")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du CSV: {e}")
            return {"error": f"Erreur de lecture du CSV: {str(e)}"}
        
        # Vérifier que la colonne de texte existe
        text_column = None
        possible_text_columns = ['text', 'comment', 'content', 'message']
        
        for col in possible_text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            if len(df.columns) >= 2:  # Considérer la deuxième colonne comme le texte
                text_column = df.columns[1]
            else:
                logger.error("Aucune colonne de texte identifiable dans le CSV")
                return {"error": "Format de CSV non reconnu"}
        
        logger.info(f"Utilisation de la colonne '{text_column}' pour les commentaires")
        
        # Préparer le fichier de sortie
        output_dir = os.path.dirname(comments_file)
        base_filename = os.path.basename(comments_file).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{base_filename}_labeled_{timestamp}.csv")
        
        # Seuil pour la détection des émotions - réduit pour moins de neutres
        threshold = 0.15
        
        # Statistiques
        emotion_counts = {}
        processed_count = 0
        
        # Créer une nouvelle colonne pour les labels
        df['label'] = 'neutral'  # Valeur par défaut
        
        # Analyser chaque commentaire
        logger.info(f"Début de l'analyse des commentaires avec un seuil de {threshold}...")
        for idx, row in df.iterrows():
            try:
                comment_text = str(row[text_column])
                if pd.isna(comment_text) or comment_text.strip() == "":
                    continue  # Ignorer les commentaires vides
                
                # Prédire l'émotion pour ce commentaire avec le modèle
                emotion = predict_emotion(comment_text, model, tokenizer, ID2LABEL, threshold)
                
                # Enregistrer le label dans le DataFrame
                df.at[idx, 'label'] = emotion
                
                # Compter cette émotion
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
                
                processed_count += 1
                
                # Log d'avancement tous les 100 commentaires
                if processed_count % 100 == 0:
                    logger.info(f"Progression: {processed_count}/{total_comments} commentaires analysés")
                    
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse du commentaire {idx}: {e}")
        
        # Enregistrer le DataFrame avec les labels dans un nouveau CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Fichier CSV avec labels généré: {output_file}")
        
        # Convertir les émotions en sentiments généraux (positif, négatif, neutre)
        positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", "love", "optimism", "pride", "relief", "approval", "caring"]
        negative_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "remorse", "sadness", "confusion", "nervousness"]
        
        sentiment_counts = {
            "positive": sum(emotion_counts.get(emotion, 0) for emotion in positive_emotions),
            "negative": sum(emotion_counts.get(emotion, 0) for emotion in negative_emotions),
            "neutral": sum(emotion_counts.get(emotion, 0) for emotion in [e for e in emotion_counts if e not in positive_emotions and e not in negative_emotions])
        }
        
        total = sum(sentiment_counts.values())
        if total > 0:
            sentiment_percentages = {
                "positive": round(sentiment_counts["positive"] / total * 100, 2),
                "negative": round(sentiment_counts["negative"] / total * 100, 2),
                "neutral": round(sentiment_counts["neutral"] / total * 100, 2)
            }
        else:
            sentiment_percentages = {"positive": 0, "negative": 0, "neutral": 0}
        
        # Préparer les résultats normalisés pour l'API
        normalized_results = []
        skipped_comments = total_comments - processed_count
        
        for idx, row in df.iterrows():
            try:
                text = str(row[text_column])
                label = row['label']
                
                # Ajouter seulement si le texte est non vide
                if not pd.isna(text) and text.strip() != "":
                    normalized_results.append({
                        "id": int(idx),
                        "text": text,
                        "label": label,
                        "probability": 0.8  # Valeur par défaut car nous n'avons pas la probabilité exacte ici
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de la normalisation du commentaire {idx}: {e}")
        
        # Retourner les résultats
        return {
            "total_comments": total_comments,
            "processed_comments": processed_count,
            "skipped_comments": skipped_comments,
            "emotion_counts": emotion_counts,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "output_file": output_file,  # Chemin vers le fichier CSV généré
            "normalized_results": normalized_results  # Résultats au format normalisé pour l'API
        }
        
    except ImportError as e:
        logger.error(f"Erreur d'importation lors de l'analyse: {e}")
        return {"error": f"Module non disponible: {str(e)}"}
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des commentaires: {e}")
        return {"error": str(e)} 