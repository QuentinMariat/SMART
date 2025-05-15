import os
import sys
import csv
import logging
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Ajouter le chemin racine au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Ajout de {project_root} au PYTHONPATH")

# Import des modules nécessaires
try:
    from src.mvp.predict import predict_emotion
    from src.config.settings import ID2LABEL, PREDICTION_PROB_THRESHOLD, TRAINING_ARGS, EMOTION_LABELS
    logger.info("Modules importés avec succès")

    from src.models.bert.bert import BERTForMultiLabelEmotion
    logger.info("Modules importés avec succès")
except ImportError as e:
    logger.error(f"Erreur lors de l'importation des modules: {str(e)}")
    raise ImportError(f"Impossible d'importer les modules nécessaires: {str(e)}")

# Variables globales pour stocker le modèle et le tokenizer Stella
stella_model = None
stella_tokenizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_stella_model():
    """
    Initialise et charge le modèle Stella et son tokenizer.
    Cette fonction doit être appelée au démarrage du serveur.
    """
    global stella_model, stella_tokenizer
    
    try:
        model_path = os.path.join(project_root, 'results/stella_model/bert_multilabel.pt')
        if not os.path.exists(model_path):
            logger.error(f"Le modèle Stella n'existe pas à l'emplacement: {model_path}")
            return False

        # Charger le tokenizer
        stella_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        vocab_size = stella_tokenizer.vocab_size
        logger.info(f"Taille du vocabulaire: {vocab_size}")

        # Initialiser le modèle
        stella_model = BERTForMultiLabelEmotion(
            num_labels=len(EMOTION_LABELS),
            vocab_size=vocab_size,
            use_pretrained=True,
            pretrained_model_name='roberta-base'
        )
        
        # Charger les poids du modèle
        stella_model.load_state_dict(torch.load(model_path, map_location=device))
        stella_model.to(device)
        stella_model.eval()
        
        logger.info("✅ Modèle Stella chargé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation du modèle Stella: {str(e)}")
        stella_model = None
        stella_tokenizer = None
        return False

def predict_emotion_stella(text, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Prédit les émotions pour un texte donné en utilisant le modèle Stella.
    Utilise le modèle global pré-chargé.
    """
    global stella_model, stella_tokenizer
    
    if stella_model is None or stella_tokenizer is None:
        raise RuntimeError("Le modèle Stella n'est pas initialisé")

    # Tokenisation
    inputs = stella_tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = stella_model(input_ids, attention_mask=attention_mask)
        # Trouver l'indice de l'émotion avec la plus haute probabilité
        max_prob_idx = torch.argmax(outputs, dim=1).item()
            
    return max_prob_idx

def analyze_youtube_comments_with_model(comments_file, model_name="mvp"):
    """
    Analyse les commentaires YouTube avec le modèle de prédiction d'émotions
    et génère un nouveau CSV avec les labels prédits.
    
    Args:
        comments_file (str): Chemin vers le fichier CSV contenant les commentaires
        model_name (str): Nom du modèle à utiliser ("mvp" ou "stella")
        
    Returns:
        dict: Statistiques d'analyse et chemin vers le nouveau fichier CSV généré
    """
    try:
        # Pour le modèle Stella, vérifier qu'il est bien initialisé
        if model_name == "stella":
            if stella_model is None or stella_tokenizer is None:
                return {"error": "Le modèle Stella n'est pas initialisé"}
        else:
            # Charger le modèle MVP
            model_path = os.path.join(project_root, 'results/mvp_model')
            if not os.path.exists(model_path):
                logger.warning(f"Le modèle MVP n'existe pas à l'emplacement: {model_path}")
                return {"error": "Modèle MVP non disponible"}
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(device)
                model.eval()
                logger.info("Modèle MVP chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle MVP: {str(e)}")
                return {"error": f"Erreur de chargement du modèle MVP: {str(e)}"}

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
        logger.info(f"Modèle utilisé: {model_name}")
        for idx, row in df.iterrows():
            try:
                comment_text = str(row[text_column])
                if pd.isna(comment_text) or comment_text.strip() == "":
                    continue  # Ignorer les commentaires vides
                
                # Prédire l'émotion pour ce commentaire avec le modèle approprié
                if model_name == "stella":
                    pred_labels = predict_emotion_stella(comment_text)
                    emotion = EMOTION_LABELS[pred_labels]
                else:
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
                pass
        
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