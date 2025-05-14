import os
import sys
import csv
import logging

logger = logging.getLogger(__name__)

def analyze_youtube_comments_with_model(comments_file):
    """
    Analyse les commentaires YouTube avec le modèle de prédiction d'émotions.
    
    Args:
        comments_file (str): Chemin vers le fichier CSV contenant les commentaires
        
    Returns:
        dict: Statistiques des émotions détectées
    """
    try:
        # Vérifier si le modèle de prédiction est disponible
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
        
        # Import predict_emotion depuis le module predict.py
        try:
            from src.mvp.predict import predict_emotion
            from src.config.settings import ID2LABEL, PREDICTION_PROB_THRESHOLD, TRAINING_ARGS
        except ImportError:
            logger.error("Impossible d'importer les modules nécessaires pour l'analyse des sentiments")
            return {"error": "Modules de prédiction non disponibles"}
        
        # Vérifier le chemin du modèle
        model_path = f"{TRAINING_ARGS['output_dir']}/final_model"
        
        if not os.path.exists(model_path):
            logger.warning(f"Le modèle n'existe pas à l'emplacement: {model_path}")
            return {"error": "Modèle non disponible"}
        
        # Charger le modèle et le tokenizer
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"Chargement du modèle depuis {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logger.info("Modèle chargé avec succès")
        
        # Lire les commentaires du fichier CSV
        comments = []
        if not os.path.exists(comments_file):
            logger.error(f"Fichier de commentaires non trouvé: {comments_file}")
            return {"error": f"Fichier non trouvé: {comments_file}"}
            
        with open(comments_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Ignorer l'en-tête
            for row in reader:
                if len(row) >= 2:
                    comments.append(row[1])
        
        # Limiter le nombre de commentaires à analyser (pour éviter une surcharge)
        max_comments = 100
        if len(comments) > max_comments:
            logger.info(f"Analyse limitée aux {max_comments} premiers commentaires")
            comments = comments[:max_comments]
        
        # Analyser chaque commentaire
        results = []
        emotion_counts = {}
        
        for comment in comments:
            emotions = predict_emotion(comment, model, tokenizer, ID2LABEL, PREDICTION_PROB_THRESHOLD)
            
            # Si des émotions sont détectées, comptabiliser la principale
            if emotions:
                top_emotion = emotions[0][0]  # Première émotion (la plus probable)
                if top_emotion in emotion_counts:
                    emotion_counts[top_emotion] += 1
                else:
                    emotion_counts[top_emotion] = 1
            
            results.append({
                "text": comment,
                "emotions": emotions
            })
        
        # Convertir les émotions en sentiments généraux (positif, négatif, neutre)
        positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", "love", "optimism", "pride", "relief"]
        negative_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "remorse", "sadness"]
        
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
        
        return {
            "total_comments": len(comments),
            "emotion_counts": emotion_counts,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "results": results
        }
        
    except ImportError as e:
        logger.error(f"Erreur d'importation lors de l'analyse: {e}")
        return {"error": f"Module non disponible: {str(e)}"}
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des commentaires: {e}")
        return {"error": str(e)}

def classify_text_sentiment(text):
    """
    Classifie un texte en sentiment positif, négatif ou neutre basé sur des mots-clés.
    Fonction de repli simple utilisée quand le modèle n'est pas disponible.
    
    Args:
        text (str): Texte à analyser
        
    Returns:
        str: "positive", "negative" ou "neutral"
    """
    text = text.lower()
    if any(word in text for word in ["love", "great", "amazing", "best", "good"]):
        return "positive"
    elif any(word in text for word in ["worst", "terrible", "disappointed", "waste", "bad"]):
        return "negative"
    else:
        return "neutral" 