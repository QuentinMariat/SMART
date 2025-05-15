from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, List, Optional
import random
import logging
from fastapi.responses import JSONResponse
from fastapi import Request
import sys
import os


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # only works in Python 3.8+
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment API")
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from scrapper.youtube import get_video_id, fetch_comments, getTopComments
from sentiment_analyzer import analyze_youtube_comments_with_model
import csv
from datetime import datetime

# Add this right after creating the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === URL input schema ===
class URLRequest(BaseModel):
    url: str

# === Normalized Comment schema ===
class NormalizedComment(BaseModel):
    id: int
    text: str
    label: str
    probability: float

# === Comment schema for frontend ===
class Comment(BaseModel):
    text: str
    sentiment: Literal["positive", "neutral", "negative"]
    likes: int
    time: str  # e.g., "12:34"

# === Response for detailed analysis ===
class DetailedAnalysisResponse(BaseModel):
    totalComments: int
    positive: float
    negative: float
    neutral: float
    comments: List[Comment]
    emotion_counts: Optional[dict] = None  # Ajout d'un champ optional pour les émotions

async def generate_analysis(url) -> DetailedAnalysisResponse:
    """
    Génère une analyse de sentiment à partir des commentaires YouTube.
    """
    logger.debug("Démarrage de l'analyse des commentaires YouTube")

    try:
        # Récupérer les commentaires YouTube
        result = await getCommentsFromYoutube(url)
        top_comments = result.get("comments", [])
        csv_file_path = result.get("file_path", "")

        # Obtenir l'ID de la vidéo
        video_id = get_video_id(url)
        
        if not csv_file_path or not os.path.exists(csv_file_path):
            logger.warning(f"Le fichier CSV n'a pas été retourné ou n'existe pas, utilisation du chemin par défaut")
            csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scrapper", "output", f"youtube_comments_{video_id}.csv")
        
        # Vérifier l'existence du fichier CSV et compter les commentaires
        total_comments = len(top_comments)
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                row_count = sum(1 for row in reader) - 1  # Soustraire l'en-tête
                if row_count > 0:
                    total_comments = row_count
                    logger.debug(f"Nombre total de commentaires dans le CSV: {total_comments}")
        else:
            logger.warning(f"Fichier CSV non trouvé: {csv_file_path}")

        # Analyser les commentaires avec le modèle
        try:
            analysis_results = analyze_youtube_comments_with_model(csv_file_path)
            
            if "error" in analysis_results:
                logger.error(f"Erreur d'analyse: {analysis_results['error']}")
                raise ValueError(analysis_results["error"])
                
        except Exception as e:
            logger.error(f"Exception lors de l'analyse: {str(e)}")
            return DetailedAnalysisResponse(
                totalComments=total_comments,
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                comments=[Comment(
                    text=f"Erreur d'analyse: {str(e)}",
                    sentiment="neutral",
                    likes=0,
                    time="00:00"
                )]
            )
        
        # Vérifier que nous avons bien des résultats normalisés
        if "normalized_results" not in analysis_results or not analysis_results["normalized_results"]:
            logger.warning("Aucun commentaire n'a pu être analysé")
            return DetailedAnalysisResponse(
                totalComments=total_comments,
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                comments=[Comment(
                    text="Aucun commentaire n'a pu être analysé avec succès.",
                    sentiment="neutral",
                    likes=0,
                    time="00:00"
                )]
            )
        
        # Récupérer les résultats normalisés
        normalized_results = analysis_results["normalized_results"]
        logger.info(f"Commentaires analysés avec succès: {len(normalized_results)}")
        
        # Log pour vérifier les décomptes d'émotions
        if "emotion_counts" in analysis_results:
            emotion_counts = analysis_results["emotion_counts"]
            logger.info(f"Emotions détectées: {emotion_counts}")
            
            # Vérifier la présence de gratitude
            if "gratitude" in emotion_counts:
                logger.info(f"✅ Gratitude détectée avec {emotion_counts['gratitude']} occurrences")
            else:
                logger.warning("⚠️ Aucune gratitude détectée dans les émotions!")
        else:
            logger.warning("⚠️ Aucun décompte d'émotions (emotion_counts) dans les résultats d'analyse!")
        
        # Récupérer les statistiques de sentiment
        positive_ratio = analysis_results['sentiment_percentages']['positive'] / 100
        negative_ratio = analysis_results['sentiment_percentages']['negative'] / 100
        neutral_ratio = analysis_results['sentiment_percentages']['neutral'] / 100
        
        # Créer un dictionnaire pour mapper les commentaires avec leurs métadonnées (likes, time)
        comment_metadata = {item.get("text", ""): item for item in top_comments}
        
        # Convertir les résultats normalisés en objets Comment pour le frontend
        comments_for_frontend = []
        
        # Définir les catégories d'émotions
        positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", "love", "optimism", "pride", "relief", "approval", "caring"]
        negative_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "remorse", "sadness", "confusion", "nervousness"]
        
        # Limite à 30 commentaires pour l'affichage
        for result in normalized_results[:30]:
            # Déterminer le sentiment général à partir de l'émotion
            label = result["label"]
            sentiment = "neutral"
            
            if label in positive_emotions:
                sentiment = "positive"
            elif label in negative_emotions:
                sentiment = "negative"
            
            # Récupérer les métadonnées si disponibles
            text = result["text"]
            metadata = comment_metadata.get(text, {})
            
            comments_for_frontend.append(Comment(
                text=text,
                sentiment=sentiment,
                likes=metadata.get("likes", 0),
                time=metadata.get("time", "00:00")
            ))
        
        response = DetailedAnalysisResponse(
            totalComments=total_comments,
            positive=round(positive_ratio, 2),
            negative=round(negative_ratio, 2),
            neutral=round(neutral_ratio, 2),
            comments=comments_for_frontend,
            emotion_counts=analysis_results.get("emotion_counts", {})
        )
        
        logger.debug("Analyse complétée avec succès")
        return response

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}", exc_info=True)
        
        # Retourner une réponse d'erreur structurée
        return DetailedAnalysisResponse(
            totalComments=0,
            positive=0.0,
            negative=0.0,
            neutral=1.0,
            comments=[Comment(
                text=f"Erreur: {str(e)}",
                sentiment="neutral",
                likes=0,
                time="00:00"
            )]
        )

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    import traceback
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "traceback": traceback.format_exc()
        }
    )

# === Endpoint for Twitter analysis ===
@app.post("/analyze/twitter", response_model=DetailedAnalysisResponse)
async def analyze_twitter(request: URLRequest):
    logger.info(f"Twitter analysis requested for URL: {request.url}")
    try:
        result = await generate_analysis(request.url)
        logger.info("Twitter analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during Twitter analysis: {str(e)}", exc_info=True)
        raise

# === Endpoint for YouTube analysis ===
@app.post("/analyze/youtube", response_model=DetailedAnalysisResponse)
async def analyze_youtube(request: URLRequest):
    logger.info(f"YouTube analysis requested for URL: {request.url}")
    try:
        result = await generate_analysis(request.url)
        logger.info("YouTube analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during YouTube analysis: {str(e)}", exc_info=True)
        raise

# === Définition du modèle pour les commentaires normalisés (pour une API future) ===
class NormalizedCommentsResponse(BaseModel):
    total_comments: int
    processed_comments: int
    skipped_comments: int
    emotion_counts: dict
    normalized_results: List[NormalizedComment]

# === Endpoint pour obtenir les commentaires au format normalisé ===
@app.post("/analyze/youtube/normalized", response_model=NormalizedCommentsResponse)
async def analyze_youtube_normalized(request: URLRequest):
    """
    Analyse les commentaires YouTube et retourne les résultats au format normalisé
    (id, texte, label, probabilité) sans les convertir pour le frontend.
    Cette API est utile pour les cas d'utilisation avancés ou l'intégration avec d'autres systèmes.
    """
    logger.info(f"Analyse normalisée demandée pour URL: {request.url}")
    try:
        # Récupérer les commentaires YouTube
        result = await getCommentsFromYoutube(url=request.url)
        csv_file_path = result.get("file_path", "")
        
        # Analyser les commentaires
        analysis_results = analyze_youtube_comments_with_model(csv_file_path)
        
        if "error" in analysis_results:
            raise ValueError(analysis_results["error"])
        
        # Créer la réponse normalisée
        normalized_response = NormalizedCommentsResponse(
            total_comments=analysis_results["total_comments"],
            processed_comments=analysis_results["processed_comments"],
            skipped_comments=analysis_results["skipped_comments"] if "skipped_comments" in analysis_results else 0,
            emotion_counts=analysis_results["emotion_counts"],
            normalized_results=[
                NormalizedComment(
                    id=item["id"],
                    text=item["text"],
                    label=item["label"],
                    probability=item["probability"]
                ) for item in analysis_results["normalized_results"]
            ]
        )
        
        logger.info("Analyse normalisée complétée avec succès")
        return normalized_response
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse normalisée: {str(e)}", exc_info=True)
        raise

async def getCommentsFromYoutube(url):
    """
    Récupère les commentaires d'une vidéo YouTube.
    """
    try:
        logger.debug(f"Tentative de récupération des commentaires YouTube pour URL: {url}")
        
        # Obtenir l'ID de la vidéo et déclencher la récupération des commentaires
        video_id = get_video_id(url)
        csv_file_path = fetch_comments(url)
        
        topComments = getTopComments(url)
        logger.debug(f"Commentaires récupérés pour la vidéo ID: {video_id}")
        
        return {
            "status": "success", 
            "message": "Récupération des commentaires réussie", 
            "file_path": csv_file_path, 
            "comments": topComments
        }
    
    except ImportError as e:
        logger.error(f"Échec d'importation du scraper YouTube: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des commentaires YouTube: {str(e)}")
        raise