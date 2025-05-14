from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, List
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
from sentiment_analyzer import analyze_youtube_comments_with_model, classify_text_sentiment
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

# === Comment schema ===
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

# === Mock comments ===
positive_comments = [
    "This is amazing! Love the content.",
    "Great video, very informative!",
    "Best explanation I've seen on this topic.",
    "Keep up the good work!",
    "I've watched this 3 times already. So good!"
]

negative_comments = [
    "This is terrible, completely wrong information.",
    "Waste of time, didn't learn anything.",
    "The worst video on this topic.",
    "Disappointed with the quality.",
    "Not what I expected at all."
]

neutral_comments = [
    "Interesting perspective.",
    "I'll need to research this more.",
    "The video quality is good.",
    "Not sure I agree with everything.",
    "Decent explanation overall."
]

async def generate_analysis(url) -> DetailedAnalysisResponse:
    logger.debug("Starting analysis generation using real comments")

    try:
        # Get top comments using YouTube scraper
        result = await getCommentsFromYoutube(url)
        top_comments = result.get("comments", [])
        csv_file_path = result.get("file_path", "")

        # Get the video ID to find the correct CSV file
        video_id = get_video_id(url)
        
        if not csv_file_path or not os.path.exists(csv_file_path):
            logger.warning(f"Le fichier CSV n'a pas été retourné ou n'existe pas, utilisation du chemin par défaut")
            csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scrapper", "output", f"youtube_comments_{video_id}.csv")
        
        nb_top_comments = len(top_comments)
        total_comments = len(top_comments)
        # Check if the file exists and count the rows
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                # Subtract 1 to account for header row (if present)
                row_count = sum(1 for row in reader) - 1
                if row_count < 0:  # In case there's no header
                    row_count = 0
                
                logger.debug(f"Found {row_count} comments in CSV file {csv_file_path}")
                
                # Update total_comments if we have more comments in the CSV than what was fetched
                if row_count > len(top_comments):
                    logger.debug(f"Using CSV comment count ({row_count}) instead of fetched count ({len(top_comments)})")
                    total_comments = row_count
        else:
            logger.warning(f"CSV file not found: {csv_file_path}")

        logger.debug(f"Fetched {total_comments} comments")

        # Utiliser le modèle d'analyse de sentiment pour les commentaires
        try:
            sentiment_analysis = analyze_youtube_comments_with_model(csv_file_path)
            if "error" in sentiment_analysis:
                logger.warning(f"Erreur lors de l'analyse des sentiments: {sentiment_analysis['error']}")
                raise ValueError(sentiment_analysis["error"])
        except Exception as e:
            logger.warning(f"Exception lors de l'analyse des sentiments, utilisation du fallback: {str(e)}")
            sentiment_analysis = {"error": str(e)}
        
        if "error" in sentiment_analysis:
            logger.warning(f"Utilisation du classificateur de fallback basé sur des mots-clés")
            # Fallback vers le classificateur basé sur des mots-clés si le modèle échoue
                    
            # Convert raw comments into Comment objects avec le fallback
            comment_objs = []
            for item in top_comments:
                sentiment = classify_text_sentiment(item.get("text", ""))
                comment_objs.append(
                    Comment(
                        text=item.get("text", ""),
                        sentiment=sentiment,
                        likes=item.get("likes", 0),
                        time=item.get("time", "00:00")
                    )
                )
                
            # Count sentiments
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for c in comment_objs:
                sentiment_counts[c.sentiment] += 1
                
            positive_ratio = sentiment_counts["positive"] / nb_top_comments if nb_top_comments else 0
            negative_ratio = sentiment_counts["negative"] / nb_top_comments if nb_top_comments else 0
            neutral_ratio = sentiment_counts["neutral"] / nb_top_comments if nb_top_comments else 0
            
            logger.info(f"Analyse de sentiment fallback: positive={positive_ratio:.2f}, negative={negative_ratio:.2f}, neutral={neutral_ratio:.2f}")
        else:
            # Utiliser les résultats de l'analyse du modèle
            logger.info(f"Analyse des sentiments réussie: {sentiment_analysis['sentiment_percentages']}")
            
            # Convertir les pourcentages en ratios (0-1)
            positive_ratio = sentiment_analysis['sentiment_percentages']['positive'] / 100
            negative_ratio = sentiment_analysis['sentiment_percentages']['negative'] / 100
            neutral_ratio = sentiment_analysis['sentiment_percentages']['neutral'] / 100
            
            # Créer les objets de commentaires avec les sentiments prédits
            comment_objs = []
            results = sentiment_analysis.get('results', [])
            
            # Mapper les résultats avec les top_comments pour conserver les likes et timestamps
            comment_map = {item.get("text", ""): item for item in top_comments}
            
            for result in results[:30]:  # Limiter à 30 commentaires pour l'affichage (augmenté de 10)
                text = result["text"]
                emotions = result["emotions"]
                
                # Déterminer le sentiment général à partir des émotions
                sentiment = "neutral"
                if emotions:
                    top_emotion = emotions[0][0]  # Émotion la plus probable
                    # Classifier les émotions en sentiments
                    positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", "love", "optimism", "pride", "relief"]
                    negative_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "remorse", "sadness"]
                    
                    if top_emotion in positive_emotions:
                        sentiment = "positive"
                    elif top_emotion in negative_emotions:
                        sentiment = "negative"
                
                # Récupérer les likes et timestamps si disponibles dans top_comments
                item_info = comment_map.get(text, {})
                
                comment_objs.append(
                    Comment(
                        text=text,
                        sentiment=sentiment,
                        likes=item_info.get("likes", 0),
                        time=item_info.get("time", "00:00")
                    )
                )
            
            # Si nous n'avons pas assez de commentaires du modèle, ajouter des commentaires des top_comments
            if len(comment_objs) < 30 and top_comments:
                logger.info(f"Ajouter {30 - len(comment_objs)} commentaires supplémentaires des top_comments")
                
                # Fonction pour convertir les top_comments en objets Comment
                def comment_to_obj(item):
                    # Déterminer le sentiment à partir du texte (fallback)
                    text = item.get("text", "")
                    sentiment = classify_text_sentiment(text)
                        
                    return Comment(
                        text=text,
                        sentiment=sentiment,
                        likes=item.get("likes", 0),
                        time=item.get("time", "00:00")
                    )
                
                # Convertir jusqu'à 30 - len(comment_objs) commentaires des top_comments
                for item in top_comments[:30 - len(comment_objs)]:
                    # Vérifier si ce commentaire n'est pas déjà dans comment_objs
                    text = item.get("text", "")
                    if not any(c.text == text for c in comment_objs):
                        comment_objs.append(comment_to_obj(item))

        response = DetailedAnalysisResponse(
            totalComments=total_comments,
            positive=round(positive_ratio, 2),
            negative=round(negative_ratio, 2),
            neutral=round(neutral_ratio, 2),
            comments=comment_objs
        )
        logger.debug("Successfully generated analysis from real comments")
        return response

    except Exception as e:
        logger.error(f"Error generating analysis from YouTube data: {str(e)}", exc_info=True)
        raise

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
        result = generate_analysis(request.url)
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


async def getCommentsFromYoutube(url):

    try:
        logger.debug(f"Attempting to scrape YouTube comments for URL: {url}")
        
        # Get video ID and trigger comment scraping
        video_id = get_video_id(url)
        csv_file_path = fetch_comments(url)
        
        topComments = getTopComments(url)
        logger.debug(f"Top comments fetched for video ID: {video_id}")
        logger.info(f"Top comments: {topComments}")
        return {"status": "success", "message": "Scraping initiated, results will be saved to CSV", "file_path": csv_file_path, "comments": topComments}
    
    except ImportError as e:
        logger.error(f"Failed to import YouTube scraper: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error scraping YouTube comments: {str(e)}")
        raise