import requests
from urllib.parse import urlparse, parse_qs
import json
import os
import csv
import logging
import sys
import re

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # only works in Python 3.8+
)

def get_video_id(url):
    # Handle youtu.be links
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    
    # Handle youtube.com/watch?v= links
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        if 'v' in query:
            return query['v'][0]
        # Handle embed URLs
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[-1].split('?')[0]

    return None

# Fonction pour nettoyer le texte des commentaires
def clean_comment_text(text):
    # Remplacer les retours à la ligne par des espaces
    text = re.sub(r'\n+', ' ', text)
    # Supprimer les caractères spéciaux qui pourraient causer des problèmes dans le CSV
    text = re.sub(r'[\r\t]', ' ', text)
    # Remplacer les séquences multiples d'espaces par un seul espace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


VIDEO_URL = 'https://www.youtube.com/watch?v=0uzCUZeBi6c'
API_URL = 'https://www.googleapis.com/youtube/v3/commentThreads'

# Récupérer l'API key depuis les variables d'environnement (si python-dotenv est installé)
try:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY")
except ImportError:
    API_KEY = os.environ.get("YOUTUBE_API_KEY")
    
if not API_KEY:
    logger.warning("YOUTUBE_API_KEY non trouvée dans les variables d'environnement")
    API_KEY = input("Veuillez entrer votre clé API YouTube: ")


def fetch_comments(video_url):

    VIDEO_ID = get_video_id(video_url)
    if not VIDEO_ID:
        logger.error("[YoutubeScrapper] Erreur : Impossible de récupérer l'ID vidéo à partir de l'URL fournie.")
        raise ValueError("L'URL fournie ne contient pas d'ID vidéo valide : " + video_url)
    COMMENTS = []

    
    params = {
        'part': 'snippet',
        'videoId': VIDEO_ID,
        'maxResults': 100,  # Maximum autorisé par YouTube API par page
        'textFormat': 'plainText',
        'key': API_KEY
    }

    pages_fetched = 0
    max_pages = 50  # Récupérer jusqu'à 50 pages = ~5000 commentaires
    
    logger.info(f"[YoutubeScrapper] Récupération de TOUS les commentaires disponibles pour la vidéo {VIDEO_ID}")
    
    while True:
        response = requests.get(API_URL, params=params)
        data = response.json()
        if 'items' not in data:
            logger.error("[YoutubeScrapper] Erreur lors de la récupération des commentaires :", json.dumps(data, indent=2))
            if response.status_code != 200:
                logger.error(f"[YoutubeScrapper] HTTP {response.status_code} - Erreur API : {response.text}")
                raise Exception(f"[fetch_comments] Erreur lors de l'appel à l'API YouTube: Status Code {response.status_code}, youtube ID : {VIDEO_ID},Response: {response.text[:200]}, erreur api : {response.text}")
            raise ValueError(f"L'API YouTube n'a pas retourné de commentaires. Peut-être un problème avec l'ID vidéo ou la clé API ?")
        
        # Ajouter tous les commentaires de cette page
        for item in data['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            # Nettoyer le texte du commentaire
            clean_text = clean_comment_text(comment_text)
            COMMENTS.append(clean_text)

        pages_fetched += 1
        logger.info(f"[YoutubeScrapper] Page {pages_fetched} récupérée, {len(COMMENTS)} commentaires au total")
        
        # Arrêter si on atteint la limite de pages ou s'il n'y a plus de pages
        if 'nextPageToken' in data and pages_fetched < max_pages:
            params['pageToken'] = data['nextPageToken']
        else:
            if 'nextPageToken' in data and pages_fetched >= max_pages:
                logger.info(f"[YoutubeScrapper] Limite de {max_pages} pages atteinte, arrêt de la récupération")
            else:
                logger.info("[YoutubeScrapper] Plus de pages à récupérer")
            break

    # Create a directory for the output if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Create the CSV file path using the video ID for uniqueness
    csv_file_path = os.path.join(output_dir, f"youtube_comments_{VIDEO_ID}.csv")
    
    # Write comments to CSV file in a format compatible with predict.py
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # Use QUOTE_ALL pour s'assurer que tout est bien échappé
        writer.writerow(['comment_id', 'text'])  # En-tête CSV
        
        for i, comment in enumerate(COMMENTS):
            writer.writerow([i, comment])

    logger.info(f"[YoutubeScrapper] Total comments saved: {len(COMMENTS)}")
    logger.info(f"[YoutubeScrapper] Comments saved to: {csv_file_path}")
    
    return csv_file_path

def getTopComments(url):
    VIDEO_ID = get_video_id(url)
    COMMENTS = []
    
    # Configuration initiale pour la pagination
    params = {
        "part": "snippet",
        "videoId": VIDEO_ID,
        "order": "relevance",  # Tri par pertinence pour obtenir les meilleurs commentaires
        "maxResults": 100,     # Maximum autorisé par l'API YouTube par page
        "key": API_KEY
    }
    
    pages_fetched = 0
    max_pages = 10  # Récupérer jusqu'à 10 pages pour les top comments
    
    logger.info(f"[YoutubeScrapper] Récupération des top commentaires pour la vidéo {VIDEO_ID}")
    
    # Boucle pour pagination
    while pages_fetched < max_pages:
        response = requests.get(API_URL, params=params)
        data = response.json()

        if 'items' not in data:
            logger.error("[YoutubeScrapper] Erreur API YouTube :", json.dumps(data, indent=2))
            if response.status_code != 200:
                logger.error(f"[YoutubeScrapper] HTTP {response.status_code} - Erreur API : {response.text}")
                raise Exception(f"[getTopComments] Erreur lors de l'appel à l'API YouTube: Status Code {response.status_code}, Response: {response.text[:200]}")
            raise ValueError("Pas de 'items' dans la réponse. Vérifie la vidéo, l'API key, ou ton quota." + data)

        # Traiter tous les commentaires de cette page
        for item in data["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment_text = clean_comment_text(snippet["textDisplay"])
            
            # Extraire la date et la formater pour affichage
            published_at = snippet["publishedAt"]
            try:
                # Convertir la date ISO en format heure:minute
                from datetime import datetime
                dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%H:%M")
            except:
                formatted_time = "00:00"  # Fallback en cas d'erreur
            
            COMMENTS.append({
                "text": comment_text,
                "likes": snippet["likeCount"],
                "time": formatted_time
            })
        
        pages_fetched += 1
        logger.info(f"[YoutubeScrapper] Page {pages_fetched} récupérée, {len(COMMENTS)} commentaires au total")
        
        # Continuer avec la page suivante si disponible
        if 'nextPageToken' in data:
            params['pageToken'] = data['nextPageToken']
        else:
            logger.info("[YoutubeScrapper] Plus de pages disponibles pour getTopComments")
            break
    
    # Trier tous les commentaires par nombre de likes (décroissant)
    top_comments = sorted(COMMENTS, key=lambda x: x["likes"], reverse=True)
    
    # Limiter à 100 commentaires les mieux notés pour l'affichage et les analyses
    return top_comments[:100]