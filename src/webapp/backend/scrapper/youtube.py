import requests
from urllib.parse import urlparse, parse_qs
import json
import os
import csv
import logging
import sys

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


VIDEO_URL = 'https://www.youtube.com/watch?v=0uzCUZeBi6c'
API_URL = 'https://www.googleapis.com/youtube/v3/commentThreads'
# Load API key from JSON file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
try:
    with open(config_path, "r") as file:
        config = json.load(file)
    API_KEY = config["youtube"]["apiKey"]
except FileNotFoundError:
    logger.warning(f"Config file not found at: {config_path}")
    API_KEY = input("Please enter your YouTube API Key: ")
except KeyError:
    logger.warning("YouTube API key not found in config file")
    API_KEY = input("Please enter your YouTube API Key: ")


def fetch_comments(video_url):

    VIDEO_ID = get_video_id(video_url)
    if not VIDEO_ID:
        logger.error("[YoutubeScrapper ❌] Erreur : Impossible de récupérer l'ID vidéo à partir de l'URL fournie.")
        raise ValueError("L'URL fournie ne contient pas d'ID vidéo valide : " + video_url)
    COMMENTS = []

    
    params = {
        'part': 'snippet',
        'videoId': VIDEO_ID,
        'maxResults': 100,
        'textFormat': 'plainText',
        'key': API_KEY
    }

    while True:
        response = requests.get(API_URL, params=params)
        data = response.json()
        if 'items' not in data:
            logger.error("[YoutubeScrapper ❌] Erreur lors de la récupération des commentaires :", json.dumps(data, indent=2))
            if response.status_code != 200:
                logger.error(f"[YoutubeScrapper ❌] HTTP {response.status_code} - Erreur API : {response.text}")
                raise Exception(f"[fetch_comments] Erreur lors de l'appel à l'API YouTube: Status Code {response.status_code}, youtube ID : {VIDEO_ID},Response: {response.text[:200]}, erreur api : {response.text}")
            raise ValueError(f"L'API YouTube n'a pas retourné de commentaires. Peut-être un problème avec l'ID vidéo ou la clé API ?")
        
        for item in data['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            COMMENTS.append(comment)

        if 'nextPageToken' in data:
            params['pageToken'] = data['nextPageToken']
        else:
            break

    #display the comments
    # Create a directory for the output if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Create the CSV file path using the video ID for uniqueness
    csv_file_path = os.path.join(output_dir, f"youtube_comments_{VIDEO_ID}.csv")

    # Write comments to CSV file
    # Check if file exists
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header only if file is being created
        if not file_exists:
            writer.writerow(['Comment Number', 'Comment Text'])  # Header row
        
        # Get the current comment count if file exists
        if file_exists:
            with open(csv_file_path, 'r', encoding='utf-8') as read_file:
                comment_count = sum(1 for _ in read_file) - 1  # Subtract 1 for header
        else:
            comment_count = 0
        
        # Add new comments with continuing numbering
        for i, comment in enumerate(COMMENTS, start=comment_count+1):
            writer.writerow([i, comment])

    logger.info(f"[YoutubeScrapper] Total new comments added: {len(COMMENTS)}")
    logger.info(f"[YoutubeScrapper] Comments saved to: {csv_file_path}")

#fetch_comments(VIDEO_URL)

def getTopComments(url):
    VIDEO_ID = get_video_id(url)
    COMMENTS = []
    params = {
        "part": "snippet",
        "videoId": VIDEO_ID,
        "order": "relevance",  # ou "time"
        "maxResults": 100,
        "key": API_KEY
        }
    response = requests.get(API_URL, params=params)
    data = response.json()

    if 'items' not in data:
        logger.error("[YoutubeScrapper ❌] Erreur API YouTube :", json.dumps(data, indent=2))
        if response.status_code != 200:
            logger.error(f"[YoutubeScrapper ❌] HTTP {response.status_code} - Erreur API : {response.text}")
            raise Exception(f"[getTopComments] Erreur lors de l'appel à l'API YouTube: Status Code {response.status_code}, Response: {response.text[:200]}")
        raise ValueError("Pas de 'items' dans la réponse. Vérifie la vidéo, l’API key, ou ton quota." + data)

    for item in data["items"]:
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        COMMENTS.append({
            "text": snippet["textDisplay"],
            "likeCount": snippet["likeCount"],
            "publishedAt": snippet["publishedAt"]
        })
    
    top_comments = sorted(COMMENTS, key=lambda x: x["likeCount"], reverse=True)[:5]

    for i, comment in enumerate(top_comments, 1):
        print("[YoutubeScrapper]")
        print(f"\nCommentaire #{i}")
        print(f"Likes : {comment['likeCount']}")
        print(f"Date : {comment['publishedAt']}")
        print(f"Texte : {comment['text']}")

    return top_comments

#getTopComments(VIDEO_URL)