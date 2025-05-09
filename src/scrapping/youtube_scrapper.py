import requests
import re
import sys
from urllib.parse import urlparse, parse_qs
import json
import os
import csv

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

def remove_emojis(text):
    """
    Supprime les emojis et autres caractères spéciaux d'un texte
    """
    # Fonction qui détermine si un caractère doit être conservé
    def is_valid_char(char):
        # Conserver uniquement les caractères ASCII imprimables, les espaces, 
        # et certains caractères de ponctuation courants
        return char.isascii() and (char.isprintable() or char.isspace())
    
    # Filtrer le texte pour ne garder que les caractères valides
    return ''.join(c for c in text if is_valid_char(c))

def clean_comment(text):
    """
    Nettoie le commentaire en supprimant les emojis et en remplaçant les sauts de ligne
    """
    # D'abord supprimer les emojis
    text = remove_emojis(text)
    
    # Supprimer complètement tous les types de sauts de ligne
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


# Utiliser l'URL fournie en argument ou une URL par défaut
if len(sys.argv) > 1:
    VIDEO_URL = sys.argv[1]
    print(f"URL de la vidéo: {VIDEO_URL}")
else:
    print(f"Aucune URL fournie")

# Load API key from JSON file

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
try:
    with open(config_path, "r") as file:
        config = json.load(file)
    API_KEY = config["youtube"]["apiKey"]
except FileNotFoundError:
    print(f"Config file not found at: {config_path}")
    API_KEY = input("Please enter your YouTube API Key: ")
except KeyError:
    print("YouTube API key not found in config file")
    API_KEY = input("Please enter your YouTube API Key: ")
VIDEO_ID = get_video_id(VIDEO_URL)
COMMENTS = []

url = 'https://www.googleapis.com/youtube/v3/commentThreads'
params = {
    'part': 'snippet',
    'videoId': VIDEO_ID,
    'maxResults': 10000,
    'textFormat': 'plainText',
    'key': API_KEY
}

print(f"Extracting comments for video ID: {VIDEO_ID}...")

while True:
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'error' in data:
        print(f"API Error: {data['error']['message']}")
        break
    
    if 'items' not in data:
        print("No comments found or video unavailable.")
        break

    for item in data['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        # Nettoyer le commentaire (supprimer emojis + gérer sauts de ligne)
        processed_comment = clean_comment(comment)
        if processed_comment:  # Ne pas ajouter les commentaires vides après nettoyage
            COMMENTS.append(processed_comment)

    if 'nextPageToken' in data:
        params['pageToken'] = data['nextPageToken']
    else:
        break

# Create a directory for the output if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

# Create the CSV file path using the video ID for uniqueness
csv_file_path = os.path.join(output_dir, f"youtube_comments_{VIDEO_ID}.csv")

# Write comments to CSV file - en utilisant les paramètres par défaut sans guillemets systématiques
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)  # N'utilise des guillemets que quand nécessaire
    writer.writerow(['Comment'])  # Un seul en-tête pour la colonne de commentaires
    
    # Écrire tous les commentaires sans numérotation
    for comment in COMMENTS:
        writer.writerow([comment])

print(f"Total comments extracted: {len(COMMENTS)}")
print(f"Comments saved to: {csv_file_path}")
