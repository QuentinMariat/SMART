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


# Vérifier les arguments de ligne de commande
if len(sys.argv) < 2:
    print("Erreur: URL de vidéo YouTube requise")
    print("Usage: python youtube_scrapper.py <URL_YOUTUBE> [CLE_API]")
    sys.exit(1)

# Premier argument est l'URL de la vidéo
VIDEO_URL = sys.argv[1]
print(f"URL de la vidéo: {VIDEO_URL}")

# Deuxième argument (optionnel) est la clé API
if len(sys.argv) > 2:
    API_KEY = sys.argv[2]
    print(f"Clé API fournie en argument: {API_KEY[:4]}...{API_KEY[-4:]}")
else:
    # Si pas de clé API en argument, essayer de charger depuis le fichier config
    print("Pas de clé API fournie en argument, recherche dans le fichier config...")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        API_KEY = config["youtube"]["apiKey"]
        print(f"Clé API trouvée dans le fichier config: {API_KEY[:4]}...{API_KEY[-4:]}")
    except FileNotFoundError:
        print(f"Fichier de configuration non trouvé: {config_path}")
        API_KEY = input("Veuillez entrer votre clé API YouTube: ")
    except KeyError:
        print("Clé API YouTube introuvable dans le fichier de configuration")
        API_KEY = input("Veuillez entrer votre clé API YouTube: ")
    except json.JSONDecodeError:
        print("Erreur de format dans le fichier config.json")
        API_KEY = input("Veuillez entrer votre clé API YouTube: ")

# Extraire l'ID de la vidéo
VIDEO_ID = get_video_id(VIDEO_URL)
if not VIDEO_ID:
    print(f"Erreur: Impossible d'extraire l'ID de la vidéo depuis l'URL: {VIDEO_URL}")
    sys.exit(1)

COMMENTS = []

url = 'https://www.googleapis.com/youtube/v3/commentThreads'
params = {
    'part': 'snippet',
    'videoId': VIDEO_ID,
    'maxResults': 10000,
    'textFormat': 'plainText',
    'key': API_KEY
}

print(f"Extraction des commentaires pour la vidéo ID: {VIDEO_ID}...")

while True:
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'error' in data:
        print(f"Erreur API: {data['error']['message']}")
        break
    
    if 'items' not in data:
        print("Aucun commentaire trouvé ou vidéo inaccessible.")
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
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
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

print(f"Total des commentaires extraits: {len(COMMENTS)}")
print(f"Commentaires sauvegardés dans: {csv_file_path}")
