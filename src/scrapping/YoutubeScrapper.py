import requests
import re
from urllib.parse import urlparse, parse_qs
import json
import os

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
    'maxResults': 100,
    'textFormat': 'plainText',
    'key': API_KEY
}

while True:
    response = requests.get(url, params=params)
    data = response.json()

    for item in data['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        COMMENTS.append(comment)

    if 'nextPageToken' in data:
        params['pageToken'] = data['nextPageToken']
    else:
        break

#display the comments
print(f"Total comments: {len(COMMENTS)}")
for i, comment in enumerate(COMMENTS, start=1):
    print(f"{i}: {comment}")
