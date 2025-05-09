import requests
import re
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

print(f"Total new comments added: {len(COMMENTS)}")
print(f"Comments saved to: {csv_file_path}")
