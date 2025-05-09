import requests
import json
import os
# Display a warning about Twitter API rate limits
print("WARNING: The Twitter API only allows 100 comments per day with the standard API access.")

# ✅ CONFIGURATION
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
try:
    with open(config_path, "r") as file:
        config = json.load(file)
    bearer_token = config["twitter"]["bearerToken"]
except FileNotFoundError:
    print(f"Config file not found at: {config_path}")
    bearer_token = input("Please enter your YouTube API Key: ")
except KeyError:
    print("YouTube API key not found in config file")
    bearer_token = input("Please enter your YouTube API Key: ")


tweet_id = "1919053040734072844"  # L’ID du tweet original

url = "https://api.twitter.com/2/tweets/search/recent"
headers = {
    "Authorization": f"Bearer {bearer_token}"
}

params = {
    "query": f"conversation_id:{tweet_id}",
    "tweet.fields": "author_id,created_at,in_reply_to_user_id",
    "max_results": 100,
}

def fetch_comments():
    comments = []
    next_token = None

    while True:
        if next_token:
            params["next_token"] = next_token
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        print(data)

        if "data" in data:
            comments.extend(data["data"])

        next_token = data.get("meta", {}).get("next_token")
        if not next_token:
            break

    return comments

# ✅ RÉSULTAT
comments = fetch_comments()
for i, comment in enumerate(comments, 1):
    print(f"{i}. @{comment['author_id']} - {comment['created_at']}\n   {comment['text']}\n")
