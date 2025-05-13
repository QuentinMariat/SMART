from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, List
import random
import logging
from fastapi.responses import JSONResponse
from fastapi import Request
import sys


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
from scrapper.youtube import *
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


        # Get the video ID to find the correct CSV file
        video_id = get_video_id(url)
        csv_file_path = f"scrapping/output/youtube_comments_{video_id}.csv"

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

        # Dummy sentiment classifier based on keywords
        def classify_sentiment(text):
            # Bon là il fait la moyenne des 5 premiers commentaires en comptant les mots, donc là on remplacera par l'IA
            text = text.lower()
            if any(word in text for word in ["love", "great", "amazing", "best", "good"]):
                return "positive"
            elif any(word in text for word in ["worst", "terrible", "disappointed", "waste", "bad"]):
                return "negative"
            else:
                return "neutral"

        # Convert raw comments into Comment objects
        comment_objs = []
        for item in top_comments:
            sentiment = classify_sentiment(item.get("text", ""))
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
        fetch_comments(url)
        
        topComments = getTopComments(url)
        logger.debug(f"Top comments fetched for video ID: {video_id}")
        logger.info(f"Top comments: {topComments}")
        return {"status": "success", "message": "Scraping initiated, results will be saved to CSV", "file_path": "scrapping/output/youtube_comments_" + video_id +".csv", "comments": topComments}
    
    except ImportError as e:
        logger.error(f"Failed to import YouTube scraper: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error scraping YouTube comments: {str(e)}")
        raise