from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import random

app = FastAPI(title="Mock Sentiment API")

# === Common schema ===
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float
    explanation: str

# === /predict schema ===
class SentimentRequest(BaseModel):
    text: str

# === /analyze/twitter and /analyze/youtube schema ===
class URLRequest(BaseModel):
    url: str

# === Mock logic ===
def get_mock_sentiment(text: str) -> str:
    lower_text = text.lower()
    positive_words = ['love', 'happy', 'great', 'awesome', 'excellent', 'fantastic', 'perfect']
    negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'disappointed', 'rude']

    positive_count = sum(word in lower_text for word in positive_words)
    negative_count = sum(word in lower_text for word in negative_words)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def get_mock_explanation(sentiment: str) -> str:
    if sentiment == "positive":
        return "The text contains predominantly positive language expressing satisfaction, happiness, or approval."
    elif sentiment == "negative":
        return "The text contains predominantly negative language expressing dissatisfaction, anger, or disappointment."
    else:
        return "The text appears to be neutral, containing mostly factual information without strong emotional language."

def predict_sentiment(text: str) -> tuple[str, float, str]:
    sentiment = get_mock_sentiment(text)
    confidence = round(random.uniform(0.5, 1.0), 4)
    explanation = get_mock_explanation(sentiment)
    return sentiment, confidence, explanation

# === Real text analysis endpoint ===
@app.post("/predict", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    sentiment, confidence, explanation = predict_sentiment(request.text)
    return SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
        explanation=explanation
    )

# === Twitter URL mock endpoint ===
@app.post("/analyze/twitter", response_model=SentimentResponse)
async def analyze_twitter(request: URLRequest):
    # Later: fetch tweet content from URL
    # For now: random sentiment
    sentiment = random.choice(["positive", "neutral", "negative"])
    confidence = round(random.uniform(0.5, 1.0), 4)
    explanation = get_mock_explanation(sentiment)
    return SentimentResponse(sentiment=sentiment, confidence=confidence, explanation=explanation)

# === YouTube URL mock endpoint ===
@app.post("/analyze/youtube", response_model=SentimentResponse)
async def analyze_youtube(request: URLRequest):
    # Later: fetch video title/description/comments
    # For now: random sentiment
    sentiment = random.choice(["positive", "neutral", "negative"])
    confidence = round(random.uniform(0.5, 1.0), 4)
    explanation = get_mock_explanation(sentiment)
    return SentimentResponse(sentiment=sentiment, confidence=confidence, explanation=explanation)
