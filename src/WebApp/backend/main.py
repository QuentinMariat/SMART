from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, List
import random
import logging



# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment API")
from fastapi.middleware.cors import CORSMiddleware

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

# === Response for detailed mock analysis ===
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

def generate_mock_analysis() -> DetailedAnalysisResponse:
    logger.debug("Starting mock analysis generation")
    total_comments = random.randint(100, 600)
    positive_ratio = random.uniform(0.3, 0.6)
    negative_ratio = random.uniform(0.1, 0.3)
    neutral_ratio = max(0.0, 1 - positive_ratio - negative_ratio)
    
    logger.debug(f"Generated ratios - positive: {positive_ratio}, negative: {negative_ratio}, neutral: {neutral_ratio}")

    def make_comments(pool, sentiment, count, like_max):
        logger.debug(f"Making {count} {sentiment} comments with max likes of {like_max}")
        return [
            Comment(
                text=random.choice(pool),
                sentiment=sentiment,
                likes=random.randint(0, like_max),
                time=f"{random.randint(0,23)}:{str(random.randint(0,59)).zfill(2)}"  # Fixed time format
            )
            for _ in range(count)
        ]

    try:
        positive_count = int(positive_ratio * 10)
        negative_count = int(negative_ratio * 10)
        neutral_count = int(neutral_ratio * 10)
        
        logger.debug(f"Creating comments - positive: {positive_count}, negative: {negative_count}, neutral: {neutral_count}")
        
        comments = (
            make_comments(positive_comments, "positive", positive_count, 1000)
            + make_comments(negative_comments, "negative", negative_count, 500)
            + make_comments(neutral_comments, "neutral", neutral_count, 200)
        )
        random.shuffle(comments)
        
        logger.debug(f"Total comments created: {len(comments)}")
        
        response = DetailedAnalysisResponse(
            totalComments=total_comments,
            positive=round(positive_ratio, 2),
            negative=round(negative_ratio, 2),
            neutral=round(neutral_ratio, 2),
            comments=comments
        )
        logger.debug("Successfully generated mock analysis")
        return response
    except Exception as e:
        logger.error(f"Error generating mock analysis: {str(e)}", exc_info=True)
        raise

# === Endpoint for Twitter analysis ===
@app.post("/analyze/twitter", response_model=DetailedAnalysisResponse)
async def analyze_twitter(request: URLRequest):
    logger.info(f"Twitter analysis requested for URL: {request.url}")
    try:
        result = generate_mock_analysis()
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
        result = generate_mock_analysis()
        logger.info("YouTube analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during YouTube analysis: {str(e)}", exc_info=True)
        raise
