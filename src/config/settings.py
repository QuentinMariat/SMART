# modèle pré-entraîné de Hugging Face à utiliser
# distilbert-base-uncased : plus petit et plus rapide
# bert-base-uncased : plus grand mais plus performant
MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "bert-base-uncased"


DATASET_NAME = "data/raw/combined_emotions.csv"

EMOTION_LABELS = [
    'joy','sad','anger','fear','love','surprise'
]

"""
DATASET_NAME = "sentiment_emotions"

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
    'neutral'
]
"""

EMOTION_THRESHOLDS = {
                        'admiration': 0.25,
                        'amusement': 0.45,
                        'anger': 0.15,
                        'annoyance': 0.10,
                        'approval': 0.30,
                        'caring': 0.40,
                        'confusion': 0.55,
                        'curiosity': 0.25,
                        'desire': 0.25,
                        'disappointment': 0.40,
                        'disapproval': 0.30,
                        'disgust': 0.20,
                        'embarrassment': 0.10,
                        'excitement': 0.35,
                        'fear': 0.40,
                        'gratitude': 0.45,
                        'grief': 0.05,
                        'joy': 0.40,
                        'love': 0.25,
                        'nervousness': 0.25,
                        'optimism': 0.20,
                        'pride': 0.10,
                        'realization': 0.15,
                        'relief': 0.05,
                        'remorse': 0.10,
                        'sadness': 0.40,
                        'surprise': 0.15,
                        'neutral': 0.25 
        }

LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}
NUM_LABELS = len(EMOTION_LABELS)

BASE_MODEL_NAME = "distilbert-base-uncased"
# hyperparamètres à ajuster pour l'entraînement
TRAINING_ARGS = {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True
}

# seuil de probabilité pour la classification multi-label : une prédiction est considérée positive pour une émotion si sa probabilité dépasse ce seuil.
PREDICTION_PROB_THRESHOLD = 0.5
