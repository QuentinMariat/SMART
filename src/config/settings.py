# modèle pré-entraîné de Hugging Face à utiliser
# Options:
# - microsoft/deberta-v3-base : Better performance than BERT
# - roberta-base : Better than BERT for emotion detection
# - xlm-roberta-base : For multilingual support
MODEL_NAME = "roberta-base"

DATASET_NAME = "go_emotions"

# Increased sequence length based on typical emotion text lengths
MAX_SEQ_LENGTH = 256

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

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
# Adjusted training parameters for multi-label learning
TRAINING_ARGS = {
    "output_dir": "./results",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "warmup_ratio": 0.1,  # 10% of training steps
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "fp16": True  # Enable mixed precision training
}

# seuil de probabilité pour la classification multi-label : une prédiction est considérée positive pour une émotion si sa probabilité dépasse ce seuil.
PREDICTION_PROB_THRESHOLD = 0.5
