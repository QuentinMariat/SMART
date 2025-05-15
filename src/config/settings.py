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
                        'amusement': 0.40,
                        'anger': 0.10,
                        'annoyance': 0.15,
                        'approval': 0.25,
                        'caring': 0.30,
                        'confusion': 0.40,
                        'curiosity': 0.30,
                        'desire': 0.20,
                        'disappointment': 0.25,
                        'disapproval': 0.20,
                        'disgust': 0.15,
                        'embarrassment': 0.15,
                        'excitement': 0.20,
                        'fear': 0.35,
                        'gratitude': 0.45,
                        'grief': 0.10,
                        'joy': 0.35,
                        'love': 0.35,
                        'nervousness': 0.25,
                        'optimism': 0.15,
                        'pride': 0.15,
                        'realization': 0.05,
                        'relief': 0.02,
                        'remorse': 0.08,
                        'sadness': 0.35,
                        'surprise': 0.10,
                        'neutral': 0.3
        }

LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}
NUM_LABELS = len(EMOTION_LABELS)

BASE_MODEL_NAME = "distilbert-base-uncased"
# Adjusted training parameters for multi-label learning
TRAINING_ARGS = {
    "output_dir": "./results",
    "num_train_epochs": 15,  # Increased epochs
    "per_device_train_batch_size": 16,  # Reduced batch size for better generalization
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
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
    "fp16": True
}

# seuil de probabilité pour la classification multi-label : une prédiction est considérée positive pour une émotion si sa probabilité dépasse ce seuil.
PREDICTION_PROB_THRESHOLD = 0.5
