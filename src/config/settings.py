# modèle pré-entraîné de Hugging Face à utiliser
# distilbert-base-uncased : plus petit et plus rapide
# bert-base-uncased : plus grand mais plus performant
MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "bert-base-uncased"

DATASET_NAME = "go_emotions"

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

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
