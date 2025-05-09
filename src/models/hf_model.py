from transformers import AutoModelForSequenceClassification
from src.config.settings import MODEL_NAME, NUM_LABELS, ID2LABEL, LABEL2ID

def get_model():
    """
    charge le modèle pré-entraîné de Hugging Face configuré pour la classification multi-label.
    """
    print(f"Loading model: {MODEL_NAME}")
    # num_labels: Indique au modèle combien de neurones il doit y avoir dans la couche de sortie pour la classification.
    # on passe id2label et label2id pour que le Trainer puisse les sauvegarder avec le modèle.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    print("Model loaded.")
    return model
