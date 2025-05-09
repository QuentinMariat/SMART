import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch

def compute_metrics(eval_pred):
    """
    Calcule les métriques pour la classification multi-label.
    Args:
        eval_pred: Tuple contenant les logits du modèle et les labels réels.
                   Format attendu par le Hugging Face Trainer.
                   (logits, labels)
    Returns:
        Dictionnaire des métriques calculées.
    """
    logits, labels = eval_pred

    # Convertir les numpy arrays (passés par le Trainer) en tenseurs PyTorch
    # Les labels sont déjà en float (0.0, 1.0) à cause du multi-hot encoding et set_format("torch")
    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels) # Ceci sera un FloatTensor

    # Appliquer la sigmoïde pour obtenir les probabilités
    predictions_tensor = torch.sigmoid(logits_tensor)

    # Convertir les probabilités en prédictions binaires en utilisant un seuil (par exemple 0.5)
    # et convertir en LongTensor (type entier pour 0 ou 1)
    threshold = 0.5
    y_pred_tensor = (predictions_tensor > threshold).long()

    # Convertir les labels réels (FloatTensor 0.0 ou 1.0) en LongTensor (0 ou 1)
    # C'est souvent cette étape qui résout l'erreur 'Float can't be cast to Long'
    # lorsque le Trainer manipule les labels pour les métriques.
    y_true_tensor = labels_tensor.long()

    # Convertir les tenseurs binaires (LongTensor) en numpy arrays pour sklearn
    y_pred = y_pred_tensor.numpy()
    y_true = y_true_tensor.numpy()

    # Calcul des métriques.
    # sklearn metrics functions (like f1_score) expect numpy arrays of integers (0 or 1)
    # with shape (n_samples, n_classes) for multi-label.
    # Nos y_true et y_pred sont maintenant dans ce format après les conversions.
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

    # Le Trainer attend un dictionnaire de métriques
    return {
        "f1": f1_macro,
    }

# Vous pouvez ajouter d'autres fonctions de métriques ici si nécessaire.
