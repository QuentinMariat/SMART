from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config.settings import DATASET_NAME, EMOTION_LABELS, NUM_LABELS

# Chargement du dataset GoEmotions
dataset = load_dataset(DATASET_NAME, split="train")

print(f"Dataset loaded: {DATASET_NAME}")
print(f"Number of samples: {len(dataset)}")

# Texte + multi-labels → vecteur multi-hot
def prepare_data(dataset):
    texts = []
    labels = []
    for example in dataset:
        texts.append(example['text'])
        label_vector = [0] * NUM_LABELS
        for label in example['labels']:
            label_vector[label] = 1
        labels.append(label_vector)
    return texts, np.array(labels)

# Préparation
texts, label_vectors = prepare_data(dataset)

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(texts, label_vectors, test_size=0.2, random_state=42)

# TF-IDF vectorisation
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Classifieur multi-label
base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
multi_label_rf = MultiOutputClassifier(base_rf)

# Entraînement
multi_label_rf.fit(X_train_tfidf, y_train)

# Prédiction
y_pred = multi_label_rf.predict(X_test_tfidf)

# Rapport par étiquette
print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS, zero_division=0))

# Matrice de confusion multi-label
mcm = multilabel_confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion pour chaque label
fig, axes = plt.subplots(7, 4, figsize=(15, 18))  # Organiser les 28 émotions en grille
axes = axes.ravel()

for i in range(NUM_LABELS):
    ax = axes[i]
    sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(EMOTION_LABELS[i])
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Vérités terrain')

plt.tight_layout()
plt.show()
