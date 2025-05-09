from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config.settings import DATASET_NAME


# Charger le dataset GoEmotions
dataset = load_dataset(DATASET_NAME)
data = dataset["train"]

print(f"Dataset loaded: {DATASET_NAME}")
print(f"Number of samples: {len(data)}")

# Extraire le label principal (le premier dans la liste)
def get_primary_label(example):
    return example["labels"][0] if len(example["labels"]) > 0 else -1

data = data.map(lambda x: {"primary_label": get_primary_label(x)})
data = data.filter(lambda x: x["primary_label"] != -1)

# Extraire les textes et les labels
texts = data["text"]
labels = data["primary_label"]

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=data.features["primary_label"].names, yticklabels=data.features["primary_label"].names)
plt.xlabel('Prédictions')
plt.ylabel('Vérités terrain')
plt.title('Matrice de confusion')
plt.show()
