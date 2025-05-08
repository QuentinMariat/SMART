# src/data/data_handler.py
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch
from src.config.settings import MODEL_NAME, DATASET_NAME, LABEL2ID, ID2LABEL, NUM_LABELS # NUM_LABELS est importé ici

def load_and_preprocess_data():
    print(f"Loading dataset: {DATASET_NAME}")
    # Chargement du dataset depuis Hugging Face datasets
    dataset = load_dataset(DATASET_NAME)

    # Correction: Adaptez la fonction pour gérer les lots (batched=True)
    def create_multi_hot_labels(examples): # Notez le 'examples' au pluriel
        batch_multi_hot_labels = [] # Pour stocker les vecteurs multi-hot pour tout le lot

        # Itérer sur chaque *liste de labels* dans le lot
        for example_labels_list in examples['labels']: # example_labels_list est la liste d'indices pour un seul texte, ex: [1, 5, 2]
            one_hot_vector = [0.0] * NUM_LABELS # Initialiser le vecteur pour ce texte
            
            # Maintenant, itérer sur chaque index de label *dans cette liste*
            for label_index in example_labels_list: # label_index est maintenant un INT, ex: 1, puis 5, puis 2
                # AJOUTEZ LA LIGNE DE DEBUG ICI, juste avant la comparaison:
                print(f"DEBUG: Processing label_index={label_index}, type(NUM_LABELS)={type(NUM_LABELS)}, value(NUM_LABELS)={NUM_LABELS}")

                if 0 <= label_index < NUM_LABELS:
                    one_hot_vector[label_index] = 1.0
                else:
                     # Optionnel: ajouter un avertissement si un indice est hors limites (ne devrait pas arriver avec GoEmotions)
                     print(f"Warning: Label index {label_index} out of bounds [0, {NUM_LABELS-1}] for batch.")
                     
            batch_multi_hot_labels.append(one_hot_vector) # Ajouter le vecteur multi-hot de ce texte au lot

        # Assigner la liste de vecteurs multi-hot au nouveau champ pour le lot
        examples['labels_multi_hot'] = batch_multi_hot_labels
        return examples

    print("Creating multi-hot labels...")
    # Appeler avec batched=True
    dataset = dataset.map(create_multi_hot_labels, batched=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples): # Cette fonction est déjà correcte pour batched=True
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'labels', 'id'])

    # renommer la colonne 'labels_multi_hot' en 'labels' car le Trainer cherche par défaut une colonne nommée 'labels'
    tokenized_dataset = tokenized_dataset.rename_column("labels_multi_hot", "labels")

    tokenized_dataset.set_format("torch")

    print("Data loading and preprocessing complete.")
    return tokenized_dataset['train'], tokenized_dataset['validation'], tokenized_dataset['test'], tokenizer
