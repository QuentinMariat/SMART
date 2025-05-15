import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from src.data.data_handler import load_and_preprocess_data
from models.bert.bert_trainer import BERTTrainer  # ta classe Trainer existante
from models.bert.bert import BERTForMultiLabelEmotion
from transformers import AutoTokenizer
from src.config.settings import MODEL_NAME, EMOTION_LABELS
from src.tokenizer.bpe_tokenizer import BPETokenizer
from torch.nn.utils.rnn import pad_sequence

bpe = BPETokenizer()
bpe.load("data/tokenizer_files/tokenizer.json")

# Simple modèle pour classification multilabel
class SimpleClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained('distilbert-base-uncased')
        self.classifier = torch.nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits

def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_masks = [torch.tensor(b["attention_mask"]) for b in batch]
    labels = [torch.tensor(b["labels"], dtype=torch.float) for b in batch]

    # Padding dynamique
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(labels)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "labels": labels_tensor,
    }

def train_model(device, fast_dev=False):
    # 1. Charger et prétraiter les données
    if fast_dev:
        train_dataset, val_dataset, test_dataset, tokenizer = load_and_preprocess_data(tokenizer=bpe,max_train_samples=5000, max_val_samples=5000, max_test_samples=5000)
    else:
        train_dataset, val_dataset, test_dataset, tokenizer = load_and_preprocess_data(tokenizer=bpe)

    # 2. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

    # 3. Modèle
    num_labels = train_dataset[0]['labels'].shape[0]  # inférer le nombre de labels
    vocab_size = tokenizer.vocab_size
    model = BERTForMultiLabelEmotion(vocab_size=vocab_size, num_labels=num_labels)

    # 4. Trainer
    trainer = BERTTrainer(model, train_loader, val_loader, num_labels, device=device)

    # 5. Entraînement
    trainer.train(epochs=3, lr=2e-5, weight_decay=0.01)

    # 8. Test (évaluation sur les données de test)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)
    predictions = trainer.predict(test_loader)

    # Exemple d'affichage pour les 5 premières prédictions
    print("\nExemples de prédictions :")
    for i, pred in enumerate(predictions[:5]):
        print(f"Sample {i}: Predicted labels = {pred}")

    trainer.evaluate_on_test(test_loader)

def predict_single_comment(comment, model, tokenizer, device, threshold=0.5):
    model.eval()
    model.to(device)

    # Tokenisation
    inputs = tokenizer(
        comment,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids)
        probs = torch.sigmoid(logits).squeeze()

    # Binariser avec un seuil
    predicted_labels = (probs >= threshold).long().cpu().numpy()

    return predicted_labels, probs.cpu().numpy()


def create_comments_dataloader(comments, tokenizer, device):
    """
    Crée un DataLoader pour un nombre variable de commentaires.
    
    Args:
        comments (list of str): Liste des commentaires à évaluer.
        tokenizer (AutoTokenizer): Le tokenizer utilisé pour transformer le texte.
        device (str): Le périphérique ('cpu' ou 'cuda').

    Returns:
        DataLoader: Un DataLoader contenant les commentaires tokenisés.
    """
    # Tokenisation des commentaires
    inputs = tokenizer(
        comments,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Préparer les données pour le DataLoader
    dataset = [{
        'input_ids': inputs['input_ids'][i],
        'attention_mask': inputs['attention_mask'][i],
        'labels': torch.zeros(len(EMOTION_LABELS))  # Placeholder pour les labels
    } for i in range(len(comments))]

    # Créer un DataLoader
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    return dataloader

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\033[93mUtilisation de l'appareil : {device}\033[0m")
    #TEMP :
    train_dataset, val_dataset, test_dataset, tokenizer = load_and_preprocess_data(tokenizer=bpe,max_train_samples=5000, max_val_samples=5000, max_test_samples=5000)

    # 2. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

    # 3. Modèle
    num_labels = train_dataset[0]['labels'].shape[0]  # inférer le nombre de labels
    vocab_size = tokenizer.vocab_size
    model = BERTForMultiLabelEmotion(vocab_size=vocab_size, num_labels=num_labels)

    # 4. Trainer
    trainer = BERTTrainer(model, train_loader, val_loader, num_labels, device=device)



    while True:
        print("1. Entraîner le modèle")
        print("2. Evaluer un commentaire")
        print("3. Quitter")
        choice = input("Choisissez une option : ")

        if choice == '1':
            print("Vitesse entrainement : (1) lent, (2) rapide")
            speed_choice = input("Choisissez une vitesse : ")
            if speed_choice == '1':
                print("Entraînement lent...")
                train_model(device, fast_dev=False)
            elif speed_choice == '2':
                print("Entraînement rapide...")
                train_model(device, fast_dev=True)
        elif choice == '2':
            print("Entrez les commentaires à évaluer (séparés par un point-virgule) :")
            comments_input = input("Commentaires : ")
            comments = comments_input.split(";")
            print("Choisir un modèle :")
            print("1. BERT")
            print("2. LocalBERT")
            model_choice = input("Choisissez un modèle : ")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) #TODO : charger le tokenizer approprié
            
            if model_choice == '1':
                print("Évaluation avec BERT...")
                model = torch.load('bert.pt', map_location=device)  # Remplacez par le chemin de votre modèle
            
            elif model_choice == '2':
                print("Évaluation avec LocalBERT...")
                #model = BERTForMultiLabelEmotion(vocab_size=tokenizer.vocab_size, num_labels=len(EMOTION_LABELS))  # 6 = nombre de labels à adapter
                try:
                    trainer.load_model('bert_multilabel.pt')  # Remplacez par le chemin de votre modèle
                except FileNotFoundError:
                    print("❌ Fichier 'bert_multilabel.pt' introuvable. Veuillez d'abord entraîner le modèle.")
                    continue

                

                
            else:
                print("Choix de modèle invalide.")
                continue

            dataloader = create_comments_dataloader(comments, tokenizer, device)

            predictions = trainer.predict(dataloader)

            for i, pred_labels in enumerate(predictions):
                predicted_emotions = [EMOTION_LABELS[j] for j, val in enumerate(pred_labels) if val == 1]
                print(f"\nCommentaire {i + 1} : {comments[i]}")
                print(f"Labels binaires prédits : {pred_labels}")
                print(f"Émotions détectées : {predicted_emotions}\n")

        elif choice == '3':
            print("Au revoir !")
            break
        else:
            print("Choix invalide, veuillez réessayer.")
    


if __name__ == "__main__":
    main()


