import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from src.data.data_handler import load_and_preprocess_data
from models.BERT.BERTTrainer import BERTTrainer  # ta classe Trainer existante
from models.BERT.BERT import BERTForMultiLabelEmotion

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
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch]).float()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Charger et prétraiter les données
    train_dataset, val_dataset, test_dataset, tokenizer = load_and_preprocess_data()

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
    


if __name__ == "__main__":
    main()
