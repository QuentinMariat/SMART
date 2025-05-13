import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config.settings import EMOTION_THRESHOLDS

class BERTTrainer:
    def __init__(self, model, train_loader, val_loader, num_labels, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_labels = num_labels
        self.device = device
        self.criterion = nn.BCELoss()  # car le modèle renvoie des probas (sigmoid)
        self.thresholds = torch.tensor(list(EMOTION_THRESHOLDS.values()), device=self.device)

    def train(self, epochs=3, lr=2e-5, weight_decay=0.01, rdrop_alpha=5.0, file_name='bert_multilabel.pt'):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

        for epoch in range(epochs):
          print(f'Epoch {epoch + 1}/{epochs}')
          self.model.train()
          train_loss = 0.0

          pbar = tqdm(self.train_loader, desc="Training", leave=False)
          for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs1 = self.model(input_ids)
                outputs2 = self.model(input_ids)

                # Perte classique (cross entropy / BCE) sur les deux sorties
                loss1 = self.criterion(outputs1, labels)
                loss2 = self.criterion(outputs2, labels)

                # Perte de divergence KL entre les deux sorties
                kl_loss = self.kl_divergence(outputs1, outputs2)

                # Perte totale avec pondération de la régularisation R-Drop
                loss = (loss1 + loss2) / 2 + self.rdrop_alpha * kl_loss


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

          avg_train_loss = train_loss / len(self.train_loader)
          print(f'Train Loss: {avg_train_loss:.4f}')

          self.evaluate()

        torch.save(self.model.state_dict(), file_name)
        print(f"✅ Modèle sauvegardé sous '{file_name}'")
    
    def kl_divergence(self, p_logits, q_logits):
        p = F.log_softmax(p_logits, dim=-1)
        q = F.log_softmax(q_logits, dim=-1)

        p_softmax = p.exp()
        q_softmax = q.exp()

        kl_pq = F.kl_div(p, q_softmax, reduction='batchmean')
        kl_qp = F.kl_div(q, p_softmax, reduction='batchmean')
        return (kl_pq + kl_qp) / 2      

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        preds = []
        targets = []

        pbar = tqdm(self.val_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                pred = (outputs > self.thresholds).float().cpu().numpy()
                target = labels.cpu().numpy()

                preds.extend(pred)
                targets.extend(target)

        avg_val_loss = val_loss / len(self.val_loader)
        f1 = f1_score(targets, preds, average='micro')
        acc = accuracy_score(targets, preds)

        print(f'Val Loss: {avg_val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}')
    
    def predict(self, test_loader):
      self.model.eval()
      preds = []
      with torch.no_grad():
          for batch in tqdm(test_loader, desc="Predicting", leave=False):
              input_ids = batch['input_ids'].to(self.device)
              outputs = self.model(input_ids)
              predicted = (outputs > self.thresholds).int().cpu().numpy()
              preds.extend(predicted)
      return preds

    def evaluate_on_test(self, test_loader):
      self.model.eval()
      val_loss = 0.0
      preds = []
      targets = []

      with torch.no_grad():
          for batch in tqdm(test_loader, desc="Testing", leave=False):
              input_ids = batch['input_ids'].to(self.device)
              labels = batch['labels'].to(self.device)

              outputs = self.model(input_ids)
              loss = self.criterion(outputs, labels)
              val_loss += loss.item()

              pred = (outputs > self.thresholds).float().cpu().numpy()
              target = labels.cpu().numpy()

              preds.extend(pred)
              targets.extend(target)

      avg_loss = val_loss / len(test_loader)
      f1 = f1_score(targets, preds, average='micro')
      acc = accuracy_score(targets, preds)
      print(f'Test Loss: {avg_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}')

    def load_model(self, path):
      self.model.load_state_dict(torch.load(path, map_location=self.device))
      self.model.to(self.device)
      print(f"✅ Modèle chargé depuis '{path}'")

