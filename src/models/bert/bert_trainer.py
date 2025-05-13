import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


from src.config.settings import EMOTION_THRESHOLDS

class BERTTrainer:
    def __init__(self, model, train_loader, val_loader, num_labels=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_labels = num_labels
        self.device = device
        self.criterion = nn.BCELoss()  # car le modèle renvoie des probas (sigmoid)
        self.thresholds = torch.tensor(list(EMOTION_THRESHOLDS.values()), device=self.device)

    def train(self, epochs=3, lr=2e-5, weight_decay=0.01, file_name='bert_multilabel.pt'):
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

              outputs = self.model(input_ids)  # segment_info=None implicite
              loss = self.criterion(outputs, labels)

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
    
    def pretrain(self, tokenizer, epochs=3, lr=2e-5, weight_decay=0.01, file_name='bert_pretrain.pt'):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(self.train_loader, desc="Training MLM", leave=False)
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch.get("token_type_ids")

                inputs_masked, mlm_labels = self._mask_tokens(input_ids.clone(), tokenizer)
                inputs_masked = inputs_masked.to(self.device)
                mlm_labels = mlm_labels.to(self.device)

                logits_mlm = self.model(inputs_masked, token_type_ids)
                loss = criterion(
                    logits_mlm.view(-1, self.model.vocab_size),
                    mlm_labels.view(-1)
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(self.train_loader)
            print(f'📊 Train MLM Loss: {avg_train_loss:.4f}')

            # Evaluation à la fin de chaque epoch
            self.evaluate_pretrain(tokenizer)

        # Sauvegarde finale
        torch.save(self.model.state_dict(), file_name)
        print(f"✅ Modèle pré-entraîné sauvegardé sous '{file_name}'")

        # Évaluation finale
        print("🔚 Évaluation finale après pré-entrainement :")
        self.evaluate_pretrain(tokenizer)


    def evaluate_pretrain(self, tokenizer):
        self.model.eval()
        val_loss = 0.0
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating MLM", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch.get("token_type_ids")

                inputs_masked, mlm_labels = self._mask_tokens(input_ids.clone(), tokenizer)
                inputs_masked = inputs_masked.to(self.device)
                mlm_labels = mlm_labels.to(self.device)

                logits_mlm = self.model(inputs_masked, token_type_ids)
                loss = criterion(
                    logits_mlm.view(-1, self.model.vocab_size),
                    mlm_labels.view(-1)
                )
                val_loss += loss.item()

                # Compute predictions
                preds = torch.argmax(logits_mlm, dim=-1)

                # Flatten and filter out positions with label -100
                active_labels = mlm_labels.view(-1) != -100
                filtered_preds = preds.view(-1)[active_labels]
                filtered_labels = mlm_labels.view(-1)[active_labels]

                all_preds.extend(filtered_preds.cpu().tolist())
                all_labels.extend(filtered_labels.cpu().tolist())

        avg_val_loss = val_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")

        print(f"🔎 Val MLM Loss: {avg_val_loss:.4f}")
        print(f"📊 Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")



    def evaluate_pretrain_on_test(self, test_loader, tokenizer):
        self.model.eval()
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Testing MLM", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch.get("token_type_ids")

                inputs_masked, mlm_labels = self._mask_tokens(input_ids.clone(), tokenizer)
                inputs_masked = inputs_masked.to(self.device)
                mlm_labels = mlm_labels.to(self.device)

                logits_mlm = self.model(inputs_masked, token_type_ids)
                loss = criterion(
                    logits_mlm.view(-1, self.model.vocab_size),
                    mlm_labels.view(-1)
                )
                test_loss += loss.item()

                # Compute predictions
                preds = torch.argmax(logits_mlm, dim=-1)

                # Flatten and filter out positions with label -100
                active_labels = mlm_labels.view(-1) != -100
                filtered_preds = preds.view(-1)[active_labels]
                filtered_labels = mlm_labels.view(-1)[active_labels]

                all_preds.extend(filtered_preds.cpu().tolist())
                all_labels.extend(filtered_labels.cpu().tolist())

        avg_test_loss = test_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)


        print("\n📝 Exemples de reconstruction MLM :")
        for i in range(min(5, input_ids.size(0))):
            original_tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
            masked_tokens = tokenizer.convert_ids_to_tokens(inputs_masked[i].tolist())
            predicted_tokens = tokenizer.convert_ids_to_tokens(torch.argmax(logits_mlm[i], dim=-1).tolist())

            print(f"\n🔹 Sample {i+1}:")
            print("Original : ", " ".join(original_tokens))
            print("Masqué   : ", " ".join(masked_tokens))
            print("Prédit   : ", " ".join(predicted_tokens))


        print(f"🧪 Test MLM Loss: {avg_test_loss:.4f}")
        print(f"📊 Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    def predict_pretrain(self, test_loader, tokenizer):
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting MLM", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch.get("token_type_ids")

                inputs_masked, mlm_labels = self._mask_tokens(input_ids.clone(), tokenizer)
                inputs_masked = inputs_masked.to(self.device)
                mlm_labels = mlm_labels.to(self.device)

                logits_mlm = self.model(inputs_masked, token_type_ids)
                preds = torch.argmax(logits_mlm, dim=-1)

                predictions.extend(preds.cpu().tolist())
                targets.extend(mlm_labels.cpu().tolist())

        return predictions, targets  # Pour inspection ou analyse plus poussée



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


    def _mask_tokens(self, inputs, tokenizer, mlm_probability=0.15):
        """
        Prépare les entrées masquées et les labels pour MLM.
        - 15% des tokens sont candidats au masquage.
        - 80% remplacés par [MASK], 10% par un token aléatoire, 10% inchangés.
        """
        device = self.device
        labels = inputs.clone()
        # Matrice de probabilité
        prob_matrix = torch.full(labels.shape, mlm_probability, device=device)
        # Ne pas masquer les tokens spéciaux
        special_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_mask = torch.tensor(special_mask, dtype=torch.bool, device=device)
        prob_matrix.masked_fill_(special_mask, 0.0)

        # Sélection des positions à masquer
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100  # ignore_index pour CrossEntropyLoss

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% -> random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
        inputs[indices_random] = random_words[indices_random]

        # 10% -> inchangés (les autres masked_indices restants)
        return inputs, labels

    

