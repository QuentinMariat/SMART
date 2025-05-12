# src/train_model.py

import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import Trainer, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.config.settings import TRAINING_ARGS
from src.data.data_handler import load_and_preprocess_data
from src.models.hf_model import get_model
from src.evaluation.metrics import compute_metrics
from src.tokenizer.bpe_tokenizer import BPETokenizer

# --- Wrapper pour exposer l'API HF au BPE maison ---
class WrappedBPETokenizer(PreTrainedTokenizerBase):
    def __init__(self, bpe_tokenizer: BPETokenizer):
        super().__init__()
        self.tokenizer = bpe_tokenizer
        self._pad_token = "<pad>"
        self._pad_token_id = len(self.tokenizer.vocab)

    @property
    def pad_token(self): return self._pad_token
    @pad_token.setter
    def pad_token(self, value): self._pad_token = value
    @property
    def pad_token_id(self): return self._pad_token_id
    @pad_token_id.setter
    def pad_token_id(self, value): self._pad_token_id = value

    def __call__(self, texts, padding=True, truncation=True, max_length=512, **kwargs):
        if isinstance(texts, str): texts = [texts]
        input_ids = [self.tokenizer.encode(t)[:max_length] for t in texts]
        if padding:
            max_len = max(len(ids) for ids in input_ids)
            input_ids = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids)

    def save_pretrained(self, save_directory):
        """
        HuggingFace Trainer calls this to save tokenizer files.
        Here, we forward to our BPE tokenizer save.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        # Save our tokenizer JSON
        path = os.path.join(save_directory, "tokenizer.json")
        self.tokenizer.save(path)
        

def train_model():
    print("Starting training process…")

    # 1) Charger ton BPE et wrapper
    bpe = BPETokenizer(vocab_size=5000)
    bpe.load("data/tokenizer.json")
    tokenizer = WrappedBPETokenizer(bpe)

    # 2) Charger les datasets (train, val, test) avec TON tokenizer
    train_ds, val_ds, test_ds, _ = load_and_preprocess_data(tokenizer=tokenizer, max_train_samples=100)

    # 3) Charger le modèle
    model = get_model()

    # 4) Préparer les TrainingArguments
    args = TrainingArguments(
        output_dir=TRAINING_ARGS["output_dir"],
        num_train_epochs=TRAINING_ARGS["num_train_epochs"],
        per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
        warmup_steps=TRAINING_ARGS["warmup_steps"],
        weight_decay=TRAINING_ARGS["weight_decay"],
        logging_dir=TRAINING_ARGS["logging_dir"],
        logging_steps=TRAINING_ARGS["logging_steps"],
        evaluation_strategy=TRAINING_ARGS["evaluation_strategy"],
        save_strategy=TRAINING_ARGS["save_strategy"],
        load_best_model_at_end=TRAINING_ARGS["load_best_model_at_end"],
        metric_for_best_model=TRAINING_ARGS["metric_for_best_model"],
        greater_is_better=TRAINING_ARGS["greater_is_better"],
        disable_tqdm=False,
        report_to=[]
    )

    # 5) Initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6) Lancer l'entraînement
    print("Training started...")
    trainer.train()
    print("Training finished.")

    # 7) Sauvegarder le meilleur modèle
    final_path = Path(TRAINING_ARGS["output_dir"]) / "final_model"
    print(f"Saving final model to {final_path}")
    trainer.save_model(final_path)
    print("Final model saved.")

if __name__ == "__main__":
    train_model()
