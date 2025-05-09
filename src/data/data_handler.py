# src/data/data_handler.py

from datasets import load_dataset
import torch
from src.config.settings import MODEL_NAME, DATASET_NAME, NUM_LABELS

def load_and_preprocess_data(tokenizer=None, max_train_samples=None):
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    # 1) Sous-échantillonnage du split train si demandé
    if max_train_samples is not None:
        ds["train"] = ds["train"].shuffle(seed=42).select(
            list(range(min(max_train_samples, len(ds["train"]))))
        )
        print(f"→ Train subset: {len(ds['train'])} exemples retenus")

    # 2) Création des labels multi-hot
    def create_multi_hot_labels(examples):
        batch = []
        for label_list in examples["labels"]:
            vec = [0.0] * NUM_LABELS
            for idx in label_list:
                if 0 <= idx < NUM_LABELS:
                    vec[idx] = 1.0
            batch.append(vec)
        examples["labels"] = batch
        return examples

    print("Creating multi-hot labels...")
    ds = ds.map(create_multi_hot_labels, batched=True)

    # 3) Préparer le tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        print(f"Loading HF tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        print("Using custom BPE tokenizer")

    # 4) Tokenisation sur chaque split
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    print("Tokenizing train split...")
    train_tok = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenizing validation split...")
    val_tok   = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenizing test split...")
    test_tok  = ds["test"].map(tokenize_fn, batched=True, remove_columns=["text"])

    # 5) Caster labels en float
    def cast_labels(examples):
        examples["labels"] = [[float(v) for v in lab] for lab in examples["labels"]]
        return examples

    train_tok = train_tok.map(cast_labels, batched=True)
    val_tok   = val_tok.map(cast_labels, batched=True)
    test_tok  = test_tok.map(cast_labels, batched=True)

    # —> ICI : on cast explicitement le format de la colonne labels en float
    from datasets import ClassLabel, Sequence, Value, Features

    # redéfinis le schéma pour que 'labels' soit un Sequence de float32
    float_features = Features({
        **train_tok.features,                # toutes les autres colonnes
        "labels": Sequence(Value("float32")) # labels devient sequence of float32
    })

    train_tok = train_tok.cast(float_features)
    val_tok   = val_tok.cast(float_features)
    test_tok  = test_tok.cast(float_features)

    # 6) Format PyTorch
    train_tok.set_format("torch")
    val_tok.set_format("torch")
    test_tok.set_format("torch")

    print("Data loading and preprocessing complete.")
    return train_tok, val_tok, test_tok, tokenizer
