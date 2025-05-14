import pandas as pd
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import torch
from src.config.settings import MODEL_NAME, DATASET_NAME, NUM_LABELS, LABEL2ID
from torch.utils.data import DataLoader, WeightedRandomSampler


def load_and_preprocess_data(
    tokenizer=None,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None
):
    """
    Charge un dataset Hugging Face, sous-échantillonne, crée labels multi-hot,
    tokenize et formate pour PyTorch.
    """
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    # 1) Sous-échantillonnage
    if max_train_samples is not None:
        ds["train"] = ds["train"].shuffle(seed=42).select(
            list(range(min(max_train_samples, len(ds["train"])))))
        print(f"→ Train subset: {len(ds['train'])} exemples retenus")
    if max_val_samples is not None:
        ds["validation"] = ds["validation"].shuffle(seed=42).select(
            list(range(min(max_val_samples, len(ds["validation"])))))
        print(f"→ Validation subset: {len(ds['validation'])} exemples retenus")
    if max_test_samples is not None:
        ds["test"] = ds["test"].shuffle(seed=42).select(
            list(range(min(max_test_samples, len(ds["test"])))))
        print(f"→ Test subset: {len(ds['test'])} exemples retenus")

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
        print(f"Loading HF tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        print("Using custom BPE tokenizer")

    # 4) Tokenisation
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    print("Tokenizing train split...")
    train_tok = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenizing validation split...")
    val_tok = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenizing test split...")
    test_tok = ds["test"].map(tokenize_fn, batched=True, remove_columns=["text"])

    # 5) Caster labels en float
    def cast_labels(examples):
        examples["labels"] = [[float(v) for v in lab] for lab in examples["labels"]]
        return examples

    train_tok = train_tok.map(cast_labels, batched=True)
    val_tok   = val_tok.map(cast_labels, batched=True)
    test_tok  = test_tok.map(cast_labels, batched=True)

    # 6) Redéfinir schéma pour float32
    float_features = Features({**train_tok.features, "labels": Sequence(Value("float32"))})
    train_tok = train_tok.cast(float_features)
    val_tok   = val_tok.cast(float_features)
    test_tok  = test_tok.cast(float_features)

    # 7) Format PyTorch
    train_tok.set_format("torch")
    val_tok.set_format("torch")
    test_tok.set_format("torch")

    print("Data loading and preprocessing complete.")
    return train_tok, val_tok, test_tok, tokenizer


def load_and_preprocess_csv(
    csv_path: str,
    tokenizer=None,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_seed: int = 42,
    batch_size: int = 16
):
    """
    Charge un CSV (columns: sentence, emotion), split en train/val/test,
    encode en multi-hot, tokenize et crée DataLoaders avec WeightedRandomSampler.

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # 1) Lecture et split
    df = pd.read_csv(csv_path)
    from sklearn.model_selection import train_test_split
    df_train, df_temp = train_test_split(df, train_size=train_size, random_state=random_seed, shuffle=True)
    rel_val = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(df_temp, train_size=rel_val, random_state=random_seed, shuffle=True)

    # 2) Sous-échantillonnage
    def subsample(df_split, max_s, name):
        if max_s is not None and len(df_split) > max_s:
            df_split = df_split.sample(n=max_s, random_state=random_seed)
            print(f"→ {name} subset: {len(df_split)} exemples retenus")
        return df_split
    df_train = subsample(df_train, max_train_samples, "Train")
    df_val   = subsample(df_val,   max_val_samples,   "Validation")
    df_test  = subsample(df_test,  max_test_samples,  "Test")

    # 3) Conversion en Dataset HF
    ds_train = Dataset.from_pandas(df_train.reset_index(drop=True))
    ds_val   = Dataset.from_pandas(df_val.reset_index(drop=True))
    ds_test  = Dataset.from_pandas(df_test.reset_index(drop=True))

    # 4) Encode labels multi-hot
    def encode_labels(example):
        vec = [0.0] * NUM_LABELS
        idx = LABEL2ID.get(example['emotion'])
        if idx is not None:
            vec[idx] = 1.0
        example['labels'] = vec
        return example
    ds_train = ds_train.map(encode_labels)
    ds_val   = ds_val.map(encode_labels)
    ds_test  = ds_test.map(encode_labels)

    # 5) Tokenizer
    if tokenizer is None:
        print(f"Loading HF tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        print("Using custom tokenizer")
    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'], truncation=True, padding='max_length', max_length=128
        )
    ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=['sentence','emotion'])
    ds_val   = ds_val.map(tokenize_fn, batched=True, remove_columns=['sentence','emotion'])
    ds_test  = ds_test.map(tokenize_fn, batched=True, remove_columns=['sentence','emotion'])

    # 6) Cast labels to float32
    float_feats = Features({**ds_train.features, 'labels': Sequence(Value('float32'))})
    ds_train = ds_train.cast(float_feats)
    ds_val   = ds_val.cast(float_feats)
    ds_test  = ds_test.cast(float_feats)

    # 7) Format PyTorch
    ds_train.set_format(type='torch')
    ds_val.set_format(type='torch')
    ds_test.set_format(type='torch')

        # 8) WeightedRandomSampler pour train
    # Récupération des labels (N, num_labels)
    all_labels = torch.stack([ex['labels'] for ex in ds_train], dim=0)
    pos_freq = all_labels.mean(dim=0)
    neg_freq = 1.0 - pos_freq
    # éviter division par zéro / inf
    pos_weight = torch.where(
        pos_freq > 0,
        neg_freq / pos_freq,
        torch.zeros_like(pos_freq)
    )

    # Poids par exemple : somme des pos_weight des labels actifs
    example_weights = (all_labels * pos_weight.unsqueeze(0)).sum(dim=1)
    # remplacer nan et inf, garantir >0
    example_weights = torch.nan_to_num(example_weights, nan=0.0, posinf=0.0, neginf=0.0)
    example_weights = example_weights.clamp(min=0.0) + 1e-6

    sampler = WeightedRandomSampler(
        weights=example_weights.tolist(),
        num_samples=len(example_weights),
        replacement=True
    )

    # 9) Création des DataLoaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size*4, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size*4, shuffle=False)

    print("CSV data loading and preprocessing complete with sampler.")
    return train_loader, val_loader, test_loader, tokenizer
