import pandas as pd
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import torch
from src.config.settings import MODEL_NAME, DATASET_NAME, NUM_LABELS, LABEL2ID


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
    random_seed: int = 42
):
    """
    Charge un CSV (columns: sentence, emotion), split en train/val/test,
    encode en multi-hot et tokenize.

    Args:
        csv_path: chemin vers le CSV.
        tokenizer: tokenizer HF, ou None pour charger AUTO.
        max_*_samples: sous-échantillonnage.
        train_size, val_size, test_size: proportions (somme =1).
        random_seed: seed.
    Returns:
        train_ds, val_ds, test_ds, tokenizer
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    # Split
    df_train, df_temp = train_test_split(df, train_size=train_size, random_state=random_seed, shuffle=True)
    rel_val = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(df_temp, train_size=rel_val, random_state=random_seed, shuffle=True)

    # Sous-échantillonnage
    def subsample(df_split, max_s, name):
        if max_s is not None and len(df_split) > max_s:
            df_split = df_split.sample(n=max_s, random_state=random_seed)
            print(f"→ {name} subset: {len(df_split)} exemples retenus")
        return df_split

    df_train = subsample(df_train, max_train_samples, "Train")
    df_val   = subsample(df_val,   max_val_samples,   "Validation")
    df_test  = subsample(df_test,  max_test_samples,  "Test")

    # HF Dataset
    ds_train = Dataset.from_pandas(df_train.reset_index(drop=True))
    ds_val   = Dataset.from_pandas(df_val.reset_index(drop=True))
    ds_test  = Dataset.from_pandas(df_test.reset_index(drop=True))

    # Encode labels multi-hot
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

    # Tokenizer
    if tokenizer is None:
        print(f"Loading HF tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        print("Using custom tokenizer")

    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'], truncation=True, padding='max_length', max_length=128
        )

    # Tokenisation
    ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=['sentence', 'emotion'])
    ds_val   = ds_val.map(tokenize_fn, batched=True, remove_columns=['sentence', 'emotion'])
    ds_test  = ds_test.map(tokenize_fn, batched=True, remove_columns=['sentence', 'emotion'])

    # Cast labels to float32
    float_feats = Features({**ds_train.features, 'labels': Sequence(Value('float32'))})
    ds_train = ds_train.cast(float_feats)
    ds_val   = ds_val.cast(float_feats)
    ds_test  = ds_test.cast(float_feats)

    # Format PyTorch
    ds_train.set_format(type='torch')
    ds_val.set_format(type='torch')
    ds_test.set_format(type='torch')

    print("CSV data loading and preprocessing complete.")
    return ds_train, ds_val, ds_test, tokenizer
