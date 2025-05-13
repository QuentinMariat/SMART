from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from datasets import ClassLabel, Sequence, Value, Features, Dataset
from src.config.settings import MODEL_NAME, DATASET_NAME, NUM_LABELS

def load_tokenizer(tokenizer=None):
    """Load or return a tokenizer for the model."""
    if tokenizer is None:
        print(f"Loading HF tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        print("Using custom BPE tokenizer")
    return tokenizer

def loading_dataset(max_train_samples=None, max_val_samples=None, max_test_samples=None):
    """Load and subsample the dataset."""
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    # Subsample train split if requested
    if max_train_samples is not None:
        ds["train"] = ds["train"].shuffle(seed=42).select(
            list(range(min(max_train_samples, len(ds["train"]))))
        )
        print(f"→ Train subset: {len(ds['train'])} exemples retenus")
    
    # Subsample validation split if requested
    if max_val_samples is not None:
        ds["validation"] = ds["validation"].shuffle(seed=42).select(
            list(range(min(max_val_samples, len(ds["validation"]))))
        )
        print(f"→ Validation subset: {len(ds['validation'])} exemples retenus")

    # Subsample test split if requested
    if max_test_samples is not None:
        ds["test"] = ds["test"].shuffle(seed=42).select(
            list(range(min(max_test_samples, len(ds["test"]))))
        )
        print(f"→ Test subset: {len(ds['test'])} exemples retenus")

    return ds

def preprocess_dataset(ds, tokenizer):
    """Preprocess the dataset: create multi-hot labels, tokenize, cast labels, and set PyTorch format."""
    # Create multi-hot labels
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

    # Tokenize each split
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
    val_tok = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenizing test split...")
    test_tok = ds["test"].map(tokenize_fn, batched=True, remove_columns=["text"])

    # Cast labels to float
    def cast_labels(examples):
        examples["labels"] = [[float(v) for v in lab] for lab in examples["labels"]]
        return examples

    train_tok = train_tok.map(cast_labels, batched=True)
    val_tok = val_tok.map(cast_labels, batched=True)
    test_tok = test_tok.map(cast_labels, batched=True)

    # Define float32 schema for labels
    float_features = Features({
        **train_tok.features,
        "labels": Sequence(Value("float32"))
    })

    train_tok = train_tok.cast(float_features)
    val_tok = val_tok.cast(float_features)
    test_tok = test_tok.cast(float_features)

    # Set PyTorch format
    train_tok.set_format("torch")
    val_tok.set_format("torch")
    test_tok.set_format("torch")

    print("Data preprocessing complete.")
    return train_tok, val_tok, test_tok, tokenizer

def cache_dataset(self):
    """Cache the dataset to disk for faster future use."""
    # Sauvegarder le dataset prétraité dans un répertoire de cache
    print(f"Caching dataset to {self.cache_dir}...")
    self.limited_dataset.save_to_disk(self.cache_dir)
    
def load_from_cache(self):
    """Load the cached dataset if it exists."""
    try:
        print("Loading dataset from cache...")
        cached_dataset = Dataset.load_from_disk(self.cache_dir)
        return cached_dataset
    except Exception as e:
        print(f"Cache not found or error occurred: {e}")
        return None

def preprocess_wikipedia_mlm(wiki_dataset, model_name="bert-base-uncased", max_length=128, train_ratio=0.8, val_ratio=0.1, load_from_cache=False, cache_dir=None):
    """
    Preprocess Wikipedia dataset for BERT MLM pretraining (without masking).
    
    Args:
        wiki_dataset: WikipediaMLMDataset instance.
        model_name (str): Hugging Face model name for tokenizer (default: bert-base-uncased).
        max_length (int): Maximum sequence length (default: 512).
        train_ratio (float): Proportion of data for training (default: 0.8).
        val_ratio (float): Proportion of data for validation (default: 0.1).
        cache_dir (str): Directory to save the cached dataset (default: None).
        load_from_cache (bool): Whether to load from cache (default: False).
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # 1. Charger le dataset si dispo 
    if load_from_cache and cache_dir:
        print(f"Loading dataset from cache at {cache_dir}...")
        tokenized_dataset = Dataset.load_from_disk(cache_dir)

    else:
        print("Preprocessing Wikipedia dataset...")
        texts = list(wiki_dataset)  # Si ce sont des str
        dataset = Dataset.from_dict({"text": texts})

        def tokenize_batch(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
        )
        
        tokenized_dataset = dataset.map(tokenize_batch, batched=True)

        # 4. Mettre au format PyTorch directement
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Gérer la sauvegarde du dataset
        if not(load_from_cache) and cache_dir:
            print(f"Saving dataset to cache at {cache_dir}...")
            tokenized_dataset.save_to_disk(cache_dir)


    
    # Split dataset
    print("Splitting dataset...")
    # Split dataset
    num_examples = len(tokenized_dataset)
    train_size = int(train_ratio * num_examples)
    val_size = int(val_ratio * num_examples)

    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, train_size + val_size))
    test_dataset = tokenized_dataset.select(range(train_size + val_size, num_examples))

    
    print(f"Preprocessing complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, tokenizer