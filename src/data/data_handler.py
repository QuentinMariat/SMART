from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from datasets import ClassLabel, Sequence, Value, Features, Dataset
from src.config.settings import MODEL_NAME, DATASET_NAME, NUM_LABELS, EMOTION_LABELS

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

def preprocess_dataset(ds, tokenizer, max_length=256):
    """Preprocess the dataset with improved text cleaning and label handling."""
    
    # Text cleaning function
    def clean_text(text):
        # Remove extra whitespace
        text = ' '.join(text.split())
        # RoBERTa handles casing internally, no need to lowercase
        return text
    
    # Apply text cleaning
    print("Cleaning text...")
    ds = ds.map(lambda x: {'text': clean_text(x['text'])})

    # Create multi-hot labels with label statistics
    def create_multi_hot_labels(examples):
        batch = []
        label_counts = [0] * NUM_LABELS
        
        for label_list in examples["labels"]:
            vec = [0.0] * NUM_LABELS
            for idx in label_list:
                if 0 <= idx < NUM_LABELS:
                    vec[idx] = 1.0
                    label_counts[idx] += 1
            batch.append(vec)
        
        examples["labels"] = batch
        return examples

    print("Creating multi-hot labels...")
    ds = ds.map(create_multi_hot_labels, batched=True)

    # Calculate and print label distribution
    label_counts = [0] * NUM_LABELS
    for example in ds["train"]:
        for idx, value in enumerate(example["labels"]):
            if value > 0:
                label_counts[idx] += 1
    
    print("\nLabel distribution in training set:")
    for idx, count in enumerate(label_counts):
        print(f"{EMOTION_LABELS[idx]}: {count} ({count/len(ds['train'])*100:.2f}%)")

    # Tokenize with dynamic max length based on dataset statistics
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True
        )

    print(f"\nTokenizing with max_length={max_length}...")
    train_tok = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tok = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])
    test_tok = ds["test"].map(tokenize_fn, batched=True, remove_columns=["text"])

    # Calculate sequence length statistics
    lengths = [len(tokenizer.tokenize(text)) for text in ds["train"]["text"]]
    print(f"\nSequence length statistics:")
    print(f"Mean: {sum(lengths)/len(lengths):.1f}")
    print(f"Max: {max(lengths)}")
    print(f"95th percentile: {sorted(lengths)[int(len(lengths)*0.95)]}")

    # Cast labels to float
    def cast_labels(examples):
        examples["labels"] = [[float(v) for v in lab] for lab in examples["labels"]]
        return examples

    train_tok = train_tok.map(cast_labels, batched=True)
    val_tok = val_tok.map(cast_labels, batched=True)
    test_tok = test_tok.map(cast_labels, batched=True)

    # Set PyTorch format
    columns = ["input_ids", "attention_mask", "labels"]  # Removed token_type_ids as RoBERTa doesn't use them
    train_tok.set_format("torch", columns=columns)
    val_tok.set_format("torch", columns=columns)
    test_tok.set_format("torch", columns=columns)

    print("\nData preprocessing complete.")
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
    from transformers import AutoTokenizer
    from datasets import Dataset
    import math

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if load_from_cache and cache_dir:
        print(f"Loading dataset from cache at {cache_dir}...")
        tokenized_dataset = Dataset.load_from_disk(cache_dir)
    else:
        print("Preprocessing Wikipedia dataset...")
        texts = list(wiki_dataset)
        dataset = Dataset.from_dict({"text": texts})

        # Tokenisation sans truncation
        def tokenize_no_trunc(batch):
            return tokenizer(batch["text"], return_special_tokens_mask=True)

        tokenized = dataset.map(tokenize_no_trunc, batched=True, remove_columns=["text"])

        # Découpage en chunks de max_length
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // max_length) * max_length  # tronquer à un multiple de max_length
            result = {
                k: [concatenated[k][i:i + max_length] for i in range(0, total_length, max_length)]
                for k in concatenated.keys()
            }
            return result

        tokenized_dataset = tokenized.map(group_texts, batched=True)

        # Conversion PyTorch
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"] if "token_type_ids" in tokenized_dataset.column_names else ["input_ids", "attention_mask"])

        if cache_dir:
            print(f"Saving dataset to cache at {cache_dir}...")
            tokenized_dataset.save_to_disk(cache_dir)

    # Split
    num_examples = len(tokenized_dataset)
    train_size = int(train_ratio * num_examples)
    val_size = int(val_ratio * num_examples)

    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, train_size + val_size))
    test_dataset = tokenized_dataset.select(range(train_size + val_size, num_examples))

    print(f"Preprocessing complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, tokenizer
