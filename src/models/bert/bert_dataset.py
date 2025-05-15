from datasets import load_dataset, Dataset

class WikipediaMLMDataset:
    """A class to load and iterate over Wikipedia dataset for MLM pretraining."""
    
    def __init__(self, language="en", version="20231101", split="train", num_examples=1000):
        """
        Initialize the Wikipedia dataset loader.
        
        Args:
            language (str): Language code (e.g., 'en' for English, 'fr' for French).
            version (str): Dataset version (e.g., '20231101').
            split (str): Dataset split (e.g., 'train').
            num_examples (int): Number of examples to load.
        """
        self.language = language
        self.version = version
        self.split = split
        self.num_examples = num_examples
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(
            "wikimedia/wikipedia",
            f"{version}.{language}",
            split=split,
            streaming=False,
            trust_remote_code=True
        )
        
        print(f"Loading dataset from version {version}...")
        self.dataset = load_dataset(
            "wikimedia/wikipedia",
            f"{version}.{language}",
            split=split,
            streaming=False  # Nous n'utilisons pas le streaming ici
        )
        
        # Limiter le nombre d'exemples téléchargés
        self.limited_dataset = self.dataset.select(range(min(self.num_examples, len(self.dataset))))
        print(f"Loaded {len(self.limited_dataset)} examples from {split} split.")
        


    
    
    def get_iterator(self):
        """
        Get an iterator over the text column of the dataset.
        
        Yields:
            str: Text content of each Wikipedia article.
        """
        for example in self.limited_dataset:
            yield example["text"]
    
    def __iter__(self):
        """Make the class iterable."""
        return self.get_iterator()
    
    def __len__(self):
        """Return the number of examples."""
        return self.num_examples

# Example usage:
if __name__ == "__main__":
    # Initialize the dataset (English, 5 examples)
    wiki_dataset = WikipediaMLMDataset(language="en", version="20231101", num_examples=5)
    
    # Iterate over the dataset
    for i, text in enumerate(wiki_dataset):
        print(f"=== Article {i} ===\n{text[:500]}\n")