import re
import json
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def get_vocab(self):
        return self.vocab

    def train(self, corpus: list[str]):
        # Tokenisation initiale : caractères (sans </w>)
        tokenized_words = [list(word) for line in corpus for word in line.strip().split()]
        vocab_counter = Counter([" ".join(chars) for chars in tokenized_words])

        def get_stats(vocab):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            return pairs

        def merge_vocab(pair, vocab):
            pattern = re.escape(" ".join(pair))
            regex = re.compile(r"(?<!\S)" + pattern + r"(?!\S)")
            merged = "".join(pair)
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = regex.sub(lambda m: merged, word)
                new_vocab[new_word] = freq
            return new_vocab

        for _ in range(self.vocab_size):
            pairs = get_stats(vocab_counter)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            vocab_counter = merge_vocab(best, vocab_counter)

        # Construire le vocab final
        tokens = set()
        for word in vocab_counter:
            tokens.update(word.split())
        tokens.update(["<unk>", " "])
        self.vocab = {token: idx for idx, token in enumerate(sorted(tokens))}

    def encode(self, text: str):
        # On part de la liste de caractères
        tokens = list(text)
        # Appliquer toutes les merges BPE
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == merge:
                    tokens[i:i+2] = ["".join(merge)]
                else:
                    i += 1
        # Convertir en IDs
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

    def decode(self, token_ids: list[int]):
        inv = {v:k for k,v in self.vocab.items()}
        # Reconstruire la chaîne
        return "".join(inv.get(tid, "<unk>") for tid in token_ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges},
                      f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]

    def get_tokens_with_values(self, text: str):
        tokens = list(text)
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == merge:
                    tokens[i:i+2] = ["".join(merge)]
                else:
                    i += 1
        return [(tok, self.vocab.get(tok, self.vocab["<unk>"]))
                for tok in tokens]

    def __call__(self, texts, padding=True, truncation=True, max_length=512):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) for t in texts]
        # tronquer
        input_ids = [e[:max_length] for e in encoded]
        # attention mask
        attention_mask = [[1]*len(ids) for ids in input_ids]
        return {"input_ids": input_ids,
                "attention_mask": attention_mask}
