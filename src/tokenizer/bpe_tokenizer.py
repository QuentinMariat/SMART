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
        # Tokenisation initiale : caractères + </w> pour marquer fin de mot
        # On conserve les espaces et ajoute </w> seulement à la fin de chaque ligne complète
        tokenized_words = [list(word) + ["</w>"] for line in corpus for word in line.strip().split()]
        vocab_counter = Counter([" ".join(line) for line in tokenized_words])

        def get_stats(vocab):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            return pairs

        def merge_vocab(pair, vocab):
            # On échappe la paire pour la recherche, mais
            # on utilise un callable pour remplacer littéralement.
            regex = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
            merged = "".join(pair)
            new_vocab = {}
            for word, freq in vocab.items():
                # on passe un callable pour que merged soit inséré tel quel
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

        # Création du vocabulaire final à partir des tokens (et non des mots concaténés)
        tokens = set()
        for word in vocab_counter:
            tokens.update(word.split())

        tokens.update(["<unk>", " "])
        self.vocab = {token: idx for idx, token in enumerate(sorted(tokens))}

    def encode(self, text: str):
        tokens = list(text) + ["</w>"]
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == merge:
                    tokens[i:i+2] = ["".join(merge)]
                else:
                    i += 1
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def decode(self, token_ids: list[int]):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded_tokens = [inv_vocab.get(tid, "<unk>") for tid in token_ids]
        text = "".join(decoded_tokens)
        return text.replace("</w>", " ")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.merges = [tuple(pair) for pair in data["merges"]]

    def get_tokens_with_values(self, text):
        tokens = list(text) + ["</w>"]
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == merge:
                    tokens[i:i+2] = [''.join(merge)]
                else:
                    i += 1
        return [(token, self.vocab.get(token, self.vocab["<unk>"])) for token in tokens]
