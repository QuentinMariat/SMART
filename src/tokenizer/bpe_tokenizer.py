import re
import json
from collections import Counter, defaultdict
import torch
from pathlib import Path

class BPETokenizer:
    """
    Custom BPE tokenizer compatible with HuggingFace-style interface.
    Provides `train`, `encode`, `decode`, `get_tokens_with_values`, `__call__`, `save`, and `load`.
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.pad_token_id = None
        self.cls_token_id = None
        self.sep_token_id = None

    def train(self, corpus: list[str]):
        """
        Train BPE tokenizer on a list of text lines or a CSV file path.
        """
        # corpus must be a list of strings
        # corpus is list of strings
        lines = corpus

        # Initial tokenization with end-of-word marker with end-of-word marker
        tokenized = [list(word) + ['</w>']
                     for line in lines
                     for word in line.strip().split()]
        vocab_counter = Counter(' '.join(chars) for chars in tokenized)

        def get_stats(vocab):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            return pairs

        def merge_vocab(pair, vocab):
            merged = ''.join(pair)
            new_vocab = {}
            for word, freq in vocab.items():
                symbols = word.split()
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                new_vocab[' '.join(new_symbols)] = freq
            return new_vocab

        # BPE merge loop
        for _ in range(self.vocab_size):
            pairs = get_stats(vocab_counter)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            vocab_counter = merge_vocab(best, vocab_counter)

        # Build vocab tokens
        tokens = set(tok for word in vocab_counter for tok in word.split())
        tokens.update(['<unk>'])
        # Assign IDs
        self.vocab = {tok: idx for idx, tok in enumerate(sorted(tokens))}
        # Add special tokens if missing
        for tok in ['[PAD]', '[CLS]', '[SEP]']:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        # Store IDs
        self.pad_token_id = self.vocab['[PAD]']
        self.cls_token_id = self.vocab['[CLS]']
        self.sep_token_id = self.vocab['[SEP]']

    def encode(self, text: str, max_length: int = None):
        """
        Encode a single string to token IDs, adding [CLS] and [SEP].
        """
        tokens = list(text)
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == merge:
                    tokens[i:i+2] = [''.join(merge)]
                else:
                    i += 1
        if max_length:
            tokens = tokens[: max_length - 2]
        ids = [self.vocab.get(tok, self.vocab.get('<unk>')) for tok in tokens]
        return [self.cls_token_id] + ids + [self.sep_token_id]

    def decode(self, token_ids: list[int]):
        """
        Decode a list of token IDs back to string. Removes [CLS]/[SEP] and merges.
        """
        inv = {v: k for k, v in self.vocab.items()}
        toks = [inv.get(tid, '<unk>') for tid in token_ids
                if tid not in {self.pad_token_id, self.cls_token_id, self.sep_token_id}]
        return ''.join(toks)

    def get_tokens_with_values(self, text: str):
        """
        Return list of (token, token_id) pairs for the given text.
        """
        tokens = list(text)
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == merge:
                    tokens[i:i+2] = [''.join(merge)]
                else:
                    i += 1
        return [(tok, self.vocab.get(tok, self.vocab.get('<unk>'))) for tok in tokens]

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        all_ids = []
        for txt in texts:
            # Encode with truncate reserve for CLS/SEP
            ids = self.encode(txt, max_length if truncation else None)
            all_ids.append(ids)

        # Determine pad length
        if padding == 'max_length':
            pad_len = max_length
        elif padding in (True, 'longest'):
            pad_len = max(len(ids) for ids in all_ids)
        else:
            pad_len = None

        # Apply padding or keep as is
        if pad_len is not None:
            padded = [
                (ids + [self.pad_token_id] * (pad_len - len(ids)))[:pad_len]
                if len(ids) != pad_len else ids
                for ids in all_ids
            ]
            attention = [
                [1] * min(len(ids), pad_len) + [0] * max(0, pad_len - len(ids))
                for ids in all_ids
            ]
        else:
            padded = all_ids
            attention = [[1] * len(ids) for ids in all_ids]

        result = {'input_ids': padded, 'attention_mask': attention}
        if return_tensors == 'pt':
            result = {
                'input_ids': torch.tensor(padded, dtype=torch.long),
                'attention_mask': torch.tensor(attention, dtype=torch.long),
            }
        return result

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab, 'merges': self.merges},
                      f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = [tuple(m) for m in data['merges']]
        # Reload special IDs
        self.pad_token_id = self.vocab['[PAD]']
        self.cls_token_id = self.vocab['[CLS]']
        self.sep_token_id = self.vocab['[SEP]']
