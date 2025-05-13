# src/tokenizer/wrapped_bpe.py

import os
from pathlib import Path
from typing import List, Optional, Union
from typing import Optional, List, Dict, Any

import json
from transformers import PreTrainedTokenizer
from transformers.utils import logging

from src.tokenizer.bpe_tokenizer import BPETokenizer

logger = logging.get_logger(__name__)

class WrappedBPETokenizer(PreTrainedTokenizer):
    """
    Expose ton BPETokenizer comme un tokenizer HuggingFace-compatible,
    prêt pour entraîner un BERT from scratch.
    """

    # On définit les tokens spéciaux
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    unk_token = "[UNK]"

    def __init__(self, bpe: BPETokenizer, do_lower_case: bool = True, **kwargs):
        # 1) Construire en premier ton vocab BPE + spéciaux
        self.bpe = bpe
        self.do_lower_case = do_lower_case

        # Cloner le vocab BPE
        self.vocab = dict(bpe.get_vocab())
        # Ajouter les spéciaux
        for tok in [self.cls_token, self.sep_token, self.pad_token, self.unk_token]:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # 2) Appeler le super avec tes tokens spéciaux
        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            cls_token=self.cls_token,
            sep_token=self.sep_token,
            **kwargs
        )

        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]

        self.model_max_length = 512

    def _tokenize(self, text: str) -> List[str]:
        # 1) Normalize casing
        if self.do_lower_case:
            text = text.lower()

        # 2) Appeler ton BPE pour découper
        #    La méthode encode() retourne déjà les ids, 
        #    mais on veut les tokens strings ici :
        ids = self.bpe.encode(text)
        tokens = [ self.inv_vocab.get(i, self.unk_token) for i in ids ]
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        # Transforme token -> id
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        # Transforme id -> token
        return self.inv_vocab.get(index, self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict:
        return dict(self.vocab)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Écrit un 'vocab.txt' (un token par ligne), et le merges.json de ton BPE.
        Le Trainer HuggingFace s'appuiera sur vocab.txt.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # 1) vocab.txt
        vocab_file = save_directory / "vocab.txt"
        with open(vocab_file, "w", encoding="utf-8") as vf:
            # On écrit un token par ligne
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                vf.write(token + "\n")
        logger.info(f"Saved vocab.txt with {len(self.vocab)} tokens to {vocab_file}")

        # 2) merges.json (ton BPE)
        merges_file = save_directory / "merges.json"
        with open(merges_file, "w", encoding="utf-8") as mf:
            json.dump(self.bpe.merges, mf, ensure_ascii=False, indent=2)
        logger.info(f"Saved merges.json to {merges_file}")

        # 3) (Optionnel) tokenizer_config.json pour HuggingFace
        config = {
            "do_lower_case": self.do_lower_case,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "model_max_length": 512
        }
        with open(save_directory / "config.json", "w", encoding="utf-8") as cf:
            json.dump(config, cf, ensure_ascii=False, indent=2)
        logger.info(f"Saved tokenizer_config.json to {save_directory}")

        return {"vocab_file": str(vocab_file)}
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        # On saute la logique HF et on appelle directement notre BPE
        return self._tokenize(text)


    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        # 1) Filtrer les specials si demandé
        if skip_special_tokens:
            special_ids = {
                self.cls_token_id,
                self.sep_token_id,
                self.pad_token_id,
                self.unk_token_id
            }
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        # 2) Convertir IDs -> chaînes BPE
        tokens = [ self.inv_vocab.get(tid, self.unk_token) for tid in token_ids ]

        # 3) Recomposer la chaîne avec ton BPE
        #    (ici, tokens sont déjà les sous-mots mergés)
        text = "".join(tokens)

        return text
    
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)
    
    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors=None,
        **kwargs
    ) -> Dict[str, Any]:
        # 1) Tokenisation
        tokens = self._tokenize(text)
        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]

        # 2) Ajouter [CLS] et [SEP]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]

        # 3) Troncature
        if max_length is None:
            max_length = self.model_max_length
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        # 4) Padding
        if padding and max_length is not None and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len

        # 5) Masques & token_type_ids
        attention_mask = [1 if i < len(ids) else 0 for i in range(len(ids))]
        token_type_ids = [0] * len(ids)

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }