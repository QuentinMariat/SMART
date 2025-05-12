# src/tokenizer/wrapped_bpe.py

import os
from pathlib import Path
from typing import List, Optional, Union

import json
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging

from src.tokenizer.bpe_tokenizer import BPETokenizer

logger = logging.get_logger(__name__)

class WrappedBPETokenizer(PreTrainedTokenizerBase):
    """
    Expose ton BPETokenizer comme un tokenizer HuggingFace-compatible,
    prêt pour entraîner un BERT from scratch.
    """

    # On définit les tokens spéciaux
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    unk_token = "[UNK]"

    def __init__(
        self,
        bpe: BPETokenizer,
        do_lower_case: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bpe = bpe
        self.do_lower_case = do_lower_case

        # Construire vocab spécial + BPE
        # On commence par cloner le vocab bpe
        self.vocab = dict(bpe.get_vocab())

        # On ajoute les spéciaux si besoin
        for tok in [self.cls_token, self.sep_token, self.pad_token, self.unk_token]:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

        # Inverse pour decoder
        self.inv_vocab = {v:k for k,v in self.vocab.items()}

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
        with open(save_directory / "tokenizer_config.json", "w", encoding="utf-8") as cf:
            json.dump(config, cf, ensure_ascii=False, indent=2)
        logger.info(f"Saved tokenizer_config.json to {save_directory}")

        return {"vocab_file": str(vocab_file)}

    # On hérite __call__ / encode_plus / batch_encode_plus de PreTrainedTokenizerBase,
    # qui s'appuient sur _tokenize et _convert_* ci-dessus.

