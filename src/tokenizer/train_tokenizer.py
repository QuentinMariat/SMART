import csv
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.normalizer import normalize_text
from src.tokenizer.wrapped_bpe import WrappedBPETokenizer
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer

def load_corpus_from_csv(csv_path: str) -> list[str]:
    """
    Lit un fichier texte (une colonne, une ligne = un commentaire brut),
    applique la normalisation et renvoie la liste des textes.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.error(f"Fichier {csv_path} introuvable.")
        return []

    corpus: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            text = raw_line.strip()
            if not text:
                continue
            normalized = normalize_text(text)
            corpus.append(normalized)

    logger.info(f"{len(corpus)} commentaires chargés et normalisés depuis {csv_path}.")
    return corpus

def main():
    bpe = BPETokenizer(vocab_size=7500)
    
    # remplace ici par ton chemin vers le CSV

    csv_path = "data/raw/entrainement.csv"  
    tokenizer_save_path = "data/tokenizer_files/tokenizer.json"

    if not Path(csv_path).exists():
        logger.error(f"Fichier {csv_path} introuvable.")
        return

    
    # Charger corpus depuis CSV
    corpus = load_corpus_from_csv(csv_path)
    logger.info(f"{len(corpus)} textes chargés depuis le CSV.")
    
    """
    # Entraîner le tokenizer
    bpe.train(corpus)
    bpe.save(tokenizer_save_path)
    logger.info("Tokenizer entraîné et sauvegardé.")
    """
    
    # Charger le tokenizer depuis le fichier JSON
    bpe_save_path = "data/tokenizer_files/tokenizer.json"
    if Path(bpe_save_path).exists():
        bpe.load(bpe_save_path)
        logger.info("Tokenizer chargé depuis le fichier JSON.")
    

    # Wrapper pour HuggingFace
    #tokenizer = WrappedBPETokenizer(bpe, do_lower_case=True)

    # Test rapide
    logger.info("Testing bpe...")
    test_text = "welcome Quentin. welcome everyone!"
    test_text = normalize_text(test_text)
    encoded_text = bpe.encode(test_text, max_length=128)
    decoded_text = bpe.decode(encoded_text)
    tokens_with_values = bpe.get_tokens_with_values(test_text)

    logger.info(f"Tokens with values: {tokens_with_values}")
    logger.info(f"Encoded text '{test_text}': {encoded_text}")
    logger.info(f"Decoded text: {decoded_text}")

    """
    text = "This game looks amazing!"
    enc = tokenizer(text)

    print("Input IDs:", enc["input_ids"])
    print("Attention Mask:", enc["attention_mask"])
    print("Decoded:", tokenizer.decode(enc["input_ids"]))

    print("CLS ID:", tokenizer.convert_tokens_to_ids("[CLS]"))
    print("SEP ID:", tokenizer.convert_tokens_to_ids("[SEP]"))
    print("PAD ID:", tokenizer.convert_tokens_to_ids("[PAD]"))
    print("UNK ID:", tokenizer.convert_tokens_to_ids("[UNK]"))

    batch = tokenizer(
    ["I love this trailer", "Horrible experience"],
    padding=True, truncation=True, max_length=10
    )

    for seq_ids, tok_ids, mask in zip(
        batch["input_ids"],
        batch["token_type_ids"],
        batch["attention_mask"]
    ):
        print("IDs :", seq_ids)
        print("TT  :", tok_ids)
        print("AM  :", mask)
        print("Dec :", tokenizer.decode(seq_ids))
        print("---")
    """

if __name__ == "__main__":
    main()
