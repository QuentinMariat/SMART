import json
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.prepare_corpus import extract_texts_from_jsonl_file


 
def load_corpus_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("texts", [])
 
def main():

    tokenizer = BPETokenizer(vocab_size=2000)

    corpus_path = "data/splits/corpus.json"
    tokenizer_save_path = "data/tokenizer.json"

    if not Path(corpus_path).exists():
        logger.error(f"Fichier {corpus_path} introuvable.")
        return

    # Charger corpus depuis JSON
    corpus = load_corpus_from_json(corpus_path)
    logger.info(f"{len(corpus)} textes chargés depuis le corpus.")

    # Entraîner le tokenizer
    tokenizer.train(corpus)
    tokenizer.save(tokenizer_save_path)
    logger.info("Tokenizer entraîné et sauvegardé.")
 
    logger.info("Testing tokenizer...")
    test_text = "Bonjour, comment ça va?"
    encoded_text = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(encoded_text)
 
    tokens_with_values = tokenizer.get_tokens_with_values(test_text)
    logger.info(f"Tokens with values: {tokens_with_values}")
    logger.info(f"Encoded text {test_text}: {encoded_text}")
    logger.info(f"Decoded text: {decoded_text}")
 
 
if __name__ == "__main__":
    main()
