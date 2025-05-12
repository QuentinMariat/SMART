import csv
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.normalizer import normalize_text

def load_corpus_from_csv(csv_path):
    """
    Lit un CSV à une colonne où chaque ligne est un commentaire brut,
    applique la normalisation et renvoie la liste des textes.
    """
    corpus = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Si votre CSV comporte une en-tête, décommentez la ligne suivante :
        # headers = next(reader, None)

        for row in reader:
            # Assure-toi qu'il y a bien au moins une colonne
            if not row:
                continue
            text = row[0].strip()
            if not text:
                continue

            # Normalise le texte (minuscules, suppression accents, ponctuation…)
            normalized = normalize_text(text)
            corpus.append(normalized)

    logger.info(f"{len(corpus)} commentaires chargés et normalisés depuis {csv_path}.")
    return corpus

def main():
    tokenizer = BPETokenizer(vocab_size=10000)
    
    # remplace ici par ton chemin vers le CSV

    csv_path = "data/raw/casual_data_windows.csv"  
    tokenizer_save_path = "data/tokenizer_files/tokenizer.json"

    if not Path(csv_path).exists():
        logger.error(f"Fichier {csv_path} introuvable.")
        return

    # Charger corpus depuis CSV
    corpus = load_corpus_from_csv(csv_path)
    logger.info(f"{len(corpus)} textes chargés depuis le CSV.")

    
    # Entraîner le tokenizer
    tokenizer.train(corpus)
    tokenizer.save(tokenizer_save_path)
    logger.info("Tokenizer entraîné et sauvegardé.")
    
    """
    # Charger le tokenizer depuis le fichier JSON
    tokenizer_save_path = "data/tokenizer_files/tokenizer.json"
    if Path(tokenizer_save_path).exists():
        tokenizer.load(tokenizer_save_path)
        logger.info("Tokenizer chargé depuis le fichier JSON.")
    """

    # Test rapide
    logger.info("Testing tokenizer...")
    test_text = "welcome Quentin, welcome everyone!"
    test_text = normalize_text(test_text)
    encoded_text = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(encoded_text)
    tokens_with_values = tokenizer.get_tokens_with_values(test_text)

    logger.info(f"Tokens with values: {tokens_with_values}")
    logger.info(f"Encoded text '{test_text}': {encoded_text}")
    logger.info(f"Decoded text: {decoded_text}")

if __name__ == "__main__":
    main()
