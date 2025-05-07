import json
from pathlib import Path

def extract_texts_from_jsonl_file(file_path, nb_lines=50):
    corpus = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= nb_lines:
                break
            try:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if text:
                    corpus.append(text)
            except json.JSONDecodeError:
                print(f"Ligne {i+1} illisible dans {file_path}")
    return corpus

def save_corpus(corpus, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"texts": corpus}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_dir = "data/raw/train.jsonl"       # Dossier où tu mets tous tes JSON
    output_file = "data/splits/corpus.json"      # Fichier de sortie attendu par ton tokenizer

    texts = extract_texts_from_jsonl_file(input_dir, nb_lines=50)
    print(f"{len(texts)} documents récupérés.")
    
    save_corpus(texts, output_file)
    print(f"Corpus sauvegardé dans {output_file}")
