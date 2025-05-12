# src/utils/text_normalizer.py

import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalise un texte en :
      1) Passe tout en minuscules
      2) Supprime les accents
      3) Remplace les URLs par un token <URL>
      4) Supprime la ponctuation (sauf </w> si jamais présent)
      5) Condense les espaces multiples en un seul
      6) Strip en début/fin
    """
    # 1) minuscules
    text = text.lower()

    # 2) supprimer accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    #Supprimer les caractères répétés trop souvent
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)

    # 3) remplacer URL éventuelles
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)

    # 4) ponctuation → on garde </w> séparément géré par le tokenizer
    #    ainsi que l’espace
    text = re.sub(r"[^\w\s</>]", " ", text)

    # 5) espaces multiples
    text = re.sub(r"\s+", " ", text)

    # 6) strip
    return text.strip()
