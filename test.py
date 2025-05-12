from datasets import load_dataset
import torch

# Charge le dataset raw
ds = load_dataset("go_emotions", name="raw")

# Compte total
total_count = len(ds["train"])
print(f"Total d'exemples : {total_count}")

# Compte unique (sur le texte)
unique_text_count = len(set(ds["train"]["text"]))
print(f"Nombre de textes uniques : {unique_text_count}")
