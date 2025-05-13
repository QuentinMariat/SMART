import torch

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    Prépare les entrées masquées et les labels pour MLM.
    - 15% des tokens sont candidats au masquage.
    - 80% remplacés par [MASK], 10% par un token aléatoire, 10% inchangés.
    """
    labels = inputs.clone()
    # Matrice de probabilité
    prob_matrix = torch.full(labels.shape, mlm_probability)
    # Ne pas masquer les tokens spéciaux
    special_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    prob_matrix.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), 0.0)

    # Sélection des positions à masquer
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # ignore_index pour CrossEntropyLoss

    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% -> random
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 10% -> inchangés (les autres masked_indices restants)
    return inputs, labels