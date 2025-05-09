import torch
import torch.optim as optim
import torch.nn as nn

# Paramètres
vocab_size = 5500
embedding_dim = 256

# Création manuelle de la matrice d'embedding
embedding_weights = torch.randn(vocab_size, embedding_dim)  # Initialisation aléatoire

# Exemple de tokens (IDs)
token_ids = torch.tensor([12, 432, 8, 999])  # IDs entre 0 et vocab_size-1

# Lookup manuel
embedded_tokens = embedding_weights[token_ids]

# Affiche le vecteur du premier token
print(embedded_tokens[0])  # Vecteur pour le token avec ID 12


# Création manuelle de la matrice d'embedding avec gradients activés
embedding_weights = nn.Parameter(torch.randn(vocab_size, embedding_dim, requires_grad=True))

# Optimiseur
optimizer = optim.SGD([embedding_weights], lr=0.1)

# Exemple de token ID et d'objectif fictif (on veut que le vecteur s'approche d'un vecteur cible)
token_id = torch.tensor([42])
target_vector = torch.ones(embedding_dim)  # But : rapprocher l'embedding du token 42 de ce vecteur

# Entraînement sur une itération
for step in range(100):
    optimizer.zero_grad()

    # Lookup manuel
    embedded = embedding_weights[token_id]  # Shape: (1, embedding_dim)

    # Calcul d'une perte (ex: MSE entre l'embedding et le vecteur cible)
    loss = torch.nn.functional.mse_loss(embedded.squeeze(0), target_vector)

    # Backpropagation
    loss.backward()

    # Mise à jour des poids
    optimizer.step()

    print(f"Loss: {loss.item()}")