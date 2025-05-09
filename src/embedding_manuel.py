import torch
import torch.nn as nn
import torch.optim as optim

class ManualEmbeddingTrainer:
    def __init__(self, vocab_size, embedding_dim, lr=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_weights = nn.Parameter(torch.randn(vocab_size, embedding_dim, requires_grad=True))
        self.optimizer = optim.SGD([self.embedding_weights], lr=lr)

    def get_embedding(self, token_ids):
        """Retourne les embeddings pour une liste de token IDs"""
        return self.embedding_weights[token_ids]

    def train_token(self, token_id, target_vector, steps=100, verbose=True):
        """Entraîne un seul token vers un vecteur cible"""
        for step in range(steps):
            self.optimizer.zero_grad()
            embedded = self.embedding_weights[token_id]  # Shape: (1, embedding_dim)
            loss = nn.functional.mse_loss(embedded.squeeze(0), target_vector)
            loss.backward()
            self.optimizer.step()
            if verbose and step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")
        return self.embedding_weights[token_id].detach()

# Exemple d'utilisation
if __name__ == "__main__":
    vocab_size = 5500
    embedding_dim = 256
    trainer = ManualEmbeddingTrainer(vocab_size, embedding_dim)

    # Affiche un vecteur d'embedding avant apprentissage
    token_ids = torch.tensor([12, 432, 8, 999])
    print("Embedding initial pour le token 12 :")
    print(trainer.get_embedding(token_ids)[0])

    # Entraîne le token 12 à ressembler à un vecteur de 1
    target = torch.ones(embedding_dim)
    learned_vector = trainer.train_token(torch.tensor([12]), target, steps=100)

    print("\nVecteur appris pour le token 12 :")
    print(learned_vector)
