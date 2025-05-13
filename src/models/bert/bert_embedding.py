import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=128, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)

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

if __name__ == "__main__":
    vocab_size = 5500
    embed_size = 256
    seq_len = 10  # Longueur de la séquence

    # Crée un batch de 1 séquence de longueur 10
    token_ids = torch.tensor([[12, 432, 8, 999, 1, 234, 77, 321, 1024, 45]])  # (1, 10)
    segment_ids = torch.zeros_like(token_ids)  # Tous dans le segment 0

    # Initialise l'embedding
    bert_embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=embed_size, seq_len=seq_len)

    # Calcule les embeddings
    embeddings = bert_embedding(token_ids)  # (1, 10, 256)

    # Affiche le premier vecteur d'embedding
    print("Premier vecteur d'embedding :")
    print(embeddings[0, 0])  # Shape: (256,)