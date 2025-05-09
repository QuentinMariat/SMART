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
        self.pe = pe.unsqueeze(0)   
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
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
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence, segment_label):
        if segment_label is None:
          segment_label = torch.zeros_like(sequence)
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
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
