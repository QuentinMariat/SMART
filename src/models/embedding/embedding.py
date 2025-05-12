import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):   
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class BERTEmbedding(nn.Module):
    """
    BERT Embedding incluant :
        - Token Embedding
        - Segment Embedding
        - Positional Embedding
    En assumant que les 4 derniers tokens du vocab sont : [CLS], [SEP], [PAD], [UNK] (dans cet ordre)
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=512):
        super().__init__()

        # Déduction des IDs spéciaux à partir du vocab_size
        self.CLS_ID = vocab_size - 4
        self.SEP_ID = vocab_size - 3
        self.PAD_ID = vocab_size - 2
        self.UNK_ID = vocab_size - 1

        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=self.PAD_ID)
        self.segment = nn.Embedding(3, embed_size, padding_idx=self.PAD_ID)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        token_emb = self.token(sequence)
        pos_emb = self.position(sequence)
        seg_emb = self.segment(segment_label)
        return self.dropout(token_emb + pos_emb + seg_emb)


vocab_size = 5504  # Les 4 derniers sont [CLS, SEP, PAD, UNK]
embed_size = 256

bert_embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=embed_size)

# Simulons une séquence contenant [CLS], [SEP], [PAD], [UNK] à la fin
# On les place explicitement ici pour l’exemple
cls_id = vocab_size - 4
sep_id = vocab_size - 3
pad_id = vocab_size - 2
unk_id = vocab_size - 1

tokens_ids = [12, 432, 8, 77, cls_id, sep_id, pad_id, unk_id]  # longueur = 8
token_ids = torch.tensor([tokens_ids])  # (1, 8)
segment_ids = torch.zeros_like(token_ids)

embeddings = bert_embedding(token_ids, segment_ids)

print("Embedding du token [PAD] :")
print(embeddings[0, -2])  # devrait être un vecteur neutre (non entraîné)

