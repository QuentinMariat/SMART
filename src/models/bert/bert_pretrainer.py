import torch
import torch.nn 
from models.bert.bert import BERT


class BERTMLMHead(torch.nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # Transformation avant prédiction
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.LayerNorm(d_model)
        )
        # Décoder : partager poids avec l'embedding
        self.decoder = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, sequence_output):
        # sequence_output: (batch, seq_len, d_model)
        x = self.transform(sequence_output)               # (batch, seq_len, d_model)
        logits = self.decoder(x) + self.bias             # (batch, seq_len, vocab_size)
        return logits

class BERTForMLMPretraining(torch.nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        super().__init__()
        # Backbone BERT
        self.bert = BERT(vocab_size, d_model, n_layers, heads, dropout)
        # Tête MLM
        self.mlm_head = BERTMLMHead(d_model, vocab_size)

    def forward(self, input_ids, token_type_ids=None):
        # Encodage
        sequence_output = self.bert(input_ids, token_type_ids)  # (batch, seq_len, d_model)
        # Prédiction MLM
        logits_mlm = self.mlm_head(sequence_output)            # (batch, seq_len, vocab_size)
        return logits_mlm