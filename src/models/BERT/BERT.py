import torch
from BERTEmbedding import BERTEmbedding
from AttentionLayer import EncoderLayer

class BERT(torch.nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param d_model: BERT model hidden size
        :param n_layers: numbers of Transformer blocks
        :param heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.feed_forward_hidden = d_model * 4

        # embedding: sum of token, positional, and segment embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model)

        # stack of Transformer encoder layers
        self.encoder_blocks = torch.nn.ModuleList([
            EncoderLayer(d_model, heads, self.feed_forward_hidden, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, segment_info=None):
        # attention mask for padded tokens
        # mask shape: (batch_size, 1, seq_len, seq_len)
        #mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask = (x > 0).unsqueeze(1).unsqueeze(2)

        # embedding lookup + add segment embeddings if provided
        x = self.embedding(x, segment_info)

        # pass through each Transformer layer
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
        return x


class MultiLabelEmotionClassifier(torch.nn.Module):
    """
    Multi-label emotion classifier using BERT's [CLS] token representation.
    """
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.activation = torch.nn.Sigmoid()

    def forward(self, bert_output):
        # bert_output: (batch_size, seq_len, hidden_size)
        cls_token = bert_output[:, 0]         # [CLS] token representation
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)   # (batch_size, num_labels)
        probs = self.activation(logits)       # probabilities in [0,1]
        return probs


class BERTForMultiLabelEmotion(torch.nn.Module):
    """
    BERT model with a multi-label emotion classification head.
    """
    def __init__(self, vocab_size, num_labels,
                 d_model=768, n_layers=12, heads=12, dropout=0.1):
        super().__init__()
        # Core BERT encoder
        self.bert = BERT(vocab_size, d_model, n_layers, heads, dropout)
        # Classification head
        self.classifier = MultiLabelEmotionClassifier(d_model, num_labels, dropout)

    def forward(self, input_ids, segment_info=None):
        # Encode inputs
        sequence_output = self.bert(input_ids, segment_info)
        # Predict multi-label emotions
        return self.classifier(sequence_output)
    
# Example usage:
# model = BERTForMultiLabelEmotion(vocab_size=30522, num_labels=NUM_LABELS)
# outputs = model(input_ids=batch_ids, segment_info=batch_segments)
# loss_fn = torch.nn.BCEWithLogitsLoss()  # si on veut utiliser les logits directement
# Ou torch.nn.BCELoss() si on utilise Sigmoid outputs
    
