import torch
from models.bert.bert_embedding import BERTEmbedding
from models.bert.attention_layer import EncoderLayer
from transformers import AutoModel, AutoConfig


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

    def forward(self, input_ids, token_type_ids=None):
        """
        :param input_ids: Tensor (batch_size, seq_len)
        :param token_type_ids: Optional Tensor (batch_size, seq_len)
        """
        # Mask for padding tokens (1 where token exists, 0 for padding)
        mask = (input_ids > 0).unsqueeze(1).unsqueeze(2)

        # Embedding (supports optional segment embeddings)
        x = self.embedding(input_ids, token_type_ids)

        # Encoder layers
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
        # probs = self.activation(logits)       # probabilities in [0,1]
        return logits            # TODO passer à return logits pour BCEWithLogitsLoss



class BERTForMultiLabelEmotion(torch.nn.Module):
    def __init__(self, num_labels, vocab_size=None,
                 use_pretrained=False, pretrained_model_name=None,
                 d_model=768, n_layers=12, heads=12, dropout=0.1):
        super().__init__()

        self.use_pretrained = use_pretrained

        if use_pretrained and pretrained_model_name:
            print(f"🔄 Loading pretrained model: {pretrained_model_name}")
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
            print(f"✅ Successfully loaded pretrained model")
            print(f"Model config: {self.bert.config}")
            hidden_size = self.bert.config.hidden_size
            
            # Freeze only the first 2 layers for transfer learning
            modules = list(self.bert.modules())
            encoder_layers = [m for m in modules if isinstance(m, type(modules[-1]))]
            layers_to_freeze = min(2, len(encoder_layers))
            for param in list(self.bert.parameters())[:layers_to_freeze]:
                param.requires_grad = False
            print(f"💡 Froze first {layers_to_freeze} layers for transfer learning")
        else:
            print("⚠️ Using custom BERT model without pretraining")
            assert vocab_size is not None, "vocab_size must be provided if not using pretrained model"
            self.bert = BERT(vocab_size, d_model, n_layers, heads, dropout)
            hidden_size = d_model

        # Use MultiLabelEmotionClassifier instead of Sequential
        self.classifier = MultiLabelEmotionClassifier(hidden_size, num_labels, dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if self.use_pretrained:
            # Handle models that don't use token_type_ids (like RoBERTa)
            if 'roberta' in self.bert.config.model_type:
                outputs = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)
            else:
                outputs = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
            sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        else:
            sequence_output = self.bert(input_ids, token_type_ids)

        return self.classifier(sequence_output)

    
# Example usage:
# model = BERTForMultiLabelEmotion(vocab_size=30522, num_labels=NUM_LABELS)
# outputs = model(input_ids=batch_ids, segment_info=batch_segments)
# loss_fn = torch.nn.BCEWithLogitsLoss()  # si on veut utiliser les logits directement
# Ou torch.nn.BCELoss() si on utilise Sigmoid outputs

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
        self.vocab_size = vocab_size

    def forward(self, input_ids, token_type_ids=None):
        # Encodage
        sequence_output = self.bert(input_ids, token_type_ids)  # (batch, seq_len, d_model)
        # Prédiction MLM
        logits_mlm = self.mlm_head(sequence_output)            # (batch, seq_len, vocab_size)
        return logits_mlm
    
