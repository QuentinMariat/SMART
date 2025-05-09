class BERT(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings, type_vocab_size, dropout_prob):
        super(BERT, self).__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size)
        self.encoder = BertEncoder(hidden_size, num_hidden_layers, num_attention_heads, intermediate_size)
        self.pooler = BertPooler(hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask)
        pooled_output = self.pooler(encoder_output[-1])
        return pooled_output
    
