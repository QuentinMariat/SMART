import torch

class MultiHeadedAttention(torch.nn.Module):
  def __init__(self, d_model, heads, dropout=0.1):
    super(MultiHeadedAttention, self).__init__()
    assert d_model % heads == 0, "d_model must be divisible by heads"
    self.d_k = d_model // heads
    self.heads = heads
    self.dropout = torch.nn.Dropout(dropout)

    self.query = torch.nn.Linear(d_model, d_model) # On utilise ça pour l'instant, maybe on remplace plus tard
    self.key = torch.nn.Linear(d_model, d_model)
    self.value = torch.nn.Linear(d_model, d_model)
    self.output_linear = torch.nn.Linear(d_model, d_model)

  def forward(self, query, key, value, mask=None):
    

    query = self.query(query)
    key = self.key(key)
    value = self.value(value)

    query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3) 
    key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3)
    value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0,2,1,3)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5) # nevoen


    class FeedForward(torch.nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super(FeedForward, self).__init__()
            self.linear1 = torch.nn.Linear(d_model, d_ff)
            self.dropout = torch.nn.Dropout(dropout)
            self.linear2 = torch.nn.Linear(d_ff, d_model)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.functional.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x
        
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)  # Skip connection
        x = self.layer_norm1(x)  # Layer normalization

        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)  # Skip connection
        x = self.layer_norm2(x)  # Layer normalization

        return x