import math
import torch
import torch.nn as nn


class  InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Embedding layer provided by pytorch - maps numbers to the same vector every time.
        # - We always map the same word to the same embedding.
        # - However the values in the vector aren't fixed - they're learned by the model.
        self.embedding = nn.Embedding(vocab_size, d_model) 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # To prevent overfitting

        # Note: We're using a slightly modified formule compared to what we have seen in the paper and presentation
        #       using log space - this is for numerical stability, but the result should be the same.

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0:2] = torch.sin(position * div_term)
        # Apply she cos to odd positions
        pe[:, 1:2] = torch.cos(position * div_term)
        # We need to add a batch dimension
        pe = pe.unsqueeze(0) # -> (1, seq_len, d_model)
        # If you have a tensor that you want to keep inside the module not as a (learned) parameter,
        # but you want it to be saved when you save the file of the model you should register it as a buffer.
        # This way the tensor will be saved as in the file along with the state of the model.
        self.register_buffer('pe', pe)


    def forward(self, x):
        # We need to add the positional encoding to every word inside the sentence.
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # We also tell the model that we don't want to learn this positional encoding
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative # nn.Parameter makes this parameter learnable
        self.bias = nn.Parameter(torch.zeros(1)) # Additive

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        # Make sure that the d_model is divisible by h
        assert d_model % h == 0
        self.d_k = d_model // h

        # Prepare matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model) # In slides the shape is (h * d_v, d_model), but (d_v=d_k)*h = d_model

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # @ is matrix multiplication in the Pytorch, Transpose last two dimensions
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores   # first value for the model, second value is for visualisation

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) -->  (batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model) == (batch, seq_len, h*d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x, self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
