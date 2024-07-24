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
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

