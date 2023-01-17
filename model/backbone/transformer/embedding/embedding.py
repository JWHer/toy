import math
import torch
from torch import nn, Tensor

from .positional_encoding import PositionalEncoding


class Embedding(nn.Module):
    """Embedding

    3.4
    Share the same weight matrix between the two embedding layers
    and the pre-softmax linear transformation
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=1)
        self.positional_embedding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, token: Tensor):
        output = self.token_embedding(token)
        output *= math.sqrt(self.d_model)
        output = self.positional_embedding(output)
        return self.dropout(output)
