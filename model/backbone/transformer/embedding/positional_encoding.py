import math
import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """Positional Encoding
    Ref: https://tutorials.pytorch.kr/beginner/transformer_tutorial.html#id1
    
    3.5 formular:
        PE(pos,2i) = sin(pos/10000^(2i/d_model))
        PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # [0], [1], [2] ... [max_len]
        position = torch.arange(max_len).unsqueeze(1)
        # 1 / 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 각 글자 마다(max_len), 하나씩, d_model만큼 position
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to the input embeddings

        Args:
            x (Tensor): input embedding

        Returns:
            Tensor: x += pos_enc
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
