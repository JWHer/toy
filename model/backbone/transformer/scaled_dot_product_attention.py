import math
import torch
from torch import nn, Tensor

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Ref: https://github.com/hyunwoongko/transformer

    3.2.1 formular (1):
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_key)) @ V
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        # d_model(d_key) -> Multi-head attention -> num_heads * depth
        batch_size, num_heads, max_len, depth = key.size()
        
        key_t = key.transpose(2, 3)
        # Paper: {10000 * 64 @ 64 * 10000} x 8 heads as parellel
        score = (query @ key_t) / math.sqrt(depth)
        
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
            
        score = self.softmax(score)
        value = score @ value
        
        return value, score
