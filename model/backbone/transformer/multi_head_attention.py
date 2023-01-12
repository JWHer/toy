import math
import torch
from torch import nn, Tensor

from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention
    Ref: https://wikidocs.net/31379
    
    3.2.2 formular:
        MultiHead(Q, K, V) = Concat(head_1, ... , head_h) @ W^O
            where head_i = Attention(QW^Q, KW^K, VW^V)
    """
    def __init__(self, d_model: int, num_heads: int, d_key: int = None, d_value: int = None):
        """Multi-Head Attention
        
        Args:
            d_model (int): 512
            num_heads (int): 8
            d_key (int, optional): _description_. Defaults to None.
            d_value (int, optional): _description_. Defaults to None.
        """
        # TODO add bias
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_key = d_key if d_key is not None else d_model
        self.d_value = d_value if d_value is not None else d_value
        self.num_heads = num_heads
        # Paper: d_key = d_value = d_model / h = 64
        self.depth = d_model // num_heads
        
        self.attention = ScaledDotProductAttention()
        self.weight_query = nn.Linear(d_model, d_model)
        self.weight_key = nn.Linear(d_model, self.d_key)
        self.weight_value = nn.Linear(d_model, self.d_value)
        self.weight_out = nn.Linear(d_model, d_model)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        query = self.weight_query(query)
        key = self.weight_key(query)
        value = self.weight_value(value)
        
        query = self.split(query)
        key = self.split(key)
        value = self.split(value)
        
        output, attention = self.attention(query, key, value, mask)
        
        output = self.concat(output)
        output = self.weight_out(output)
        
        return output
        
    def split(self, tensor: Tensor):
        # batch_size * max_len * d_model = size
        batch_size, max_len, d_model = tensor.size()
        
        # d_model -> split -> num_heads * depth
        tensor = tensor.view(batch_size, max_len, self.num_heads, self.depth).transpose(1, 2)
        return tensor 

    def concat(self, tensor: Tensor):
        # batch_size * num_heads * max_len * depth = size
        batch_size, num_heads, max_len, depth = tensor.size()
        
        # num_heads * depth -> concat -> d_model
        tensor = tensor.transpose(1,2).contiguos().view(batch_size, max_len, self.d_model)
        return tensor
