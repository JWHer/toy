import torch
from torch import nn, Tensor

from .embedding.embedding import Embedding
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork


class EncoderLayer(nn.Module):
    """EncoderLayer
    
    3.1 Encoder
    """
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int, batch_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        
    def forward(self, _input: Tensor, padding_mask: Tensor) -> Tensor:
        """Implements of encoder layer
        Input -> Multi-Head Attenton(
                    Split -> Scaled dot product attention -> Concat)
        -> Residual connection & Normalization -> Feed Forward -> Res & Norm
        -> Output
    

        Args:
            _input (Tensor): positional encoded tensor
            padding_mask (Tensor): padding mask

        Returns:
            Tensor: _description_
        """
        orig_input = _input
        
        # query = key = value = _input
        output = self.attention(_input, _input, _input, mask = padding_mask)
        output = self.dropout1(output)
        # Residual connection and layer normalization
        output = self.norm1(output + orig_input)
        output1 = output
        output = self.ffn.forward(output)

        output = self.dropout2(output)
        # Residual connection and Layer normalization
        output = self.norm2(output + output1)        
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int,
                 num_heads: int, ffn_hidden: int, num_layers: int, batch_size: int,
                 dropout: float = 0.1):
        """Encoder

        Args:
            vocab_size (int): _description_
            d_model (int): 512
            max_len (int): 5000
            num_heads (int): 8
            ffn_hidden (int): 2024
            num_layers (int): 6
            dropout (float, optional): Defaults to 0.1.
        """
        super().__init__()
        
        self.embedding = Embedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model,
                          num_heads=num_heads,
                          ffn_hidden=ffn_hidden,
                          batch_size=batch_size,
                          dropout=dropout)
                for _ in range(num_layers)])

    def forward(self, token: Tensor, padding_mask: Tensor):
        output = self.embedding(token)

        for layer in self.layers:
            output = layer(output, padding_mask)

        return output
