import torch
from torch import nn, Tensor

from .embedding.embedding import Embedding
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork


class DecoderLayer(nn.Module):
    """DecoderLayer
    
    3.1 Decoder
    """
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int, batch_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, ffn_hidden)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=dropout)
        
    def forward(self, _input: Tensor, encoder_input: Tensor,
                look_ahead_mask: Tensor, padding_mask: Tensor) -> Tensor:
        """Implements of Decoder layer
        Input -> Multi-Head Attenton(
                    Split -> Scaled dot product attention -> Concat) -> Residual connection & Normalization
        -> Encoder & Multi-Head Attenton(
                    Split -> Scaled dot product attention -> Concat) -> Res & Norm
        -> Feed Forward -> Res & Norm -> Output
    

        Args:
            _input (Tensor): positional encoded tensor
            look_ahead_mask (Tensor): look ahead mask
            padding_mask (Tensor): padding mask

        Returns:
            Tensor: output
        """
        orig_input = _input
        
        # query = key = value = _input
        output = self.attention1(_input, _input, _input, mask = look_ahead_mask)
        output = self.dropout1(output)
        # Residual connection and layer normalization
        output = self.norm1(output + orig_input)
        output1 = output
        
        output = self.attention2(_input, encoder_input, encoder_input, mask = padding_mask)
        output = self.dropout2(output + output1)
        output2 = output
        
        output = self.ffn.forward(output)
        output = self.dropout3(output)
        # Residual connection and Layer normalization
        output = self.norm3(output + output2)        
        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int,
                 num_heads: int, ffn_hidden: int, num_layers: int, batch_size: int,
                 dropout: float = 0.1):
        """Decoder

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
            [DecoderLayer(d_model=d_model,
                          num_heads=num_heads,
                          ffn_hidden=ffn_hidden,
                          batch_size=batch_size,
                          dropout=dropout)
                for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, token: Tensor, encoder_input: Tensor,
                look_ahead_mask: Tensor, padding_mask: Tensor):
        output = self.embedding(token)

        for layer in self.layers:
            output = layer(output, encoder_input, look_ahead_mask, padding_mask)

        return self.linear(output)
