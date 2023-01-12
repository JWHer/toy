import torch
from torch import nn, Tensor

from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_network import PositionWiseFeedForwardNetwork
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """EncoderLayer
    Ref: https://github.com/hyunwoongko/transformer
    
    3.1 Encoder
    """
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int, dropout: float = 0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(batch_size, max_len, d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, ffn_hidden)
        self.norm2 = nn.LayerNorm(batch_size, max_len, d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        
    def forward(self, _input: Tensor, mask: Tensor):
        orig_input = _input
        
        # query = key = value = _input
        output = self.attention(_input, _input, _input, mask = mask)
        output = self.dropout1(output)
        # Residual connection and layer normailation
        output = self.norm1(output + orig_input)
        output1 = output
        output = self.ffn.forward(output)

        output = self.dropout2(output)
        # Residual connection and Layer normailation
        output = self.norm2(output + output1)        
        return output


# wip
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
