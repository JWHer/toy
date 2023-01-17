import torch
from torch import nn, Tensor

from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, batch_size: int, d_model: int = 512,
                 max_len: int = 5000, num_heads: int = 8, ffn_hidden: int = 2024, num_layers: int = 6):
        super().__init__()

        self.encoder = Encoder(vocab_size, d_model, max_len,
                               num_heads, ffn_hidden, num_layers, batch_size)
        self.decoder = Decoder(vocab_size, d_model, max_len,
                               num_heads, ffn_hidden, num_layers, batch_size)

    def forward(self, sorce, target):
        source_mask = self.make_mask()

        look_ahead_mask = self.make_mask(target, )

        encoder_output = self.encoder(source, padding_mask)
        output = self.decoder(target, encoder_output,
                              look_ahead_mask, padding_mask)
        return output

    def padding_mask(self, _input: Tensor) -> Tensor:
        """Padding Mask
        Mask zero(<pad>) token

        Args:
            _input (Tensor): token list

        Returns:
            Tensor: Padding Mask
        """
        mask = torch.zeros(_input.size())
        return mask.masked_fill(_input == 0, 1)

    def look_ahead_mask(self, _input: Tensor):
        sequence_len = _input.size()[-1]
        mask = torch.ones(sequence_len, sequence_len).triu(diagonal=1)
        return torch.maximum(mask, self.padding_mask(_input))
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
