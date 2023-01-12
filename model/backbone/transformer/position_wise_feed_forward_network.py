import torch
from torch import nn, Tensor

class PositionWiseFeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Networks
    Each of the layers in encoder and decoder contains FFN
    which applied to each position separately and identically.
    
    While FFN are the same across different positions,
    they use different parameters from layer to layer.
    
    3.3 formular (2):
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1):
        """Position-wise Feed-Forward Networks

        Args:
            d_model (int): 512
            hidden (int): 2048
            dropout (float, optional): Defaults to 0.1.
        """
        super().__init__()
        self.weight1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.weight2 = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor):
        x = self.weight1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight2(x)
        return x
