import math
import numbers
from loguru import logger
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from backbone.base_backbone import Backbone


class RnnBackbone(Backbone):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0, bidirectional: bool = False,
                 **kwargs) -> None:
        factory_kwargs = {'device': kwargs.pop(
            'device', 'cpu'), 'dtype': kwargs.pop('dtype', torch.float)}
        super().__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # for layer in range(num_layers):
        #     for direction in range(num_directions):
        #         layer_input_size = input_size if layer == 0 else hidden_size * num_directions

        #         w_ih = Parameter(torch.empty(
        #             (hidden_size, layer_input_size), **factory_kwargs))
        #         w_hh = Parameter(torch.empty(
        #             (hidden_size, hidden_size), **factory_kwargs))
        #         b_ih = Parameter(torch.empty(hidden_size, **factory_kwargs))
        #         b_hh = Parameter(torch.empty(hidden_size, **factory_kwargs))

        #         suffix = '_reverse' if direction == 1 else ''
        #         param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
        #         param_names = [x.format(layer, suffix) for x in param_names]

        #         for name, param in zip(param_names, (w_ih, w_hh, b_ih, b_hh)):
        #             setattr(self, name, param)
        self.i2h = nn.Linear(input_size + hidden_size,
                             hidden_size, **factory_kwargs)
        self.i2o = nn.Linear(input_size + hidden_size,
                             output_size, **factory_kwargs)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        num_directions = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * num_directions, self.hidden_size)

    def forward(self, input: Tensor, hidden: Tensor):
        orig_input = input

        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
