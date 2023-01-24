import numpy
import pytest
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from backbone.transformer import PositionalEncoding, MultiHeadAttention,\
    PositionWiseFeedForwardNetwork, ScaledDotProductAttention, Transformer

class TestTransformer:
    d_model = 512
    max_len = 50
    
    def test_pos_encoding_plt(self):
        pos_encoding = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)
        pos_encode = pos_encoding.forward(torch.zeros(self.max_len, 1, self.d_model)).permute(1, 0, 2)[0]
        plt.pcolormesh(pos_encode, cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, 512))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()
        
    def test_scaled_dot_product(self):
        scaled_dot_product = ScaledDotProductAttention()
        query = Tensor([[0, 10, 0]])    # (1, 3)
        # TODO [[0, 0, 10], [0, 10, 0], [10, 10, 0]]
        key = Tensor([[10,0,0],         # (4, 3)
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]])
        value = Tensor([[   1,0],       # (4, 2)
                      [  10,0],
                      [ 100,5],
                      [1000,6]])
        
        # add batch, num_heads
        query = query[None,None,:]
        key = key[None,None,:]
        value = value[None,None,:]
        
        # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_key)) @ V
        output, attention = scaled_dot_product(query, key, value)
        
        # remove batch, num_heads
        output = output[0,0,:]
        attention = attention[0,0,:]
        
        # Attention = (1, 3) @ (3, 4) = (1, 4)
        assert torch.allclose(attention, Tensor([[0, 1, 0, 0]]))
        # Output = softmax(1, 4) @ (4, 2) = (1, 2)
        assert torch.allclose(output, Tensor([[10, 0]]))
    
    def test_scaled_dot_product_with_mask(self):
        scaled_dot_product = ScaledDotProductAttention()
        query = Tensor([[0, 10, 0]])    # (1, 3)
        key = Tensor([[10,0,0],         # (4, 3)
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]])
        value = Tensor([[   1,0],       # (4, 2)
                      [  10,0],
                      [ 100,5],
                      [1000,6]])
        mask = Tensor([[1, 0, 1, 0]])   # (1, 4)
        
        query = query[None,None,:]
        key = key[None,None,:]
        value = value[None,None,:]
        mask = mask[None,None,:]
        
        output, attention = scaled_dot_product(query, key, value, mask=mask)
        
        output = output[0,0,:]
        attention = attention[0,0,:]
        
        # [0, 1, 0, 0] -> mask [1, 0, 1, 0] (second and forth) -> [0, -inf, 0, -inf]
        # [0, -inf, 0, -inf] -> softmax -> [0.5, 0, 0.5, 0]
        assert torch.allclose(attention, Tensor([[0.5, 0, 0.5, 0]]))
        assert torch.allclose(output, Tensor([[50.5, 2.5]]))

    def test_padding_mask(self):
        transformer = Transformer(vocab_size=10000, batch_size=1)
        _input = Tensor([[1, 21, 777, 0, 0]])
        mask = transformer.padding_mask(_input)
        
        # mask zero(<pad>) token
        assert torch.equal(mask, Tensor([[0, 0, 0, 1, 1]]))

    def test_look_ahead_mask(self):
        transformer = Transformer(vocab_size=10000, batch_size=1)
        _input = Tensor([[1, 2, 0, 4, 5]])
        mask = transformer.look_ahead_mask(_input)
        
        # mask zero(<pad>) token
        assert torch.equal(mask, Tensor([[[0, 1, 1, 1, 1],
                                         [0, 0, 1, 1, 1],
                                         [0, 0, 1, 1, 1],
                                         [0, 0, 1, 0, 1],
                                         [0, 0, 1, 0, 0]]]))

    def test_num_parameters(self):
        transformer = Transformer(vocab_size=1000, batch_size=1)
        
        # Embedding = (512 * 5000) * 2
        # 1024000
        # Encoder = 6 * (512 * 512 * 3 + 2 * (512 * 2024))
        # 17154048
        # Decoder = 6 * (512 * 512 * 6 + 2 * (512 * 2024))
        # 21872640
        # Output = 512 * 1000
        # 512000
        # = 45380296
        assert transformer.num_parameters()
