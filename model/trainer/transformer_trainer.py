import os
import string
import time
from loguru import logger

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchtext.datasets import WikiText2, AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from trainer.base_trainer import Trainer


class TransformerTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class WikiDataset(Dataset):
    """Wiki Dataset
    Ref: https://tutorials.pytorch.kr/beginner/transformer_tutorial.html#id2
    """
    def __init__(self, name='train'):
        train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        self.text_iter = WikiText2(split=name)
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in self.text_iter]
        self.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def tensor2str(self, tensor: Tensor):
        return self.vocab.lookup_tokens(tensor)


class NewsDataset(Dataset):
    pass
