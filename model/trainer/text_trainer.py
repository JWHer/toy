import os
import string
import torch
import unicodedata
from loguru import logger
from torch.utils.data.dataset import Dataset

from trainer.base_trainer import Trainer


class TextTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def train_one_iter(self):
        pass

def unicode2ascii(s):
    all_letters = string.ascii_letters + ".,;'"
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in all_letters)

def ascii2tensor(c):
    '''One-Hot 벡터'''
    all_letters = string.ascii_letters + ".,;'"
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][all_letters.find(c)] = 1
    return tensor

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]

def load_data(directory):
    category_lines = {}
    all_categories = []

    for filename in os.listdir(directory):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(os.path.join(directory, filename))
        category_lines[category] = lines
    return category_lines
