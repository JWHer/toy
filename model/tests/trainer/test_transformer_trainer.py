import numpy
import pytest
import torch
from matplotlib import pyplot as plt

from trainer.transformer_trainer import TransformerTrainer, WikiDataset

class TestTransformerTrainer:
    # transformer_trainer = 
    wiki_dataset = WikiDataset()
    
    def test_wiki_dataset_load(self):
        assert self.wiki_dataset[0]
    