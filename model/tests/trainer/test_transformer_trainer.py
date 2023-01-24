import os
import json
import numpy
import pytest
import torch
from matplotlib import pyplot as plt

from trainer.transformer_trainer import TransformerTrainer, WikiDataset, SQuADataset


def transformer_trainer():
    with open('./configs/transformer.json') as config_file:
        config = json.load(config_file)
    return TransformerTrainer(**config)


@pytest.fixture
def trainer():
    yield transformer_trainer()


class TestTransformerTrainer:
    transformer_trainer = transformer_trainer()

    def test_before_train(self):
        self.transformer_trainer.before_train()
        assert self.transformer_trainer.model
        # assert next(self.transformer_trainer.model.parameters()).is_mps == (
        #     torch.backends.mps.is_available() and torch.backends.mps.is_built())
        assert self.transformer_trainer.train_dataloader
        assert self.transformer_trainer.test_dataloader
        assert self.transformer_trainer.optimizer

    @pytest.mark.skip()
    def test_train_dataloader(self, trainer):
        """This case is too long.
        So skip it if you don't need to validate your dataset.
        """
        dataset_name = 'train'
        train_dataloader = trainer._get_dataloader(dataset_name)
        batch_size = trainer.cfg['dataset'][dataset_name]['batch_size']

        total = len(train_dataloader.dataset)
        batch_iter_num = len(train_dataloader)
        assert total <= batch_iter_num * batch_size
        assert batch_iter_num * batch_size < total + batch_size

        for idx, (source, target) in enumerate(train_dataloader):
            assert source is not None
            assert target is not None
            assert torch.equal(source[1:], target[:-1])
            if idx == len(train_dataloader)-1:
                return
            assert source.size(0) == batch_size
            assert target.size(0) == batch_size
        assert idx+1 == len(train_dataloader)

    def test_before_epoch(self):
        """This test should be preceded by test_before_train"""
        self.transformer_trainer.before_epoch()
        assert self.transformer_trainer.model.training

    def test_train_one_epoch(self):
        self.transformer_trainer.train_one_epoch()
        assert True


class TestWikiDataset:
    wiki_dataset = WikiDataset()

    def test_wiki_dataset_load(self):
        source, target = self.wiki_dataset[0]
        assert source
        assert target

class TestSQuADataset:
    squad = SQuADataset()

    def test_squad_load(self):
        assert self.squad[0]
