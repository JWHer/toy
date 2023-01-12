import os
import json
import pytest
import torch

from trainer.text_trainer import TextTrainer, NameDataset


class TestTextTrainer:
    text_trainer = TextTrainer(**{
        "model": {
            "backbone": {
                "name": "rnn",
                "input_size": 56,
                "hidden_size": 128,
                "output_size": 18
            }
        },
        "optimizer": {
            "name": "SGD",
            "lr": 0.01,
            "momentum": 0.5
        },
        "dataset": {
            "train": {
                "root_dir": "../data/names",
                "ratio": 0.7,
                "batch_size": 1,
                "shuffle": True
            },
            "test": {
                "root_dir": "../data/names",
                "ratio": 0.3,
                "batch_size": 1,
                "shuffle": True
            }
        },
        "max_epoch": 10,
        "classes": [
            "czech",
            "german",
            "arabic",
            "japanese",
            "chinese",
            "vietnamese",
            "russian",
            "french",
            "irish",
            "english",
            "spanish",
            "greek",
            "italian",
            "portuguese",
            "scottish",
            "dutch",
            "korean",
            "polish"
        ],
        "log_dir": "./result/name/"
    })

    def test_before_train(self):
        self.text_trainer.before_train()
        assert self.text_trainer.model
        assert next(self.text_trainer.model.parameters()).is_mps == (
            torch.backends.mps.is_available() and torch.backends.mps.is_built())
        assert self.text_trainer.train_dataloader
        assert self.text_trainer.test_dataloader
        assert self.text_trainer.optimizer

    @pytest.mark.skip()
    def test_train_dataloader(self):
        """This case is too long.
        So skip it if you don't need to validate your dataset.
        """
        batch_size = 1
        for idx, (line, answer, _input, category) in enumerate(self.text_trainer.train_dataloader):
            assert line is not None
            assert answer is not None
            assert _input is not None
            assert len(_input) == batch_size
            assert category is not None
        assert idx+1 == len(self.text_trainer.train_dataloader)

    def test_before_epoch(self):
        self.text_trainer.before_epoch()
        assert self.text_trainer.model.training

    def test_train_one_epoch(self):
        self.text_trainer.train_one_epoch()
        assert True


class TestNameDataset:
    name_dataset = NameDataset()

    def test_unicode2ascii(self):
        assert self.name_dataset.unicode2ascii('Ślusàrski') == 'Slusarski'

    def test_ascii2tensor(self):
        tensor = self.name_dataset.ascii2tensor('c')
        assert torch.sum(tensor) == 1

    def test_line2tensor(self):
        line = 'John'
        tensor = self.name_dataset.line2tensor(line)
        size = tensor.size()
        assert len(size) == 3
        assert size[0] == len(line)
        assert size[1] == 1
        assert size[2] == len(self.name_dataset.all_letters)

    def test_load_data(self):
        dirname = '/Users/jwher/dev/toy/data/names'
        categories = self.name_dataset.load_data(dirname)
        assert len(categories.keys()) == len(os.listdir(dirname))
