import os
import pytest
import torch

from trainer.text_trainer import *


def test_unicode2ascii():
    assert unicode2ascii('Ślusàrski') == 'Slusarski'

def test_load_data():
    dirname = '/Users/jwher/dev/toy/data/names'
    categories = load_data(dirname)
    assert len(categories.keys()) == len(os.listdir(dirname))

def test_ascii2tensor():
    tensor = ascii2tensor('c')
    assert torch.sum(tensor) == 1
