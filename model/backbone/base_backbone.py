import torch.nn as nn
from abc import ABC, abstractmethod
from loguru import logger

class Backbone(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def forward(self, x, *args):
        pass

class BackboneFactory:
    _builders = {}

    @classmethod
    def register_backbone(cls, name:str, builder):
        BackboneFactory._builders[name] = builder

    @classmethod
    def get_backbone(cls, name:str, **kwargs) -> Backbone:
        builder = cls._builders.get(name)
        if not builder: raise ValueError(f"No such backbone[{name}]")
        return builder(**kwargs)
