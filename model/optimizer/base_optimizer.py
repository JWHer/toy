from torch.optim import Optimizer
from abc import ABC, abstractmethod
from loguru import logger

class OptimizerFactory:
    _builders = {}

    @classmethod
    def register_optimizer(cls, name:str, builder):
        OptimizerFactory._builders[name] = builder

    @classmethod
    def get_optimizer(cls, name:str, **kwargs) -> Optimizer:
        builder = cls._builders.get(name)
        if not builder: raise ValueError(f"No such trainer[{name}]")
        return builder(**kwargs)
