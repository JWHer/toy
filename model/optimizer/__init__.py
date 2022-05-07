import torch.optim as optim
from .base_optimizer import OptimizerFactory

OptimizerFactory.register_optimizer('SGD', optim.SGD)
OptimizerFactory.register_optimizer('Adam', optim.Adam)
