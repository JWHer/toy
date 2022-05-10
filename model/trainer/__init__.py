from .base_trainer import TrainerFactory
from .mnist_trainer import MnistTrainer

TrainerFactory.register_trainer('mnist', MnistTrainer)
