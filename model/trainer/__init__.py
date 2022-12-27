from .base_trainer import TrainerFactory
from .mnist_trainer import MnistTrainer
from .text_trainer import TextTrainer

TrainerFactory.register_trainer('mnist', MnistTrainer)
