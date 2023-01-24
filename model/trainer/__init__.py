from .base_trainer import TrainerFactory
from .mnist_trainer import MnistTrainer
from .text_trainer import TextTrainer
from .transformer_trainer import TransformerTrainer

TrainerFactory.register_trainer('mnist', MnistTrainer)
TrainerFactory.register_trainer('text', TextTrainer)
TrainerFactory.register_trainer('transformer', TransformerTrainer)
