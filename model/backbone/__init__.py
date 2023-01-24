from .base_backbone import BackboneFactory
from .mnist_backbone import MnistBackbone
from .rnn_backbone import RnnBackbone
from .transformer import Transformer

BackboneFactory.register_backbone('mnist', MnistBackbone)
BackboneFactory.register_backbone('rnn', RnnBackbone)
BackboneFactory.register_backbone('transformer', Transformer)