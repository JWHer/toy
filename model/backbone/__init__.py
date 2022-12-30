from .base_backbone import BackboneFactory
from .mnist_backbone import MnistBackbone
from .rnn_backbone import RnnBackbone

BackboneFactory.register_backbone('mnist', MnistBackbone)
BackboneFactory.register_backbone('rnn', RnnBackbone)