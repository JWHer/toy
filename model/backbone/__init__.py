from .base_backbone import BackboneFactory
from .mnist_backbone import MnistBackbone

BackboneFactory.register_backbone('mnist', MnistBackbone)