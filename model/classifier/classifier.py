from abc import abstractmethod
import torch.nn as nn

from backbone.base_backbone import BackboneFactory

class Classifier(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.model_cfg = kwargs.pop('model')
        backbone_cfg:dict = self.model_cfg.pop('backbone', None)
        if backbone_cfg is None: raise AttributeError('Backbone config was not provided')
        backbone_name = backbone_cfg.pop('name', None)
        if not backbone_name: raise AttributeError('Backbone name was not provided')
        self.backbone = BackboneFactory.get_backbone(backbone_name, **backbone_cfg)

        neck_cfg:dict = self.model_cfg.pop('neck', None)
        if neck_cfg is not None:
            neck_name = neck_cfg.pop('name', None)
            # TODO NeckFactory
            raise NotImplementedError('Neck not implemented')

        head_cfg:dict = self.model_cfg.pop('head', None)
        if head_cfg is not None:
            head_name = head_cfg.pop('name', None)
            # TODO HeadFactory
            raise NotImplementedError('Head not implemented')

    @abstractmethod
    def forward(self, x, *args):
        features = self.backbone(x, *args)

        # if self.neck is not None:
        #     features = self.neck(features)[-1]
        
        # if self.head is not None:
        #     # assert 'targets' in kwargs
        #     # loss, acc = self.head(features, kwargs['targets'])
        #     # outputs = {"total_loss": loss, "train_accuracy": acc}
        #     outputs = self.head(features)
        return features
