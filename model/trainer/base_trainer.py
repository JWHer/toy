import os, torch, shutil, copy
import torch.utils.data
from abc import ABC, abstractmethod
from loguru import logger
from thop import profile

from optimizer.base_optimizer import OptimizerFactory

class Trainer(ABC):
    DEFAULT_CONFIG = {
        'lr_cfg': None,
        'start_epoch': 0,
        'best_ap': 0
    }

    def __init__(self, **kwargs) -> None:
        """Trainer base class"""
        self.cfg = kwargs.copy()
        for key, value in kwargs.items():
            if value is None: continue
            setattr(self, key, value)
        self._set_defaults()
        self.model:torch.nn.Module = None
        
    def _set_defaults(self):
        for key, value in Trainer.DEFAULT_CONFIG.items():
            if not hasattr(self, key) and value is not None: setattr(self, key, value)

    def _get_dataloader(self, name='train'):
        data_cfg = self.dataset.copy()
        if name not in data_cfg: raise AttributeError(f'Dataset[{name}] was not provided')
        dataset_cfg = data_cfg.pop(name, False)
        dataset = self.load_dataset(name, dataset_cfg)

        for user_key in ['root_dir', 'annotation', 'compression']:
            if user_key in dataset_cfg: del dataset_cfg[user_key]
        return torch.utils.data.DataLoader(dataset, **dataset_cfg)

    def _get_optimizer(self):
        # self.start_lr = ""
        optimizer_cfg = self.optimizer.copy()
        optimizer_name = optimizer_cfg.pop('name', None)
        if optimizer_name is None: raise AttributeError('Optimizer name was not provided')
        optimizer_cfg['params'] = self.model.parameters()

        return OptimizerFactory.get_optimizer(optimizer_name, **optimizer_cfg)

    @abstractmethod
    def load_dataset(self, name:str, data_cfg:dict):
        images = []
        if 'compression' in data_cfg:
            #unzip
            pass
        else:
            # load images
            # images = [read(image) for image in os.listdir(data_cfg['root_dir'])]
            pass
        labels=[]
        # labels = json.load()
        
        return (
            images,
            labels
        )

    @abstractmethod
    def before_train(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def after_train(self):
        pass

    def save_ckpt(self, is_best=False):
        save_dir = self.log_dir
        logger.info(f"Save weights to {save_dir}")
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"epoch{self.epoch+1}_ckpt.pth")
        torch.save(ckpt_state, filename)
        if is_best:
            best_filename = os.path.join(save_dir, "best_ckpt.pth")
            shutil.copyfile(filename, best_filename)

    def get_model_info(self):
        # tsize = (self.data['img_size'], self.data['img_size'])
        tsize = (28, 28)
        stride = tsize[0]
        img = torch.zeros((1, 3, stride, stride), device=next(self.model.parameters()).device)
        flops, params = profile(copy.deepcopy(self.model), inputs=(img, ), verbose=False)
        params /= 1e6
        flops /= 1e9
        flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
        info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
        return info

    def show_data(self, name='train', idx=None, save=False):
        import random, numpy as np
        from matplotlib import pyplot as plt
        dataset = getattr(self, f"{name}_dataloader").dataset

        if idx is None:
            idx = random.randint(0,len(dataset))
        img, label = dataset[idx]
        title = f"{name}_dataset_idx-{idx}_label-{label.item()}"
        img = np.asarray(img).squeeze()

        plt.title(title)
        plt.imshow(img)
        if save: plt.savefig(f'{self.log_dir}/{title}.png')
        else: plt.show()

class TrainerFactory:
    _builders = {}

    @classmethod
    def register_trainer(cls, name:str, builder):
        TrainerFactory._builders[name] = builder

    @classmethod
    def get_trainer(cls, name:str, **kwargs) -> Trainer:
        builder = cls._builders.get(name)
        if not builder: raise ValueError(f"No such trainer[{name}]")
        return builder(**kwargs)

# Ref. decorator Timm
# def register_model(fn):
#     # lookup containing module
#     mod = sys.modules[fn.__module__]
#     module_name_split = fn.__module__.split('.')
#     module_name = module_name_split[-1] if len(module_name_split) else ''

#     # add model to __all__ in module
#     model_name = fn.__name__
#     if hasattr(mod, '__all__'):
#         mod.__all__.append(model_name)
#     else:
#         mod.__all__ = [model_name]

#     # add entries to registry dict/sets
#     _model_entrypoints[model_name] = fn
#     _model_to_module[model_name] = module_name
#     _module_to_models[module_name].add(model_name)
#     has_pretrained = False  # check if model has a pretrained url to allow filtering on this
#     if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
#         # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
#         # entrypoints or non-matching combos
#         has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
#         _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])
#     if has_pretrained:
#         _model_has_pretrained.add(model_name)
#     return fn
