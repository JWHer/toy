import os
import string
import time
from loguru import logger

import torch
from torch import Tensor, optim
from torch.utils.data.dataset import Dataset
from torchtext.datasets import WikiText2, AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from backbone.transformer import Transformer
from trainer.base_trainer import Trainer


class TransformerTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()
        
    def before_train(self):
        # TODO must provide vocab_size
        self.model = Transformer(**self.cfg)
        logger.info(f'The model has {self.model.count_parameters} trainable parameters.')
        if torch.backends.mps.is_available():
            self.model.to('mps')
        self.train_dataloader = self._get_dataloader('train')
        self.test_dataloader = self._get_dataloader('test')
        self.optimizer = self._get_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self.cfg['scheduler'])
        self.start_time = time.time()
        
    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_one_epoch()
            self.after_epoch()
    
    def before_epoch(self):
        logger.info(f"---> start train epoch{self.epoch}")
        self.model.train()
        
    def train_one_epoch(self):
        running_loss = 0.0

        for i, batch in enumerate(self.train_dataloader):
            source = batch.source
            target = batch.target
            if torch.backends.mps.is_available():
                pass
            
            # Backward
            self.optimizer.zero_grad()
            output = model(source, target)
            loss
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            logger.debug(
                f'Train Batch: [{(idx+1)*len(_input)}/{len(self.train_dataloader.dataset)}]\
                \tLoss: {loss:.6f}'
            )
        logger.info()
        
    def evaluate(self):
        # TODO
        pass

    def after_epoch(self):
        pass
    
    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100)
        )
        if 'tensorboard' in self.cfg:
            self.tensorboard.add_scalar('map', self.map, self.epoch)        

    def load_dataset(self, name: str, data_cfg: dict):
        dataset = WikiDataset(name=name)
        return dataset

class WikiDataset(Dataset):
    """Wiki Dataset
    Ref: https://tutorials.pytorch.kr/beginner/transformer_tutorial.html#id2
    """
    def __init__(self, name='train'):
        train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        self.text_iter = WikiText2(split=name)
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in self.text_iter]
        self.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def tensor2str(self, tensor: Tensor):
        return self.vocab.lookup_tokens(tensor)


class NewsDataset(Dataset):
    pass
