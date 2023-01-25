import os
import string
import math
import time
from datetime import timedelta
from loguru import logger
from typing import Tuple

import torch
import torch.nn.functional as F
from nltk.translate import bleu_score
from torch import Tensor, optim
from torch.utils.data.dataset import Dataset
from torchtext.datasets import WikiText2, AG_NEWS, SQuAD2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from backbone.transformer import Transformer
from trainer.base_trainer import Trainer


class TransformerTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.best_loss = 0
        
    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()
        
    def before_train(self):
        self.train_dataloader = self._get_dataloader('train')
        self.test_dataloader = self._get_dataloader('dev')
        
        train_vocab_size = self.train_dataloader.dataset.vocab_size()
        batch_size = self.train_dataloader.batch_size
        
        self.model_cfg = self.cfg.pop('model')
        backbone_cfg:dict = self.model_cfg.pop('backbone', None)
        if backbone_cfg is None: raise AttributeError('Backbone config was not provided')
        backbone_name = backbone_cfg.pop('name', None)
        self.model = Transformer(vocab_size=train_vocab_size,
                                 batch_size=batch_size,
                                 **backbone_cfg)
        logger.info(f'The model has {self.model.num_parameters()} trainable parameters.')
        
        # if torch.backends.mps.is_available():
        #     logger.info(f'Set backends to MPS.')
        #     self.model.to('mps')
        self.optimizer = self._get_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self.cfg['scheduler'])
        
    def train_in_epoch(self):
        self.start_time = time.time()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_one_epoch()
            self.after_epoch()
        self.end_time = time.time()
    
    def before_epoch(self):
        logger.info(f"---> start train epoch{self.epoch}")
        self.model.train()
        
    def train_one_epoch(self):
        running_loss = 0.0

        for idx, (source, target) in enumerate(self.train_dataloader):
            # if torch.backends.mps.is_available():
            #     source = source.to('mps')
            #     target = target.to('mps')
            
            # Backward
            self.optimizer.zero_grad()
            output = self.model(source, target)
            output = output.transpose(2,1)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            logger.debug(
                f'Train Batch: [{(idx+1)*self.train_dataloader.batch_size}/{len(self.train_dataloader.dataset)}]\
                \tLoss: {loss:.6f}'
            )
        batch_iter_num = len(self.train_dataloader)
        self.train_loss = running_loss/batch_iter_num
        logger.info(
            f'Train Epoch: [{self.epoch+1}/{self.max_epoch}]\tLoss: {self.train_loss:.6f}')
        
    def evaluate(self):
        """Evaluate
        Ref: https://github.com/hyunwoongko/transformer/blob/master/train.py#L78
        """
        logger.info("Start Evaluation")
        
        self.model.eval()
        test_loss = 0
        bleus = []
        with torch.no_grad():
            for idx, (source, target) in enumerate(self.test_dataloader):
                # if torch.backends.mps.is_available():
                #     source = source.to('mps')
                #     target = target.to('mps')
                output = self.model(source, target)
                output = output.transpose(2,1)
                loss = F.cross_entropy(output, target)
                test_loss += loss.item()

                batch_bleus = []
                outputs = output.max(dim=1)[1]
                for batch_idx in range(source.size(0)):
                    source_words = self.test_dataloader.dataset.tensor2str(source[batch_idx,:].tolist())
                    target_words = self.test_dataloader.dataset.tensor2str(target[batch_idx,:].tolist())
                    output_words = self.test_dataloader.dataset.tensor2str(outputs[batch_idx,:].tolist())
                    logger.debug(f"Question: {' '.join(source_words)}\n\Right Answer: {' '.join(target_words)}\n\Model Answer: {' '.join(output_words)}")
                    if len(target_words) == 0: continue
                    batch_bleu = bleu_score.sentence_bleu(target_words, output_words)
                    batch_bleus.append(batch_bleu)
                bleu = sum(batch_bleus) / len(batch_bleus)
                bleus.append(bleu)

        test_loss /= len(self.train_dataloader)
        self.test_loss = test_loss
        logger.info(f'Test set loss: [{test_loss:.4f}]')
        
        total_blue = sum(bleus) / len(bleus)
        self.bleu = total_blue
        test_metric = {
            'loss': test_loss,
            'bleu': total_blue
        }
        logger.info('Evaluation scores')
        for key, value in test_metrics.items():
            logger.info(f"{key:<20}{value:.6f}")
            if 'tensorboard' in self.cfg:
                self.tensorboard.add_scalar(f'val/{key}', value, self.epoch)
        return test_metric

    def after_epoch(self):
        metrics = self.evaluate()
        is_best = metrics['loss'] < self.best_loss
        if is_best:
            self.best_loss = metrics['loss']
            logger.info('Best ckpt')
        self.save_ckpt(is_best)
    
    def after_train(self):
        logger.info(f'Epoch: {self.epoch + 1} | Time: {timedelta(seconds=self.end_time-self.start_time)}')
        logger.info(f'\tTrain Loss: {self.train_loss:.3f} | Train PPL: {math.exp(self.train_loss):7.3f}')
        logger.info(f'\tVal Loss: {self.test_loss:.3f} |  Val PPL: {math.exp(self.test_loss):7.3f}')
        logger.info(f'\tBLEU Score: {self.bleu:.3f}')
        if 'tensorboard' in self.cfg:
            self.tensorboard.add_scalar('map', self.map, self.epoch)        

    def load_dataset(self, name: str, data_cfg: dict):
        dataset = SQuADataset(name=name)
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
        self.data = list(filter(lambda t: t.numel() > 0, data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        if idx+1 >= len(self.data): return self.data[idx], torch.tensor(0)
        return self.data[idx], self.data[idx+1]
    
    def tensor2str(self, tensor_list: list):
        return self.vocab.lookup_tokens(tensor_list)
    
    def vocab_size(self):
        return len(self.vocab)


class NewsDataset(Dataset):
    pass


class SQuADataset(Dataset):
    def __init__(self, name='train'):
        CONTEXT_IDX = 0
        QUESTION_IDX = 1
        ANSWER_IDX = 2
        LINE_NUM_IDX = 3
        
        train_iter = SQuAD2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        make_question_token = lambda tup: self.tokenizer(f"<cls> {tup[QUESTION_IDX]} <sep> {tup[CONTEXT_IDX]} <sep>")
        self.vocab = build_vocab_from_iterator(
            map(make_question_token, train_iter), min_freq=3, specials=['<pad>', '<unk>', '<cls>', '<sep>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        self.text_iter = SQuAD2(split=name)
        
        pad = self.vocab.lookup_indices(['<pad>'])[0]
        # FIXME
        max_len = 1000
        def padding(tensor):
            t_len = tensor.size(0)
            if t_len > max_len:
                return tensor[:max_len]
            else:
                return F.pad(tensor, pad=(0, max_len-t_len), value=pad)
        self.data = [
            (
                padding(torch.tensor(self.vocab(make_question_token(item)), dtype=torch.long) ),
                padding(torch.tensor(self.vocab(self.tokenizer(item[ANSWER_IDX][0])), dtype=torch.long)),
            ) for item in self.text_iter ]
        self.data = self.data[:len(self.data)//100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.data[idx]
    
    def tensor2str(self, tensor_list: list):
        # FIXME dinamically remove special tokens
        tensor_list = list(filter(lambda token: token>3, tensor_list))
        return self.vocab.lookup_tokens(tensor_list)
    
    def vocab_size(self):
        return len(self.vocab)    


# import json
# from torchdata.datapipes.iter import IterDataPipe
# class _ParseSQuADQAData(IterDataPipe):
#     """
#     Ref: https://github.com/pytorch/data/blob/main/examples/text/squad2.py
#     """
#     def __init__(self, json_dict) -> None:
#         self.json_dict = json_dict

#     def __iter__(self):
#         raw_json_data = self.json_dict["data"]
#         for layer1 in raw_json_data:
#             for layer2 in layer1["paragraphs"]:
#                 for layer3 in layer2["qas"]:
#                     _context, _question = layer2["context"], layer3["question"]
#                     _answers = [item["text"] for item in layer3["answers"]]
#                     _answer_start = [item["answer_start"] for item in layer3["answers"]]
#                     if len(_answers) == 0:
#                         _answers = [""]
#                         _answer_start = [-1]
#                     yield (_context, _question, _answers, _answer_start)
