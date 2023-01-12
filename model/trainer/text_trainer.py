import os
import string
import time
import unicodedata
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataset import Dataset

from classifier.classifier import Classifier
from trainer.base_trainer import Trainer


class TextTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def before_train(self):
        self.model = Classifier(**self.cfg)
        if torch.backends.mps.is_available():
            self.model.to('mps')
        self.train_dataloader = self._get_dataloader('train')
        self.test_dataloader = self._get_dataloader('test')
        self.optimizer = self._get_optimizer()
        self.start_time = time.time()

    def before_epoch(self):
        logger.info(f"---> start train epoch{self.epoch}")
        self.model.train()

    def train_one_epoch(self):
        running_loss = 0.0

        # No Batches
        for idx, (line, answer, _input, label) in enumerate(self.train_dataloader):
            line = line[0]
            answer = answer[0]
            _input = _input[0]
            label = label[0]
            hidden = self.model.backbone.init_hidden()
            if torch.backends.mps.is_available():
                _input = _input.to('mps')
                label = label.to('mps')
                hidden = hidden.to('mps')

            # Backward
            self.optimizer.zero_grad()
            for i, ascii_tensor in enumerate(_input):
                output, hidden = self.model(ascii_tensor, hidden)
            loss = F.nll_loss(output, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            logger.debug(
                f'Train Batch: [{(idx+1)*len(_input)}/{len(self.train_dataloader.dataset)}]\
                \tLoss: {loss:.6f}'
            )
            if idx % 100 == 0:
                guess, guess_i = self.get_class(output)
                correct = '✓' if guess_i == label[0] else '✗ (%s)' % answer
                print('%d %d%% (%s) %.4f %s / %s %s' % (idx, idx / len(self.train_dataloader.dataset)
                      * 100, self.time_scince(self.start_time), loss, line, guess, correct))
        # batch_iter_num = len(self.train_dataloader.dataset) / self.train_dataloader.batch_size\
        #     + (len(self.train_dataloader.dataset) % self.train_dataloader.batch_size > 0)
        batch_iter_num = len(self.train_dataloader.dataset)
        logger.info(
            f'Train Epoch: [{self.epoch+1}/{self.max_epoch}]\tLoss: {running_loss/batch_iter_num:.6f}')

    def evaluate(self):
        logger.info("Start Evaluation")

        self.model.eval()
        test_loss = 0
        test_metrics = {}
        outputs = []
        labels = []
        with torch.no_grad():
            for idx, (_, _, _input, label) in enumerate(self.test_dataloader):
                _input = _input[0]
                label = label[0]
                hidden = self.model.backbone.init_hidden()
                if torch.backends.mps.is_available():
                    _input = _input.to('mps')
                    label = label.to('mps')
                    hidden = hidden.to('mps')

                for i, ascii_tensor in enumerate(_input):
                    output, hidden = self.model(ascii_tensor, hidden)
                test_loss += F.nll_loss(output, label, reduction='sum').item()

                # FIXME precision error
                # metrics = self._eval_metric(output, label)
                # for key, value in metrics.items():
                #     if key not in test_metrics: test_metrics[key] = 0 # additive identity
                #     test_metrics[key] += value
                outputs.append(output[0])
                labels.append(label)

            outputs = torch.stack(outputs, 0)
            labels = torch.cat(labels, 0)
            test_metrics = self._eval_metric(outputs, labels)

        test_loss /= len(self.test_dataloader.dataset)
        # for key in test_metrics:
        #     test_metrics[key] /= len(self.test_dataloader.dataset)
        logger.info(f'Test set loss: [{test_loss:.4f}]')

        logger.info('Evaluation scores')
        for key, value in test_metrics.items():
            logger.info(f"{key:<20}{value:.6f}")
            if 'tensorboard' in self.cfg:
                self.tensorboard.add_scalar(f'val/{key}', value, self.epoch)
        return test_metrics

    def _eval_metric(self, logits, labels, training=False) -> dict:
        score, preds = torch.max(logits, dim=1)
        metrics = {}
        metrics['accuracy'] = self._accuracy(preds, labels)
        if not training:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels.cpu(), preds.cpu(), beta=1, average='weighted', zero_division=1)
            metrics['precision'] = precision.item()
            metrics['recall'] = recall.item()
            metrics['f1'] = f1.item()
        return metrics

    def _accuracy(self, preds, labels):
        assert preds.shape == labels.shape
        n = preds.shape[0]
        accuracy = torch.sum(preds == labels).item() / n
        return accuracy

    def after_epoch(self):
        metrics = self.evaluate()
        is_best = metrics['precision'] > self.best_ap
        self.best_ap = max(self.best_ap, metrics['precision'])  # is_best? :
        if is_best:
            logger.info('Best ckpt')
        self.save_ckpt(is_best)

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_one_epoch()
            self.after_epoch()

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100)
        )
        if 'tensorboard' in self.cfg:
            self.tensorboard.add_scalar('map', self.map, self.epoch)

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def load_dataset(self, name: str, data_cfg: dict):
        dataset = NameDataset(
            name=name, directory=data_cfg['root_dir'], ratio=data_cfg['ratio'])
        return dataset
    
    def get_class(self, output: Tensor):
        top_v, top_i = torch.topk(output, 1)
        category_i = top_i[0].item()
        return self.classes[category_i], category_i


class NameDataset(Dataset):
    def __init__(self, name="all", directory=None, ratio=1.0):
        self.name = name
        self.category_lines = {}
        self.all_categories = []
        self.ratio = ratio
        if directory is not None:
            self.load_data(directory)

    def __len__(self):
        return sum(list(len(self.category_lines[key]) for key in self.all_categories))

    def __getitem__(self, idx, cat_idx=0):
        category = self.all_categories[cat_idx]
        category_line = self.category_lines[category]
        if len(category_line) > idx:
            return category_line[idx], category, self.line2tensor(category_line[idx]), torch.tensor([cat_idx])
        else:
            return self.__getitem__(idx-len(category_line), cat_idx+1)

    def get_random_line(self):
        pass

    @property
    def all_letters(self):
        return string.ascii_letters + ".,;'"

    def output2category(self, output: Tensor):
        top_v, top_i = torch.topk(output, 1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

    def unicode2ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn' and c in self.all_letters)

    def ascii2tensor(self, c):
        '''One-Hot 벡터'''
        tensor = torch.zeros(1, len(self.all_letters))
        tensor[0][self.all_letters.find(c)] = 1
        return tensor

    def line2tensor(self, line):
        tensor = torch.zeros(len(line), 1, len(self.all_letters))
        for li, c in enumerate(line):
            tensor[li][0][self.all_letters.find(c)] = 1
        return tensor

    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        if self.ratio:
            num = int(len(lines) * self.ratio)
            if self.name == 'test':
                lines = lines[-num:]
            else:
                lines = lines[:num]
        return [self.unicode2ascii(line) for line in lines]

    def load_data(self, directory) -> dict:
        for filename in os.listdir(directory):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.read_lines(os.path.join(directory, filename))
            self.category_lines[category] = lines
        return self.category_lines
