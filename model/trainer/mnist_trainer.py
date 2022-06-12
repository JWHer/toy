import codecs, torch, os, gzip, numpy as np
import torch.nn.functional as F
from typing import IO, Union
from PIL import Image
from loguru import logger
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support

from trainer.base_trainer import Trainer
from classifier.classifier import Classifier

class MnistTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def train_one_iter(self):
        pass

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
        # Batches
        for idx, (inputs, labels) in enumerate(self.train_dataloader):
            # update_lr
            loss = 0

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Backward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # loss = outputs['total_loss']
            # loss = criterion(outputs, labels)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            logger.debug(
                f'Train Batch: [{(idx+1)*len(inputs)}/{len(self.train_dataloader.dataset)}]\
                \tLoss: {loss:.6f}'
            )
        # batch_iter_num = len(self.train_dataloader.dataset) / self.train_dataloader.batch_size\
        #     + (len(self.train_dataloader.dataset) % self.train_dataloader.batch_size > 0)
        batch_iter_num = len(self.train_dataloader.dataset)
        logger.info(f'Train Epoch: [{self.epoch}/{self.max_epoch}]\tLoss: {running_loss/batch_iter_num:.6f}')

    def after_epoch(self):
        metrics = self.evaluate()
        is_best = metrics['precision'] > self.best_ap
        self.best_ap = max(self.best_ap, metrics['precision']) # is_best? :
        if is_best: logger.info('Best ckpt')
        self.save_ckpt(is_best)

    def before_train(self):
        self.model = Classifier(**self.cfg)
        if torch.cuda.is_available(): self.model.cuda()
        # logger.info(f"Model Summary: {self.get_model_info()}")
        self.train_dataloader = self._get_dataloader('train')
        self.test_dataloader = self._get_dataloader('test')
        self.optimizer = self._get_optimizer()
        # self.lr = self._get_lr()

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if 'tensorboard' in self.cfg:
            self.tensorboard.add_scalar('map', self.map, self.epoch)

    def evaluate(self):
        logger.info("Start Evaluation")

        self.model.eval()
        test_loss = 0
        test_metrics = {}
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = self.model(inputs)
                test_loss += F.nll_loss(outputs, labels, reduction='sum').item()
                # pred = outputs.data.max(1, keepdim=True)[1]
                # correct += pred.eq(labels.data.view_as(pred)).sum()

                metrics = self._eval_metric(outputs, labels)
                for key, value in metrics.items():
                    if key not in test_metrics: test_metrics[key] = 0 # additive identity
                    test_metrics[key] += value

        test_loss /= len(self.test_dataloader.dataset)
        for key in test_metrics:
            test_metrics[key] /= len(self.test_dataloader.dataset)
        logger.info(f'Test set loss: [{test_loss:.4f}]')

        logger.info('Evaluation scores')
        for key, value in metrics.items():
            logger.info(f"{key:<20}{value:.6f}")
            if 'tensorboard' in self.cfg:
                self.tensorboard.add_scalar(f'val/{key}', value, self.epoch)
        return metrics

    def _eval_metric(self, logits, labels, training=False) -> dict:
        score, preds = torch.max(logits, dim=1)
        metrics = {}
        metrics['accuracy'] = self._accuracy(preds, labels)
        if not training:
            precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), beta=1, average='weighted', zero_division=1)
            metrics['precision']  = precision.item()
            metrics['recall']     = recall.item()
            metrics['f1']         = f1.item()
        return metrics

    def _accuracy(self, preds, labels):
        assert preds.shape == labels.shape
        n = preds.shape[0]
        accuracy = torch.sum(preds == labels) / n
        return accuracy.item()

    def load_dataset(self, name:str, data_cfg:dict):
        dataset = MnistDataset(
            read_image_file(os.path.join(data_cfg['root_dir'], data_cfg['compression'])),
            read_label_file(os.path.join(data_cfg['root_dir'], data_cfg['annotation'])),
        )
        # # with open(os.path.join(data_cfg['root_dir'], f"{name}.pt"), 'wb') as f:
        # #     torch.save(dataset, f)

        # dataset = datasets.MNIST(
        #     './torchvision', train=(name=='train'), download=True,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])
        # )
        return dataset

    # def show_data(self, name='train', idx=None, save=False):
    #     # you can override here
    #     pass

class MnistDataset(Dataset):
    def __init__(self, images:torch.Tensor, labels:torch.Tensor, transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx]), self.labels[idx]
        return self.images[idx].float().unsqueeze(0), self.labels[idx]

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)

SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}

def read_sn3_pascalvincent_tensor(path: Union[str, IO], strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with gzip.open(path, 'rb') as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

def read_label_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()

def read_image_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x
    # imgs = [Image.fromarray(xi.numpy(), mode='L') for xi in x]
    # return imgs

# Ref. Archiver
# if __name__ != '__main__':
#     TrainerFactory.register_trainer('mnist', Mnist)
