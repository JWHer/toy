import argparse, os, torch, json, sys
from torch.backends import cudnn
from loguru import logger

from trainer.base_trainer import TrainerFactory
from utils.config import configure_nccl, get_gpu_nums

def make_parser():
    parser = argparse.ArgumentParser("Trainer")
    parser.add_argument("config", type=str, help='config name or file path')

    parser.add_argument("--log_dir", type=str, help="log file directory")
    parser.add_argument("--log_level", type=str, help="logger level", default='info')
    parser.add_argument("--train", type=str, help="train dataset dir")
    parser.add_argument("--valid", type=str, help="validation dataset dir")
    parser.add_argument("--test", type=str, help="test dataset dir")

    parser.add_argument("--gpus", type=str, help='gpu for training')
    parser.add_argument("--ckpt", type=str, help="checkpoint file path")
    parser.add_argument("--resume", type=bool, help="resume training", default=False)
    return parser

def parse_config(kwargs:argparse.Namespace) -> dict:
    kwargs = vars(kwargs)
    config:str = kwargs.pop('config')
    config_basename:str = os.path.basename(config)
    if config == config_basename: config = f"./configs/{config}.json"

    with open(config) as config_file:
        model_config = json.load(config_file)
    kwargs.update(model_config)

    return {k: v for k, v in kwargs.items() if v is not None}

@logger.catch
def main(**kwargs):
    if kwargs: logger.info(f'Kwargs provided.\n{kwargs}')

    # set environment variables for distributed training
    # configure_nccl()
    # cudnn.benchmark = True

    task = kwargs.pop("task")
    trainer = TrainerFactory.get_trainer(task, **kwargs)
    # trainer.before_train()
    # trainer.show_data(idx=0, save=True)
    trainer.train()

if __name__ == "__main__":
    
    kwargs = make_parser().parse_args()
    kwargs = parse_config(kwargs)

    if 'gpus' in kwargs:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]= kwargs['gpus']
    # kwargs["num_gpus"] = get_gpu_nums()
    
    if 'log_level' in kwargs:
        logger.remove()
        logger.add(sys.stderr, level=kwargs['log_level'].upper())

    main(**kwargs)
