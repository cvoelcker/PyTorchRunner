import sys
from typing import Tuple, Any, Type

import torch
from config_parser.config_parser import ConfigGenerator

from torch_runner.train.train import AbstractTrainer
from torch_runner.util.torch_utils import get_optimizer_from_str
from torch_runner.data.base import BasicDataSet


def setup_config(file_location: str, argv) -> Tuple[Any, ConfigGenerator]:
    config = ConfigGenerator(file_location)
    return config(argv), config


def load_and_check_experiment_config(config):
    try:
        experiment_config = config.experiment_config
    except AttributeError as e:
        print('Config does not specify experiment setup')
        sys.exit(1)
    return experiment_config


def setup_trainer(trainer_class: Type[AbstractTrainer], model: torch.nn.Module, training_config):
    trainer = trainer_class()
    trainer.register_model(model)
    
    optimizer = get_optimizer_from_str(training_config.optimizer)
    trainer.register_optimizer(optimizer, training_config.lr, **training_config.optimizer_config._asdict())


def setup_train_dataloader(trainer, dataset: BasicDataSet):
    dataloader = torch.utils.data.dataloader.DataLoader(dataset)
    trainer.register_train_dataloader(dataloader)


def setup_test_dataloader(trainer, dataset: BasicDataSet):
    dataloader = torch.utils.data.dataloader.DataLoader(dataset)
    trainer.register_test_dataloader(dataloader)
