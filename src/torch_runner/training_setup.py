import os
import sys
from typing import Tuple, Any, Type
import shutil

import torch
from config_parser.config_parser import ConfigGenerator

from torch_runner.train.base import AbstractTrainer
from torch_runner.data.base import BasicDataSet
from torch_runner.util.torch_utils import get_optimizer_from_str


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


