import os
import sys
from typing import Tuple, Any, Type, Optional
import shutil

import torch
from config_parser.config_parser import ConfigGenerator

from torch_runner.train.base import AbstractTrainer
from torch_runner.data.base import BasicDataSet
from torch_runner.util.torch_utils import get_optimizer_from_str


def setup_trainer(trainer_class: Type[AbstractTrainer], model: torch.nn.Module, training_config, train_data: BasicDataSet, test_data: Optional[BasicDataSet]=None):
    trainer = trainer_class()
    trainer.register_model(model)
    
    optimizer = get_optimizer_from_str(training_config.optimizer.optimizer_name)
    if hasattr(training_config.optimizer, 'attributes'):
        trainer.register_optimizer(optimizer, training_config.optimizer.lr, **training_config.optimizer.attributes._asdict())
    else:
        trainer.register_optimizer(optimizer, training_config.optimizer.lr)
    if hasattr(training_config, 'decay_rate'):
        trainer.register_scheduler(training_config.decay_rate, training_config.decay_schedule)
    if training_config.clip_gradient:
        trainer.clip_gradient = True
        trainer.clip_gradient_value = training_config.clip_gradient_value
    else:
        trainer.clip_gradient = False
    setup_train_dataloader(trainer, train_data, training_config)
    if test_data is not None:
        setup_test_dataloader(trainer, test_data, training_config)
    return trainer


def setup_train_dataloader(trainer, dataset: BasicDataSet, config, shuffle: bool=True):
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)
    trainer.add_train_dataloader(dataloader)


def setup_test_dataloader(trainer, dataset: BasicDataSet, config, shuffle: bool=True):
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)
    trainer.add_test_dataloader(dataloader)


