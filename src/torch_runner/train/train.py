from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Type, Any

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer

from torch_runner.handlers.base import HandlerType, AbstractHandler
from torch_runner.util.data_util import DataLoaderType


class AbstractTrainer(ABC):

    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.handlers: List[AbstractHandler] = []
        self.model = model

    # handler registration block
    def register_handler(self, handler: AbstractHandler):
        self.handlers.append(handler)

    def register_step_handler(self, handler: AbstractHandler):
        handler.set_callback_type(HandlerType.AFTER_STEP)
        self.register_handler(handler)
    
    def register_epoch_handler(self, handler: AbstractHandler):
        handler.set_callback_type(HandlerType.AFTER_EPOCH)
        self.register_handler(handler)

    def register_train_handler(self, handler: AbstractHandler):
        handler.set_callback_type(HandlerType.AFTER_TRAIN)
        self.register_handler(handler)

    # Handler notification block
    def notify_handlers(self, data: Dict, notify_type: HandlerType):
        for handler in self.handlers:
            if handler.callback_type == notify_type:
                handler.notify(data)

    def notify_step_handlers(self, data: Dict):
        self.notify_handlers(data, HandlerType.AFTER_STEP)

    def notify_epoch_handlers(self, data: Dict):
        self.notify_handlers(data, HandlerType.AFTER_EPOCH)

    def notify_train_handlers(self, data: Dict):
        self.notify_handlers(data, HandlerType.AFTER_TRAIN)

    # Model and optimizer setting
    def register_model(self, model: torch.nn.Module):
        self.model = model

    def register_optimizer(self, optimizer: Type[Optimizer], lr: float, optimizer_params={}):
        if self.model is None:
            raise ValueError('Cannot register optimizer without module')
        else:
            self.optimizer = optimizer(self.model.parameters(), lr=lr, **optimizer_params) #type: ignore

    def add_train_dataloader(self, dataloader: DataLoader, data_type: DataLoaderType):
        self.train_dataloader = dataloader
    
    def add_test_dataloader(self, dataloader: DataLoader, data_type: DataLoaderType):
        self.test_dataloader = dataloader

    def train(self, epochs: int, train_only: bool = False):
        training_info_dict: Dict[str, Any] = {}
        for e in range(epochs):
            epoch_info_dict: Dict[str, Any] = {}
            for d in self.train_dataloader:
                data = self.train_step(d)
                self.notify_step_handlers(data)
                epoch_info_dict = self.append_epoch_info_dict(epoch_info_dict, data) 
                training_info_dict = self.append_training_info_dict(training_info_dict, data)
            if not train_only:
                for d in self.test_dataloader:
                    data = self.train_step(d)
                    self.notify_step_handlers(data)
                    epoch_info_dict = self.append_epoch_info_dict(epoch_info_dict, data) 
                    training_info_dict = self.append_training_info_dict(training_info_dict, data)
            self.compile_epoch_info_dict(epoch_info_dict)
            self.notify_epoch_handlers(epoch_info_dict)
        training_info_dict = self.compile_training_info_dict(training_info_dict)
        self.notify_train_handlers(training_info_dict)

    # Information dictionaries for handlers
    def append_epoch_info_dict(self, epoch_info_dict: Dict, data_dict: Dict) -> Dict:
        return epoch_info_dict

    def append_training_info_dict(self, training_info_dict: Dict, data_dict: Dict) -> Dict:
        return training_info_dict

    def compile_epoch_info_dict(self, epoch_info_dict: Dict) -> Dict:
        return epoch_info_dict

    def compile_training_info_dict(self, training_info_dict: Dict) -> Dict:
        return training_info_dict
    
    # the actual training core class

    # This needs to be overridden
    @abstractmethod
    def train_step(self, data) -> Dict:
        pass
