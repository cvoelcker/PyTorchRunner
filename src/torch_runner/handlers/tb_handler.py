from abc import ABC, abstractmethod
import os, shutil
from typing import Dict, Any, Optional, List

import torch

from .base import AbstractHandler, AbstractStepHandler, HandlerType
from torch_runner.util.tf_logger import Logger


class TensorboardHandler(AbstractHandler):

    def __init__(self, logdir: str='tb_logs', namedir: str='default', log_name_list: Optional[List[str]]=None, reset_logdir: bool=True):
        full_path = os.path.join(os.path.abspath(logdir), namedir)
        self.logger = Logger(full_path)
        print(f'Logging to {full_path}')
        if reset_logdir:
            for the_file in os.listdir(full_path):
                file_path = os.path.join(full_path, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        self.step = 0
        self.log_name_list = log_name_list
        super().__init__()

    def register_logging(self, log_key: str):
        if self.log_name_list is None:
            self.log_name_list = [log_key]
        else:
            self.log_name_list.append(log_key)

    def notify(self, data: Dict[str, Any]):
        for tag, value in data.items():
            if (self.log_name_list is not None) and (tag not in self.log_name_list):
                continue
            self.logger.scalar_summary(tag, value.detach().cpu().numpy().mean(), self.step)
        if 'step' in data:
            self.step += data['step']
        self.step += 1

    def reset(self):
        self.step = 0
        self.logger = Logger('../logs')


### Example for extending the step handler
class StepTbHandler(TensorboardHandler, AbstractStepHandler): pass


### Example for overwriting the notify method to enable logging only every n steps
class NStepTbHandler(TensorboardHandler, AbstractStepHandler):
    def __init__(self, n:int, logdir: str='tb_logs', namedir: str='default', log_name_list: Optional[List[str]]=None, reset_logdir: bool=True):
        self.n = n
        self.data = []
        super().__init__(logdir=logdir, namedir=namedir, reset_logdir=reset_logdir, log_name_list=log_name_list)

    def notify(self, data: Dict[str, Any]):
        self.data.append({k: torch.mean(v.detach().cpu()) for k, v in data.items() if (self.log_name_list is None) or (k in self.log_name_list)})
        if self.step % self.n == 0 and self.step > 0:
            d = {k: torch.mean(torch.stack([dic[k] for dic in self.data], 0)) for k in self.data[0]}
            super().notify(d)
            self.data = []
        self.step += 1 if 'step' not in data else data['step']

