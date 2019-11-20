from abc import ABC, abstractmethod
import os, shutil
from typing import Dict, Any

from .base import AbstractHandler, AbstractStepHandler, HandlerType
from torch_runner.util.tf_logger import Logger


class TensorboardHandler(AbstractHandler):

    def __init__(self, logdir: str='tb_logs', namedir: str='default', reset_logdir: bool=True):
        full_path = os.path.abspath(logdir) + '/' + namedir
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
        super().__init__(HandlerType.INVALID)

    def notify(self, data: Dict[str, Any]):
        for tag, value in data['tf_logging'].items():
            self.logger.scalar_summary(tag, value, self.step)
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
    def __init__(self, n:int, logdir: str='tb_logs', namedir: str='default', reset_logdir: bool=True):
        self.n = n
        super().__init__(logdir=logdir, namedir=namedir, reset_logdir=reset_logdir)

    def notify(self, data: Dict[str, Any]):
        if 'step' not in data:
            raise ValueError('Cannot determine whether visualization is needed')
        if data['step'] % self.n and data['step'] > 0:
            super().notify(data)

