import os
import pickle
import dill #type: ignore
from typing import Optional, List, Tuple, Any, Callable, Dict, BinaryIO

import torch

from .base import AbstractHandler


def wrap_file_open(data: Dict, file_name: str, function: Callable[[Any, BinaryIO], None]):
    """
    Helper function to allow a lambda wrappig of saving functions which require opening file wrappers
    """
    with open(file_name, 'wb') as f:
        function(data, f)


class FileHandler(AbstractHandler):
    
    SUPPORTED = ['pickle', 'torch']
    
    @staticmethod
    def setup_compression(method):
        if method == 'pickle':
            return lambda data, file_name: wrap_file_open(data, file_name, pickle.dump)
        if method == 'torch':
            return lambda data, file_name: torch.save(data, file_name)
    
    def __init__(self, log_dir: str, compression_method:str='pickle', log_name_list: Optional[List[str]]=None):
        if compression_method not in FileHandler.SUPPORTED:
            raise NotImplementedError('Currently only supports the following compression methods: ' + ' ,'.join(FileHandler.SUPPORTED))
        self.log_dir = log_dir
        self.saving = FileHandler.setup_compression(compression_method)
        self.step = 0
        self.log_name_list = log_name_list
        super().__init__() #type: ignore

    def register_logging(self, log_key: str):
        if self.log_name_list is None:
            self.log_name_list = [log_key]
        else:
            self.log_name_list.append(log_key)

    def notify(self, data):
        if 'step' in data:
            self.step += data['step']
        else:
            self.step += 1
        for key in data:
            if (self.log_name_list is not None) and (key not in self.log_name_list):
                continue
            full_path = os.path.join(self.log_dir, key + '.save')
            self.saving(data[key], full_path)