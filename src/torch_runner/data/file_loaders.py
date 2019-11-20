import os
import gzip #type: ignore
import dill #type: ignore
import pickle
from typing import Iterable, Callable, TypeVar, Generic

from abc import ABC, abstractmethod

import numpy as np #type: ignore


LoadedData = TypeVar('LoadedData')
ProvidedData = TypeVar('ProvidedData')

class DataSource(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_dataset(self):
        pass


class DirectoryLoader(DataSource, Generic[LoadedData, ProvidedData]):
    def __init__(self, directory: str = '', 
            compression_type: str = 'pickle', 
            preprocess_function: Callable[[LoadedData], LoadedData] = lambda x: x,
            **kwargs):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.compression_type = compression_type
        self.preprocess_function = preprocess_function

    def get_dataset(self) -> ProvidedData:
        dataset = []

        for imgfile in self.filenames:
            imgpath = os.path.join(self.directory, imgfile)
            if self.compression_type == 'gzip':
                with gzip.open(imgpath, 'rb') as f:
                    img = self.preprocessing_function(dill.load(f))
                dataset.append(img)
            elif self.compression_type == 'pickle':
                with open(imgpath, 'rb') as f:
                    img = self.preprocess_function(pickle.load(f))
                dataset.append(img)
        return np.array(dataset).squeeze()


class FileLoader(DataSource, Generic[LoadedData]):
    def __init__(self, file_name: str = '', 
            compression_type: str = 'pickle', 
            preprocess_function: Callable[[LoadedData], LoadedData] = lambda x: x,
            **kwargs):
        self.file_name = file_name
        self.compression_type = compression_type
        self.preprocess_function = preprocess_function

    def get_dataset(self):
        if self.compression_type == 'gzip':
            with gzip.open(self.file_name, 'rb') as f:
                dataset = self.preprocess_function(dill.load(f))
        elif self.compression_type == 'pickle':
            with open(self.file_name, 'rb') as f:
                dataset = self.preprocess_function(pickle.load(f))
        return dataset

