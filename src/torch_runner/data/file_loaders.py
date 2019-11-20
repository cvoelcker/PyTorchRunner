import os
import gzip #type: ignore
import dill #type: ignore
import pickle

from abc import ABC, abstractmethod

import numpy as np #type: ignore


class DataSource(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_dataset(self):
        pass


class DirectoryLoader(DataSource):
    def __init__(self, directory: str = '', compression_type: str = 'pickle', **kwargs):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.compression_type = compression_type

    def get_dataset(self):
        dataset = []

        for imgfile in self.filenames:
            imgpath = os.path.join(self.directory, imgfile)
            if compression_type == 'gzip':
                with gzip.open(imgpath, 'rb') as f:
                    img = dill.load(f)
                dataset.append(img)
            elif compression_type == 'pickle':
                with open(imgpath, 'rb') as f:
                    img = pickle.load(f)
                dataset.append(img)
        return dataset


class FileLoader(DataSource):
    def __init__(self, file_name: str = '', compression_type: str = '', **kwargs):
        self.file_name = file_name
        self.compression_type = compression_type

    def get_dataset(self):
        if compression_type == 'gzip':
            with gzip.open(self.file_name, 'rb') as f:
                dataset = dill.load(f)
        elif compression_type == 'pickle':
            with open(self.file_name, 'rb') as f:
                dataset = pickle.load(f)
        return dataset
