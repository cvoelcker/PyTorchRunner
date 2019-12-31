from abc import ABC, abstractmethod
from typing import List, Union, Optional, Generic, TypeVar

import torch
from torch.utils.data import Dataset

import numpy as np #type: ignore

from torch_runner.util.data_util import DataLoaderType
from .transformers import DataTransformation, TypeTransformer


LoadedData = TypeVar('LoadedData')
ProvidedData = TypeVar('ProvidedData')

class DataSource(Generic[LoadedData, ProvidedData], ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_dataset(self):
        pass


class BasicDataSet(Dataset):

    def __init__(self, source_loader: DataSource, transformations: List[DataTransformation], loader_type: Optional[str] = ''):
        self.dataset = source_loader.get_dataset()
        self.n = len(self.dataset)
        self.transformations: List[DataTransformation] = []
        for t in transformations:
            self.transformations.append(t)
        if loader_type:
            self.transformations.append(TypeTransformer(loader_type))
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.dataset)        

    def __getitem__(self, idx: int) -> Union[np.ndarray, torch.Tensor]:
        datum = self.dataset[idx]
        for transformation in self.transformations:
            datum = transformation.transform(datum)
        return datum
    
    def get_all(self):
        for datum in self:
            yield datum


class SequenceDataSet(Dataset):

    def __init__(self, source_loader: DataSource, transformations: List[DataTransformation], sequence_len:int, loader_type: Optional[str] = ''):
        self.dataset = source_loader.get_dataset()
        self.n = len(self.dataset)
        self.transformations: List[DataTransformation] = []
        self.sequence_len = sequence_len
        for t in transformations:
            self.transformations.append(t)
        if loader_type:
            self.transformations.append(TypeTransformer(loader_type))
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return (self.dataset.shape[1] - self.sequence_len) * self.dataset.shape[0]

    def __getitem__(self, idx: int) -> Union[np.ndarray, torch.Tensor]:
        seq = idx //(self.dataset.shape[1] - self.sequence_len)
        idx = idx % (self.dataset.shape[1] - self.sequence_len)
        datum = self.dataset[seq, idx:idx+self.sequence_len]
        imgs = []
        for img in datum:
            for transformation in self.transformations:
                img = transformation.transform(img)
            imgs.append(img)
        return torch.stack(imgs, 0)

    def get_all(self):
        for idx in range(self.__len__()):
            for item in self.__getitem__(idx):
                yield item


class SequenceDictDataSet(Dataset):

    def __init__(self,
            source_loader: DataSource,
            transformations: List[DataTransformation],
            sequence_len:int,
            loader_type: Optional[str] = ''):
        self.dataset = source_loader.get_dataset()
        assert 'X' in self.dataset.keys(), 'Need a primary data entry'
        self.n = len(self.dataset['X'])
        self.transformations: List[DataTransformation] = []
        self.sequence_len = sequence_len
        for t in transformations:
            self.transformations.append(t)
        if loader_type:
            self.transformations.append(TypeTransformer(loader_type))
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        l = (self.dataset['X'].shape[1] - self.sequence_len) * self.dataset['X'].shape[0]
        return l

    def __getitem__(self, idx: int) -> Union[np.ndarray, torch.Tensor]:
        seq = idx //(self.dataset['X'].shape[1] - self.sequence_len)
        idx = idx % (self.dataset['X'].shape[1] - self.sequence_len)
        datum = {}
        for k in self.dataset:
            if 'X' == k:
                imgs = []
                for img in self.dataset[k][seq][idx:idx+self.sequence_len]:
                    for transformation in self.transformations:
                        img = transformation.transform(img)
                    imgs.append(img)
                datum[k] = torch.stack(imgs, 0)
            else:
                if not hasattr(self.dataset[k], 'shape'):
                    continue
                datum[k] = self.dataset[k][seq][idx:idx+self.sequence_len]
        return datum

    def get_all(self):
        for seq in self:
            for item in seq['X']:
                yield item
