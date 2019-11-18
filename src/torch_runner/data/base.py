from abc import ABC, abstractmethod
from typing import List, Union, Optional

import torch
from torch.utils.data import Dataset

import numpy as np #type: ignore

from torch_runner.util.data_util import DataLoaderType
from .file_loaders import DataSource
from .transformers import DataTransformation, TypeTransformer


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

