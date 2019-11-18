from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from torch_runner.util.data_util import DataLoaderType


class AbstractDataSet(Dataset, ABC):
    
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass


class TypeConversionMixin(ABC):
    
    @abstractmethod
    def _get_item(self, idx: int) -> torch.Tensor:
        pass

    def __getitem__(self, idx: int):
        d = self._get_item(idx)
        return self.to_device(d)

    @abstractmethod
    def to_device(self, d: torch.Tensor) -> torch.Tensor:
        pass


class CPUTypeMixin(TypeConversionMixin):
    data_type = DataLoaderType.CPU

    def to_device(self, d):
        return d.cpu()


class GPUTypeMixin(TypeConversionMixin):
    data_type = DataLoaderType.GPU

    def to_device(self, d):
        return d.cuda()


class NumpyTypeMixin(TypeConversionMixin):
    data_type = DataLoaderType.NUMPY

    def to_device(self, d):
        return d.cpu().numpy()


#convenience classes for quick access to type mixins
class BaseCPUDataSet(CPUTypeMixin, AbstractDataSet):
    pass


class BaseGPUDataSet(GPUTypeMixin, AbstractDataSet):
    pass


class BaseNumpyDataSet(NumpyTypeMixin, AbstractDataSet):
    pass

