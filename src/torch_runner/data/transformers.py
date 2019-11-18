from abc import ABC, abstractmethod

from torchvision.transforms import transforms #type: ignore
from PIL import Image #type: ignore


class DataTransformation(ABC):

    @abstractmethod
    def transform(self, data):
        pass


class ImageTransformer(DataTransformation):

    def __init__(self, transform: transforms.Compose):
        self.transform_function = transform
    
    def transform(self, data):
        img = Image.fromarray(data)
        if self.transform is not None:
            img = self.transform_function(img)
        return img


class TypeTransformer(DataTransformation):

    transformations = {
        'cpu': lambda x: x.cpu(),
        'gpu': lambda x: x.cuda(),
        'numpy': lambda x: x.cpu().numpy(),
        }

    def __init__(self, device: str):
        if device not in TypeTransformer.transformations.keys():
            raise ValueError(f'Device type {device} not recognized')
        self.transformation = TypeTransformer.transformations[device]

    def transform(self, data):
        return self.transformations(data)
