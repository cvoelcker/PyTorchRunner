from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Optional, Any

from torchvision.transforms import transforms #type: ignore
from PIL import Image #type: ignore


class DataTransformation(ABC):

    @abstractmethod
    def transform(self, data):
        pass


class TorchVisionTransformerComposition(DataTransformation):

    possible_transforms = {
            'crop': lambda shape: transforms.Lambda(lambda x: transforms.functional.crop(x, *shape)),
            'reshape': lambda shape: transforms.Resize(shape[-3:-1]),
            'float': lambda _: transforms.Lambda(lambda x: x.float()),
            'torch': lambda _: transforms.ToTensor(),
            'normalize': lambda _: transforms.Lambda(lambda x: x/255.)
            }

    @staticmethod
    def unpack(transform_name_list, shape: Optional[Iterable[int]] = None):
        transforms_list = []
        for t in transform_name_list:
            try:
                transforms_list.append(TorchVisionTransformerComposition.possible_transforms[t](shape))
            except KeyError as e:
                raise NotImplementedError(f'Transformation {t} not available')
        return transforms.Compose(transforms_list)
        
    def __init__(self, transform_list: List[str], shape: Optional[Iterable[int]] = None):
        self.transforms = TorchVisionTransformerComposition.unpack(transform_list, shape)
    
    def transform(self, data: Dict[str, Any]):
        img = Image.fromarray(data)
        if self.transforms is not None:
            # print(self.transform)
            img = self.transforms(img)
            # print(img.shape)
        return img


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
        'cuda': lambda x: x.cuda(),
        'numpy': lambda x: x.cpu().numpy(),
        }

    def __init__(self, device: str):
        if device not in TypeTransformer.transformations.keys():
            raise ValueError(f'Device type {device} not recognized')
        self.transformation = TypeTransformer.transformations[device]

    def transform(self, data):
        return self.transformation(data)
