from typing import Type

import torch.optim


def get_optimizer_from_str(name: str) -> Type[torch.optim.Optimizer]: #type: ignore
    if name == 'Adam':
        return torch.optim.Adam
    elif name == 'Adadelta':
        return torch.optim.Adadelta #type: ignore
    elif name == 'RMSprop':
        return torch.optim.RMSprop #type: ignore
    else:
        raise NotImplementedError('Unknown optimizer')

