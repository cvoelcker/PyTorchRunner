from typing import Type

import torch.optim


def get_optimizer_from_str(name: str) -> Type[torch.optim.Optimizer]: #type: ignore
    optim = None
    if name == 'Adam':
        optim = torch.optim.Adam
    elif name == 'Adadelta':
        optim = torch.optim.Adadelta #type: ignore
    elif name == 'RMSprop':
        optim = torch.optim.RMSprop #type: ignore
    else:
        raise NotImplementedError('Unknown optimizer')
    return optim
