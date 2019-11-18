from typing import Type

from torch import optim


def get_optimizer_from_str(name: str) -> Type[optim.optimizer.Optimizer]:
    if name == 'Adam':
        return optim.Adam
    elif name == 'Adadelta':
        return optim.Adadelta #type: ignore
    elif name == 'RMSprop':
        return optim.RMSprop #type: ignore
    else:
        raise NotImplementedError('Unknown optimizer')

