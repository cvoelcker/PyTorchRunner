from abc import ABC, abstractmethod
from enum import Enum


class HandlerType(Enum):
    INVALID = -1
    BEFORE_STEP = 0
    AFTER_STEP = 1
    BEFORE_EPOCH = 2
    AFTER_EPOCH = 3
    BEFORE_TRAIN = 4
    AFTER_TRAIN = 5


class AbstractHandler(ABC):

    def __init__(self, callback_type: HandlerType):
        self.callback_type = callback_type

    @abstractmethod
    def notify(self, data):
        pass

    def set_callback_type(self, callback_type: HandlerType):
        self.callback_type = callback_type


class AbstractStepHandler(AbstractHandler):
    
    def __init__(self):
        super().__init__(HandlerType.AFTER_STEP)


class AbstractEpochHandler(AbstractHandler):
    
    def __init__(self):
        super().__init__(HandlerType.AFTER_EPOCH)


class AbstractStartHandler(AbstractHandler):

    def __init__(self):
        super().__init__(HandlerType.BEFORE_TRAIN)


class AbstractFinishHandler(AbstractHandler):

    def __init__(self):
        super().__init__(HandlerType.AFTER_TRAIN)

