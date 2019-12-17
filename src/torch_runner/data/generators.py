from typing import Callable

from .base import DataSource, LoadedData


class FunctionLoader(DataSource):
    def __init__(self, generator_function,
            generator_function_args,
            preprocessing_function: Callable[[LoadedData], LoadedData] = lambda x: x,
            **kwargs):
        self.generator_function = generator_function
        self.preprocessing_function = preprocessing_function
        self.generator_function_args = generator_function_args

    def get_dataset(self):
        return self.preprocessing_function(self.generator_function(**self.generator_function_args))
