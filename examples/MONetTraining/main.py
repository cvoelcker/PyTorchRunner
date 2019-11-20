from torchvision import transforms

from config_parser.config_parser import ConfigGenerator
from torch_runner.experiment_setup import setup_experiment, load_config, get_model
from torch_runner.data import transformers, file_loaders
from torch_runner.data.base import BasicDataSet
from spatial_monet.spatial_monet import MaskedAIR

config, config_object = load_config()

monet = get_model(config, MaskedAIR)
config, config_object = setup_experiment(config, config_object, debug=False)

data_config = config.DATA

source_loader = file_loaders.DirectoryLoader(directory=data_config.data_dir, compression_type='pickle')

transformers = [
        transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
        transformers.TypeTransformer(config.EXPERIMENT.device)
        ]

data = BasicDataSet(source_loader, transformers)
