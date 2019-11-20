import os
import sys
import shutil
from typing import Tuple, Any, Type

from config_parser.config_parser import ConfigGenerator
import torch

from torch_runner.train.base import AbstractTrainer


def load_config() -> Tuple[Any, ConfigGenerator]:
    config = ConfigGenerator(sys.argv[1])
    return config(sys.argv[2:]), config


def check_experiment_config(config):
    print(config)
    try:
        ex = config.EXPERIMENT
        ex.load_run
        ex.experiment_dir
        ex.run_name
        ex.overwrite
        ex.run_number
    except AttributeError as e:
        print(e)
        print('Config does not specify full experiment setup')
        sys.exit(1)


def clean_experiment_directory(config):
    log_dir = config.EXPERIMENT.experiment_dir
    run_name = config.EXPERIMENT.run_name
    experiment_path = os.path.join(log_dir, run_name)
    print('CLEANING FULL EXPERIMENT DIRECTORIES AT ' + experiment_path)
    answer = input('ARE YOU SURE? [y/N]: ')
    if not answer in ['y', 'Y']:
        print('Will not remove experiment structure!')
        sys.exit(1)
    shutil.rmtree(experiment_path)


def load_config_from_file(run_path: str, run_number: str):
    config_file = os.path.join(run_path, 'config.yml')
    config_object = ConfigGenerator(config_file)
    config = config_object(['--run-number', str(run_number)])
    return config, config_object


def save_config(config_object: ConfigGenerator, run_path: str, run_number: str):
    print('Saving config to ' + run_path)
    config_file = os.path.join(run_path, 'config.yml')
    config_object.config_dict['EXPERIMENT']['run_number'] = run_number
    config_object.dump_config(config_file)


def setup_run(run_path: str):
    print('Building run directory at ' + run_path)
    os.makedirs(run_path)
    os.mkdir(os.path.join(run_path, 'checkpoints'))
    os.mkdir(os.path.join(run_path, 'data'))
    os.mkdir(os.path.join(run_path, 'results'))
    os.mkdir(os.path.join(run_path, 'logging'))


def make_clean_dir(run_path: str, debug: bool=False):
    if debug:
        print('Would delete ' + run_path)
    else:
        print('overwriting experiment at ' + run_path)
        answer = input('Are you sure? [y/N]: ')
        if not answer in ['y', 'Y']:
            print('No override, could not continue setup')
            sys.exit(1)
        shutil.rmtree(run_path)
        setup_run(run_path)


def find_next_run_number(experiment_path: str):
    i = 0
    try:
        for _ in os.listdir(experiment_path):
            i += 1
    except FileNotFoundError as e:
        pass
    return i


def get_run_path(experiment_dir, run_name, run_number):
    path = os.path.join(experiment_dir, run_name)
    path = os.path.join(path, 'run_{:03d}'.format(run_number))
    return path


def setup_experiment(config, config_object: ConfigGenerator, debug: bool = False):
        # does not delete files but prints to console
        check_experiment_config(config)
        debug = debug
        load_config = config.EXPERIMENT.load_run
        log_dir = config.EXPERIMENT.experiment_dir
        run_name = config.EXPERIMENT.run_name
        overwrite = config.EXPERIMENT.overwrite
        experiment_path = os.path.join(log_dir, run_name)
        try:
            setup_new_run = False
            run_number = config.EXPERIMENT.run_number
            if not (overwrite or load_config):
                print('Given a run_number but not told to overwrite, will setup new experiment')
                setup_new_run = True
                run_number = find_next_run_number(experiment_path)
        except AttributeError as e:
            if load_config:
                print('Cannot load run without run number (default should be set to 0')
                sys.exit(1)
            setup_new_run = True
            run_number = find_next_run_number(experiment_path)
        run_path = get_run_path(log_dir, run_name, run_number)
        if load_config:
            if overwrite:
                print('Cannot overwrite an experiment while reloading it')
                sys.exit(1)
            config, config_object = load_config_from_file(run_path, run_number)
            run_number = find_next_run_number(experiment_path)
            run_path = get_run_path(log_dir, run_name, run_number)
            setup_new_run = True
        if setup_new_run:
            setup_run(run_path)
        elif overwrite:
            make_clean_dir(run_path, debug)
        save_config(config_object, run_path, run_number)
        config, config_object = load_config_from_file(run_path, run_number)

        return config, config_object


def get_model(config, model_class: Type[torch.nn.Module]):
    model_config = config.MODULE
    model = model_class(model_config._asdict())
    ex = config.EXPERIMENT
    if ex.load_run:
        path = get_run_path(ex.experiment_dir, ex.run_name, ex.run_number)
        path = os.path.join(path, 'checkpoints')
        if not os.path.exists(path):
            print('Found no model checkpoints')
            sys.exit(1)
        try:
            checkpoint_number = ex.checkpoint_number
        except AttributeError as e:
            print('Did not specify checkpoint number, using last available')
            checkpoint_number = find_next_run_number(path) - 1
        path = os.path.join(path, 'model_{:03d}'.format(checkpoint_number))
        model_state_dict = torch.load(path)
        model.load_state_dict(model_state_dict)
    return model

