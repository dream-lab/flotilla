"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import hashlib
import importlib
import inspect
import os
import pickle
import sys

import yaml

from utils.logger import FedLogger


def hash_bytestr_iter(bytesiter, hasher, ashexstr=False):
    for block in bytesiter:
        hasher.update(block)
    return hasher.hexdigest() if ashexstr else hasher.digest()


def file_as_blockiter(afile, blocksize=65536):
    with afile:
        block = afile.read(blocksize)
        while len(block) > 0:
            yield block
            block = afile.read(blocksize)


def file_as_bytes(file):
    with file:
        return file.read()


def setup_dir(dir_path: str) -> None:
    """Function that creates a directory at "dir_path", if it does not exist already"

    Args:
        path (str): Path of the directory to be created
    """
    try:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    except Exception as e:
        print(f"Exception from setup_dir: {e}")


def add_init_file_to_dir(dir_path: str, empty_init_file: bool = True) -> None:
    """Adds an __init__.py to "dir_path", if it does not exist already
    Args:
        dir_path (str): Path of the directory to add __int__.py file in
        empty_init_file (bool, optional): Adds an empty __init__.py file if Set to True. Otherwise writes the code specified in the program. Defaults to True.
    """
    filepath = f"{dir_path}/__init__.py"
    if not os.path.isfile(filepath):
        with open(filepath, "w") as f:
            if empty_init_file:
                file_contents = ""
            else:
                file_contents = """from os.path import dirname, basename, isfile, join 
                \nimport glob 
                \nmodules = glob.glob(join(dirname(__file__), "*.py")) 
                \n__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]"""

            f.write(file_contents)


def setup_model_dir(temp_dir_path: str, model_id: str) -> None:
    """Function to setup a model directory in "<temp_dir_path>/model_cache/<model_name>" and adds an __init__.py file to the direcory, if already not present

    Args:
        temp_dir_path (str):     Relative/absolute location to the temp directory
        model_id (str):   Name of the model. The SendModelFiles function in client_grpc_manager.py saves the
                            model files in "<temp_dir>/model_cache/<model_name>"
        add_init_file (bool, optional): Adds a __init__.py file to "<temp_dir>/model_cache/<model_name>" if it is not already present. Defaults to True.
    """
    try:
        setup_dir(os.path.join(temp_dir_path, "model_cache"))
        dir_path = os.path.join(temp_dir_path, "model_cache", model_id)
        setup_dir(dir_path=dir_path)
    except Exception as e:
        print(f"Exception from setup_model_dir: {e}")
    # add_init_file_to_dir(dir_path= dir_path)


def get_dataset_details(path: str) -> dict:
    # dataset_dir_path = os.path.abspath(os.path.join(path, os.pardir))
    print(path)
    summary_path = path.split(".")[1] + "_summary.data"
    summary_path = "." + summary_path
    print(summary_path)
    try:
        summary = None
        with open(summary_path, "rb") as f:
            summary = pickle.load(f)

    except Exception as e:
        print("client_file_manager.get_dataset_details:: Error reading summary.")
        print("Exception: ", e)

    return summary


def get_available_datasets(path: str) -> list:
    """Function that returns a list of all datasets present in data directory.

    Args:
        path (str): path to the temp directory

    Returns:
        list: list of model_ids
    """

    available_datasets_dir = list()
    available_datasets = dict()
    available_datasets_path = dict()
    if os.path.isdir(path):
        available_datasets_dir = [f.name for f in os.scandir(path) if f.is_dir()]
        for dataset in available_datasets_dir:
            with open(os.path.join(path, dataset, "train_dataset_config.yaml")) as file:
                try:
                    dataset_config = yaml.safe_load(file)
                except yaml.YAMLError as err:
                    print(err)
            available_datasets_path[dataset] = os.path.join(
                path, dataset, dataset_config["dataset_details"]["data_filename"]
            )
            del dataset_config["dataset_details"]["data_filename"]
            available_datasets[dataset] = dataset_config

    return available_datasets, available_datasets_path


def get_available_models(path: str) -> list:
    """Function that returns a list of all model_ids present in <temp_dir_path>/model_cache

    Args:
        path (str): path to the temp directory

    Returns:
        list: list of model_ids, model_hashes
    """

    dir_path = f"{path}/model_cache/"
    abs_dir_path = os.path.abspath(dir_path)

    available_models_hashes = dict()
    if os.path.isdir(abs_dir_path):
        available_models = [f.name for f in os.scandir(abs_dir_path) if f.is_dir()]
        for model in available_models:
            model_dir_path = os.path.join(abs_dir_path, model)
            model_hash = hex(0)
            model_files = [
                file for file in os.scandir(model_dir_path) if not os.path.isdir(file)
            ]
            hashes = [
                hashlib.sha256(file_as_bytes(open(fname, "rb"))).hexdigest()
                for fname in model_files
            ]
            for hash in hashes:
                model_hash = hex(int(model_hash, 16) + int(hash, 16))
            available_models_hashes[model] = model_hash

    return available_models_hashes


def get_model_class(path: str, model_id: str, class_name: str):
    """
    Function that takes in the location of the temp directory, the name of the ML model
    and the name of the class that contains the PyTorch model as well as the train and the
    validation functions and returns an object to that class
    """

    model_cache_dir_path = os.path.join(path, "model_cache")
    model_dir_path = os.path.join(model_cache_dir_path, model_id)
    if os.path.isdir(model_cache_dir_path):
        add_init_file_to_dir(dir_path=model_dir_path, empty_init_file=False)
        model_cache_dir_abspath = os.path.abspath(model_cache_dir_path)
        sys.path.append(model_cache_dir_abspath)
        module_imported = importlib.import_module(model_id, package=None)
        model_dir_abspath = os.path.abspath(model_dir_path)
        sys.path.append(model_dir_abspath)

        for file in module_imported.__all__:
            model_imported = importlib.import_module(
                f"{path}.model_cache.{model_id}.{file}"
            )

            for name_local in dir(model_imported):
                unknown_class = getattr(model_imported, name_local)

                if (
                    inspect.isclass(unknown_class)
                    and unknown_class.__name__ == class_name
                ):
                    print(
                        f"client_file_manager.get_model_class:: class {unknown_class} is loaded."
                    )
                    return unknown_class

        print(
            f"\nclient_file_manager.get_model_class:: {class_name} not found in {model_id}\n"
        )
        return None

    print(f"\nclient_file_manager.get_model_class:: {model_dir_path} not found\n")
    return None


def read_yaml(path: str):
    """Reads the config.yaml file from the model directory

    Args:
        path (str): Path to the config.yaml in the model directory

    """

    if os.path.isfile(path):
        with open(path) as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as err:
                print(err)

        return config
    else:
        print("config.yaml for model not found.")  # this has to go back to the server


def write_yaml(path: str, data: dict) -> None:
    """Writes a yaml file to path

    Args:
        path (str): path of the yaml file
        data (dict): data to be written to the yaml file
    """
    with open("data.yml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def OpenYaML(path: str, logger: FedLogger = None):
    with open(path) as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            config = None
            if hasattr(exc, "problem_mark"):
                if logger:
                    logger.error("fedclient.OpenYAML.exception", "Incorrect data")
            else:
                if logger:
                    logger.error(
                        "fedclient.OpenYaML.exception", f"Error reading {path}"
                    )
        finally:
            return config
