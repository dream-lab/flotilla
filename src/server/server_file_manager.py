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
import sys

import yaml

from utils.logger import FedLogger


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


def get_model_class(path: str, class_name: str):
    """
        Function that takes in the location of the temp directory, the name of the ML model
    and the name of the class that contains the PyTorch model as well as the train and the
    validation functions and returns an object to that class

    Args:
        path (str):   Relative/absolute path to the temp directory
        class_name (str): Name of the class in one of the .py files in "<temp_dir>/model_cache/<model_name>"
                          that contains the PyTorch model as well as the train and validate functions
    """
    model_name = path.split("/")[-1]
    path = os.path.abspath(path)
    model_cache_dir_path = os.path.dirname(path)
    model_dir_path = path

    if os.path.isdir(model_cache_dir_path):
        add_init_file_to_dir(dir_path=model_dir_path, empty_init_file=False)
        model_cache_dir_abspath = os.path.abspath(model_cache_dir_path)
        sys.path.append(model_cache_dir_abspath)
        module_imported = importlib.import_module(model_name, package=None)
        model_dir_abspath = os.path.abspath(model_dir_path)
        sys.path.append(model_dir_abspath)

        for file in module_imported.__all__:
            model_imported = importlib.import_module(file, package=None)

            for name_local in dir(model_imported):
                unknown_class = getattr(model_imported, name_local)
                if (
                    inspect.isclass(unknown_class)
                    and unknown_class.__name__ == class_name
                ):
                    print(
                        f"\nserver_file_manager.get_model_class:: class {unknown_class} is loaded\n"
                    )
                    return unknown_class

        print(
            f"\nserver_file_manager.get_model_class:: {class_name} not found in {model_name}\n"
        )
        return None

    print(f"\nserver_file_manager.get_model_class:: {model_dir_path} not found\n")
    return None


def get_available_datasets(path: str) -> dict:
    """Function that returns a list of all datasets present in data directory. Each sub directory needs to contain \"dataset_config.yaml\"."""

    available_datasets_dir = list()
    available_datasets = dict()
    if os.path.isdir(path):
        available_datasets_dir = [f.name for f in os.scandir(path) if f.is_dir()]
        for dataset in available_datasets_dir:
            with open(os.path.join(path, dataset, "dataset_config.yaml")) as file:
                try:
                    dataset_config = yaml.safe_load(file)
                except yaml.YAMLError as err:
                    print(err)
            available_datasets[dataset] = dataset_config
            available_datasets[dataset]["dataset_details"][
                "data_filename"
            ] = os.path.join(
                path, dataset, dataset_config["dataset_details"]["data_filename"]
            )

    print("RETURNING DATASETS")
    return available_datasets


def file_as_bytes(file):
    with file:
        return file.read()


def get_model_dir_hash(path: str) -> str:
    model_hash = hex(0)
    abs_model_path = os.path.abspath(path)
    model_files = [
        file for file in os.scandir(abs_model_path) if not os.path.isdir(file)
    ]
    hashes = [
        hashlib.sha256(file_as_bytes(open(fname, "rb"))).hexdigest()
        for fname in model_files
    ]
    for hash in hashes:
        model_hash = hex(int(model_hash, 16) + int(hash, 16))

    return model_hash


def OpenYaML(path: str, logger: FedLogger = None) -> dict[str, str] | None:
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
