# Models

This folder contains various models that the server can use for training or benchmarking. Each model is organized in a separate directory named after its `model_id`, and it includes a Python file defining the `Model` class and a `config.yaml` file with model-specific details. If a model needs a custom `Training` or `Validation` function provide them in the `trainer.py` file. If a model needs a custom `DataLoader` function provide the function in the `loader.py` file.

#### For custom files use the provided naming convention. (It might work with other names, but it's not guaranteed).

## Model Directory Structure

The models in this folder follow the following directory structure:

```text
models/
|-- model_id/
| |-- model.py
| |-- config.yaml
|-- another_model_id/
| |-- model.py
| |-- trainer.py     # If custom train/validation functions are provided
| |-- loader.py      # If custom dataloader function is provided
| |-- config.yaml
|-- ...
```


## Model Configuration

Each model directory contains a `config.yaml` file, which provides specific details about the model and its training configuration.

### `config.yaml`

The `config.yaml` file contains the following information:

- `model_details`:
  - `model_id`: A unique identifier for the model.
  - `model_class`: Name of the Python class defining the model in `model.py` file.
  - `model_tags`: A list of tags associated with the model. These tags help identify the model's characteristics or type, e.g., CNN, RNN, etc.
  - `suitable_datasets`: A list of dataset IDs that are suitable for training or benchmarking with this model.

- `training_config`:
  - `use_custom_dataloader`: A boolean flag to specify if a custom implementation of DataLoader is provided. If `False` the framework will use the default implementation of the DataLoader
    - `custom_dataloader_args`: If `use_custom_dataloader` is set to `True`, provide all the arguments that the Training function takes.
  - `use_custom_trainer`: A boolean flag to specify if a custom implementation of Training function for the model is provided.
    - `custom_trainer_args`: If `use_custom_trainer` is set to `True`, provide all the arguments that the Training function takes.
  - `use_custom_validator`: A boolean flag to specify if a custom implementatin of Validation functon for the model is provided.
    - `custom_validator_args`: If `use_custom_validator` is set to `True`, provide all the arguments that the Validation function takes.
  - `model_args`: All the arguments needed to initialize the model, for example the `num_classes` for the number of classes to train on.

Below is an example of a [`config.yaml`](AlexNet/config.yaml) file for the model named AlexNet:

```yaml
  model_details:
  model_id: AlexNet
  model_class: AlexNet_class
  model_tags: [CNN]
  suitable_datasets: [MNIST, FMNIST]

training_config:
  use_custom_dataloader: False
  custom_loader_args: None

  use_custom_trainer: False
  custom_trainer_args: None

  use_custom_validator: False
  custom_validator_args: None

  model_args:
    num_classes: 10
```
