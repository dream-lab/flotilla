# Client

The `Client` directory contains the necessary files to carry out the client-side tasks of Federated Learning. 


## 1. [client_file_manager.py](client_file_manager.py)

   `client_file_manager.py` has a number of utility functions responsible for saving the models sent by server to a temporary directory, loading the custom classes sent by the server and returning a dictionary with key-value pairs of `model_name:model_object`. This allows the client to access and manage the models effectively.

## 2. [client_dataset_loader.py](client_dataset_loader.py)

   `client_dataset_loader.py` implements functions that loads the training data using either the default or the custom dataloader functions and returns the train and test dataloaders required for the training and benchmarking.

## 3. [client_mqtt_manager.py](client_mqtt_manager.py)

   `client_mqtt_manager.py` implements the `MQTT (Message Queuing Telemetry Transport)` functionality for the client. It implements the `ClientMQTTManager` class that manages all MQTT functionalities of the client. It allows the client to send its details such as client's `gRPC endpoint`, client id, and periodically sends the heartbeat. `gRPC endpoint` is essential for server to establish the gRPC connection to send model files, invoke benchmarking and training.

## 4. [client_grpc_manager.py](client_grpc_manager.py)

   `client_grpc_manager.py` implements the client-side functionalities, allowing it to receive models as well as invoke model benchmark and training rounds in [client_trainer.py](client_trainer.py).

## 5. [client_manager.py](client_manager.py)

   `client_manager.py` oversees the federated learning process on the client side, creating a `gRPC server` instance at client, instantiating `MQTT` and `ClientEdgeService` class.

## 6. [client_trainer.py](client_trainer.py)

   `client_trainer.py` carries out the training function at the client and returns the results to [client_grpc_manager.py](client_grpc_manager.py).

# Notes

The client fetches all the training configurations from server and MQTT,gRPC configurations from [client_config.yaml](..%2Fconfig%2Fclient_config.yaml).

Ensure that the client files are properly organized and located in the `Client` directory to facilitate smooth execution and coordination of the federated learning process.
