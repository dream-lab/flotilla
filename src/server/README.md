# Server

The `Server` directory contains the necessary files to carry out the server-side tasks of Federated learning.

## 1. [server_file_manager.py](server_file_manager.py)

   `server_file_manager.py` is responsible for reading all the models in the [server_models](../server_models) directory and returning a dictionary with key-value pairs of `model_name:model_object`. This allows the server to access and manage the models effectively.

## 2. [server_model_manager.py](server_model_manager.py)

   `server_model_manager.py` provides the model instances and validation data required for the global model at [server_manager.py](server_manager.py). It acts as a bridge between the server manager and the models, ensuring that the server can utilize the model's functionalities effectively during the training and inference stages.

## 3. [server_mqtt_manager.py](server_mqtt_manager.py)

   `server_mqtt_manager.py` manages the MQTT (Message Queuing Telemetry Transport) functionality for the server. It receives the crucial information of clients such as client's gRPC, client id, and periodically gets the heartbeat responses. Client's gRPC endpoint is essential for establishing the gRPC communications for sending model files, benchmarking and training.

## 4. [server_manager.py](server_manager.py)

   `server_manager.py` plays a crucial role in management of sending model files, benchmarking and training of client's, aggregation, and communication with the clients. It oversees the federated learning process on the server side, selecting the clients as per client selction, coordinating model updates and aggregating the client contributions to build the global model.

## Additional Functionality

The `Server` directory also contains the following modules that provide necessary functionalities for server_manager:

- Aggregator ([aggregator.py](aggregator.py)): The aggregator module handles the model weights aggregation received from the clients during the federated learning process. It ensures that the model updates from different clients are combined effectively to create an updated global model.

- Client Selection Loader ([load_client_selection.py](load_client_selection.py)): The client selection loader module provides the functionality for selecting clients for the federated learning process. It determines the clients that will participate in each round of training to create a diverse and representative dataset.

- Loss Function Loader ([load_loss.py](load_loss.py)): The loss function loader module handles loading the appropriate loss function for the federated learning process. The choice of the loss function depends on the specific problem being addressed by the models.

- Optimizer Loader ([load_optimizer.py](load_optimizer.py)): The optimizer loader module is responsible for loading the suitable optimizer to be used during model training. Optimizers are crucial for updating the model weights and minimizing the loss during the training process.

# Notes

The server fetches all the training configurations so, user can select the appropriate client selection scheme, loss function, optimizer, etc from [training_config.yaml](..%2Fconfig%2Ftraining_config.yaml) 

and MQTT,gRPC configurations from [server_config.yaml](..%2Fconfig%2Fserver_config.yaml)


Ensure that the server files are properly organized and located in the `Server` directory to facilitate smooth execution and coordination of the federated learning process.
