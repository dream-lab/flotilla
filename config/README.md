# Flotilla Configuration

This directory contains the configuration files and logger setup for Flotilla. These configuration files define various settings and parameters required for different aspects of the project. Below are detailed explanations of each configuration file:

## 1. [training_config.yaml](training_config.yaml)

This file contains the configuration settings for the training session, benchmark, and training process.

### `session_config`:

- `session_id`: A unique identifier for the training session, which helps track and manage different training runs.
- `aggregator`: The type of aggregator used during federated learning. Set to `None` for default aggregation.
- `client_selection`: The client selection method used in federated learning. This determines how clients are selected to participate in each training round. Possible values include 'default', 'random', or custom selection strategies.
- `percentage_client_selection`: The percentage of clients selected in each training round when using random client selection.

### `benchmark_config`:

- `bench_model_id`: A unique identifier for the benchmark model.
- `bench_model_dir`: The directory path to the benchmark model files.
- `bench_model_class`: The class name of the benchmark model in the code.
- `bench_dataset_id`: A unique identifier for the benchmark dataset.
- `bench_minibatch_count`: The number of minibatches used for benchmarking the device's training performance.
- `bench_batch_size`: The batch size used during the benchmarking process.
- `learning_rate`: The learning rate used for benchmark training.
- `bench_timeout_duration_s`: The timeout duration in seconds for the benchmark training process.

### `train_config`:

- `model_id`: A unique identifier for the training model.
- `model_dir`: The directory path to store the model files.
- `model_class`: The class name of the model in the code.
- `dataset_id`: A unique identifier for the training dataset.
- `num_training_rounds`: The number of federated learning rounds to perform.
- `epochs`: The number of epochs to train the model on each client during a federated learning round.
- `batch_size`: The batch size used during federated learning training.
- `learning_rate`: The learning rate used during federated learning training.
- `train_timeout_duration_s`: The timeout duration in seconds for each federated learning training round.
- `loss_function`: The loss function used for model optimization during training.
- `optimizer`: The optimization algorithm used for model training.
- `validation_data_path`: The directory path to fetch the validation data.
- `validation_batch_size`: The batch size of the data used for evaluating the global model. The evaluation is done for 1 minibatch.

## 2. [server_config.yaml](server_config.yaml)

This file contains the communication configuration settings for the server.

### `comm_config`:

- `mqtt`: Configuration for MQTT (Message Queuing Telemetry Transport) communication protocol:
  - `type`: The type of MQTT configuration. Set to `server` to specify server-related MQTT settings.
  - `mqtt_broker`: The IP address or hostname of the MQTT broker.
  - `mqtt_broker_port`: The port number for the MQTT broker.
  - `mqtt_sub_timeout_s`: The timeout duration in seconds for MQTT subscriptions.
  - `mqtt_server_topic`: The topic name used by the server to publish messages.
  - `mqtt_client_topic`: The topic name used by clients to publish messages.

- `grpc`: Configuration for gRPC (Google Remote Procedure Call) communication protocol:
  - `chunk_size_bytes`: The chunk size in bytes used for data transmission.
  - `timeout_s`: The timeout duration in seconds for gRPC communication.

### `temp_dir_path`:

The directory path where temporary files are stored on the client.

## 3. [client_config.yaml](client_config.yaml)

This file contains the communication configuration settings for the client.

### `comm_config`:

- `mqtt`: Configuration for MQTT communication protocol:
  - `type`: The type of MQTT configuration. Set to `client` to specify client-related MQTT settings.
  - `mqtt_broker`: The IP address or hostname of the MQTT broker.
  - `mqtt_broker_port`: The port number for the MQTT broker.
  - `mqtt_sub_timeout_s`: The timeout duration in seconds for MQTT subscriptions.
  - `heartbeat_timeout_s`: The timeout duration in seconds for client heartbeat messages.

- `grpc`: Configuration for gRPC communication protocol:
  - `workers`: The number of worker threads to handle gRPC communication.
  - `sync_port`: The port number for synchronous gRPC communication.
  - `async_port`: The port number for asynchronous gRPC communication.

### `dataset_config`:

- `datasets_dir_path`: The directory path where datasets are stored on the client.

### `general_config`:

- `temp_dir_path`: The directory path where temporary files are stored on the client.
- `cleanup_session`: Set to `True` to enable session cleanup after training; otherwise, set to `False`.
- `use_gpu`: Set to `True` to use GPU for training (if available); otherwise, set to `False`.

## 4. [logger.conf](logger.conf)

This file configures the loggers, handlers, and formatters for the project.

- `[loggers]`: Defines the names of different loggers used in the project.
- `[handlers]`: Specifies the names of the log handlers (`fileHandler` and `streamHandler`) used to handle log messages.
- `[formatters]`: Defines the names and formatting details of log formatters.
- `[logger_xxx]`: Each logger name (e.g., `logger_SERVER_MANAGER`) is associated with a specific log level (e.g., `DEBUG`) and a log handler (e.g., `fileHandler`).
- `[formatter_fileFormatter]`: Specifies the format and date format for log messages.
- `[handler_fileHandler]` and `[handler_streamHandler]`: Configures the log handlers, including their log levels, associated formatters, and any additional arguments.

Please note that these configurations are user specific and are used to set up various aspects of the training, server communication, and logging functionality. Make sure to adjust the values accordingly for your specific use case.

For more information on how to use and customize these configuration files, refer to the project documentation or relevant code comments.
