mkdir -p ../server_logs
mkdir -p ../checkpoints
docker run --network flotilla-network \
    --name server \
	--env REDIS_IP=<redis_server_ip> --env REDIS_PORT=<redis_port> \
	--env MQTT_IP=<mqtt_broker_ip> --env MQTT_PORT=<mqtt_port> \
	--env MQTT_TIMEOUT=1 \
	--memory 32GB \
	--mount type=bind,source=<absolute_path_of_dir_created_in_line_1>,target=/src/logs/ \
	--mount type=bind,source=<absolute_path_to_server_validation_data_dir>,target=/src/val_data/ \
	--mount type=bind,source=<absolute_path_of_dir_created_in_line_2>,target=/src/checkpoint/ \
    --mount type=bind,source=<absolute_path_to_the_model_dir>,target=/src/models/ \
	-p 12345:12345 \
	-ti flotilla-server:latest