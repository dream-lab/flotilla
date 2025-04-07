mkdir -p ../server_logs
mkdir -p ../server_logs/docker_stats

docker run --network flotilla-network \
    --name server \
	--env REDIS_IP=10.24.24.32 --env REDIS_PORT=6379 \
	--env MQTT_IP=10.24.24.32 --env MQTT_PORT=1884 \
	--env MQTT_TIMEOUT=1 \
	--memory 32GB \
	--mount type=bind,source=/home/fedml/flotilla_final/fedml-ng/server_logs,target=/src/logs/ \
	--mount type=bind,source=/home/fedml/flotilla_final/fedml-ng/val_data,target=/src/val_data/ \
	--mount type=bind,source=/home/fedml/flotilla_final/fedml-ng/checkpoints,target=/src/checkpoint/ \
    --mount type=bind,source=/home/fedml/flotilla_final/fedml-ng/models,target=/src/models/ \
	-p 12345:12345 \
	-ti flotilla-server:latest