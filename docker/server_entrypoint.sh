sed -i "s/type:.*/type: docker_server/1" config/server_config.yaml
sed -i "s/timeout_s:.*/timeout_s: $GRPC_TIMEOUT/1" config/server_config.yaml
sed -i "s/mqtt_broker:.*/mqtt_broker: $MQTT_IP/1" config/server_config.yaml
sed -i "s/mqtt_broker_port:.*/mqtt_broker_port: $MQTT_PORT/1" config/server_config.yaml
sed -i "s/mqtt_sub_timeout_s:.*/mqtt_sub_timeout_s: $MQTT_TIMEOUT/1" config/server_config.yaml
sed -i "s/state_hostname:.*/state_hostname: $REDIS_IP/1" config/server_config.yaml
sed -i "s/state_port:.*/state_port: $REDIS_PORT/1" config/server_config.yaml
cat config/server_config.yaml
python ./flo_server.py --monitor
