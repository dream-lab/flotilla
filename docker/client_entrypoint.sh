sed -i "s/type:.*/type: docker_client/1" config/client_config.yaml
sed -i "s/client_name:.*/client_name: docker_$(hostname)/1" config/client_config.yaml
sed -i "s/mqtt_broker:.*/mqtt_broker: $MQTT_IP/1" config/client_config.yaml
sed -i "s/mqtt_broker_port:.*/mqtt_broker_port: $MQTT_PORT/1" config/client_config.yaml
cat config/client_config.yaml
python ./flo_client.py --monitor
