cpu=0
for i in $(seq 0 <num_clients>); do
        mkdir -p ../logs/client_$i
        docker run --network flotilla-network \
                --env MQTT_IP=<mqtt_broker_ip> --env MQTT_PORT=<mqtt_broker_port> \
                --name="client_$i" \
                --memory="4096m" \
                --cpuset-cpus="$cpu" \
                --mount type=bind,source=<absolute_path_of_the_dir_created_in_line_3>,target=/src/logs \
                --mount type=bind,source=<absolute_path_to_client_i_training_data>,target=/src/data \
                --log-driver=json-file \
                -dti flotilla-client:latest
        echo "$cpu"
        cpu=$(($cpu+1))
done
