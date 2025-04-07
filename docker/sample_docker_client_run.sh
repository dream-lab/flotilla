# docker rm -f $(docker ps -aq)
# cpu=0
# for i in $(seq 1 208); do
#         mkdir -p client_$i
#         docker run --network rigel-overlay \
#                 --env MQTT_IP=10.0.4.254 --env MQTT_PORT=1883 \
#                 --memory="4096m" \
#                 --cpuset-cpus="$cpu" \
#                 --mount type=bind,source=/root/client_$i,target=/fedml-ng/logs \
#                 --mount type=bind,source=/root/client_data/part_$i,target=/fedml-ng/data \
#                 --log-driver=none \
#                 -dti flotilla-client
#         echo "$cpu"
#         cpu=$(($cpu+1))
# done
# docker stats

# docker rm -f $(docker ps -aq)
cpu=0
for i in $(seq 18 20); do
        mkdir -p ../logs/client_$i
        docker run --network flotilla-network \
                --env MQTT_IP=10.24.24.32 --env MQTT_PORT=1884 \
                --name="client_$i" \
                --memory="4096m" \
                --cpuset-cpus="$cpu" \
                --mount type=bind,source=/home/fedml/flotilla_final/fedml-ng/logs/client_$i,target=/src/logs \
                --mount type=bind,source=/home/fedml/flotilla_final/fedml-ng/client_data_CIFAR10/part_$i,target=/src/data \
                --log-driver=json-file \
                -dti flotilla-client:latest
        echo "$cpu"
        cpu=$(($cpu+1))
done
