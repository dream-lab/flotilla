"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import json
import time
from threading import Event, Thread

import paho.mqtt.client as mqtt

from utils.hardware_info import get_hardware_info
from utils.logger import FedLogger


class MQTTManager:
    def __init__(self, config: dict) -> None:
        self.logger: FedLogger = FedLogger(id="0", loggername="SERVER_MQTT_MANAGER")
        self.hw_info: dict = get_hardware_info()

        self.mqtt_broker: str = config["mqtt_broker"]
        self.mqtt_broker_port: int = config["mqtt_broker_port"]
        self.mqtt_server_topic: str = config["mqtt_server_topic"]
        self.mqtt_client_topic: str = config["mqtt_client_topic"]
        self.mqtt_sub_timeout: int = config["mqtt_sub_timeout_s"]
        self.mqtt_heartbeat_interval_s: int = config["mqtt_heartbeat_interval_s"]
        self.num_heartbeats_timestamp_cached: int = config[
            "num_heartbeats_timestamp_cached"
        ]
        self.max_heartbeats_miss_threshold: int = config["max_heartbeat_miss_threshold"]
        self.type_: str = config["type"]

        self.heard_from_client_event: Event = Event()

    def mqtt_ad(self, client_info, stop_event, grpc_event):
        def on_connect(client, userdata, flags, rc):
            self.logger.info(
                "MQTT.server.connect.request", f"MQTT connection status,{rc}"
            )

        def on_subscribe(client, userdata, mid, granted_qos):
            self.logger.info(
                "MQTT.server.subscribe.request", f"subscribe tracking variable:,{mid}"
            )

        def on_publish(client, userdata, mid):
            self.logger.info(
                "MQTT.server.publish.request", f"publish tracking variable:,{mid}"
            )

        def message_ad_response(client, userdata, message):
            info = json.loads(str(message.payload.decode()))

            client_id = list(info.keys())[0]

            client_name = info[str(client_id)]["payload"]["name"]
            client_info.put(f"{client_id}.client_name", client_name)

            grpc_ep = info[str(client_id)]["payload"]["grpc_ep"]
            client_info.put(f"{client_id}.grpc_ep", grpc_ep)

            benchmark_info = info[str(client_id)]["payload"]["benchmark_info"]
            client_info.put(f"{client_id}.benchmark_info", benchmark_info)

            hardware_information = info[str(client_id)]["payload"]["hw_info"]
            client_info.put(f"{client_id}.hardware_information", hardware_information)

            role = info[str(client_id)]["payload"]["type"]
            client_info.put(f"{client_id}.role", role)

            dataset_details = info[str(client_id)]["payload"]["datasets"]
            client_info.put(f"{client_id}.dataset_details", dataset_details)

            models = info[str(client_id)]["payload"]["models"]
            client_info.put(f"{client_id}.models", models)

            client_info.put(f"{client_id}.is_active", True)
            client_info.put(f"{client_id}.is_training", False)
            client_info.put(f"{client_id}.heartbeat.timestamp", [time.time()])
            client_info.put(f"{client_id}.heartbeat.interval", 0)
            client_info.put(f"{client_id}.join_timestamp", time.time())

            self.logger.info(
                "MQTT.server.ad_response_received",
                f"{client_id},{client_name}",
            )
            self.logger.info("MQTT.server.ad_response", f"{client_name}:{info}")
            userdata.set()
            print("MQTT.ad_response from", client_id)

        def message_heartbeat_response(client, userdata, message):
            info = json.loads(str(message.payload.decode()))
            client_id = info["id"]

            self.logger.info(
                "MQTT.server.heartbeat.received.message",
                f"heartbeat from client and timestamp:,{client_id},{info['timestamp']}",
            )

            try:
                server_time = time.time()
                client_heartbeat_timestamp = (
                    client_info.get(f"{client_id}.heartbeat.timestamp")
                    if isinstance(
                        client_info.get(f"{client_id}.heartbeat.timestamp"), list
                    )
                    else [time.time()]
                )
                interval = round(
                    (server_time - client_heartbeat_timestamp[-1]),
                    2,
                )

                if (
                    len(client_heartbeat_timestamp)
                    == self.num_heartbeats_timestamp_cached
                ):
                    client_heartbeat_timestamp.pop()
                client_heartbeat_timestamp.append(server_time)

                client_info.put(f"{client_id}.heartbeat.interval", interval)
                client_info.put(
                    f"{client_id}.heartbeat.timestamp", client_heartbeat_timestamp
                )
            except KeyError:
                self.logger.warn(
                    "MQTT.server.heartbeat.invalid.client",
                    f"Ignoring heartbeat as client_session doesn't contain Client,{id}",
                )

        client_user_data = self.heard_from_client_event
        client = mqtt.Client(f"flo_server", userdata=client_user_data)

        client.on_connect = on_connect
        client.on_subscribe = on_subscribe
        client.on_publish = on_publish

        client.connect(self.mqtt_broker, self.mqtt_broker_port, keepalive=60)
        self.logger.info(
            "MQTT.server.broker.connect",
            f"Connected to MQTT Broker at,{self.mqtt_broker}:{self.mqtt_broker_port}",
        )

        client.message_callback_add(self.mqtt_client_topic, message_ad_response)
        client.message_callback_add("heartbeat", message_heartbeat_response)
        client.subscribe(self.mqtt_client_topic)
        client.subscribe("heartbeat")
        self.logger.info(
            "MQTT.server.subscribed.topics",
            f"Subscribed to topics:,{self.mqtt_client_topic},heartbeat",
        )

        client.loop_start()

        payload = json.dumps(
            {
                "type": self.type_,
                "timestamp": time.time(),
                "grpc_ep": None,
                "mqtt_client_topic": self.mqtt_client_topic,
                "cluster_id": None,
                "heartbeat_interval": self.mqtt_heartbeat_interval_s,
                "hw_info": self.hw_info,
            }
        )

        client.publish(topic=self.mqtt_server_topic, payload=payload, retain=True)

        self.logger.info(
            "MQTT.server.advert.publish.topic",
            f"Payload published on topic,{self.mqtt_server_topic}",
        )
        self.logger.debug(
            "MQTT.server.advert.publish.payload",
            f"Payload for client: ,{payload}",
        )

        wait_increment = 0
        while not self.heard_from_client_event.is_set():
            if wait_increment == 0:
                time.sleep(self.mqtt_sub_timeout)
            else:
                self.logger.warn("MQTT.server.await", "No clients found")
                self.heard_from_client_event.wait(self.mqtt_sub_timeout)
            wait_increment += 1

        self.logger.info("MQTT.server.await.timeout", "Waiting for clients timeout")
        self.logger.debug(
            "MQTT.server.clients", f"Clients populated: ,{client_info.keys()}"
        )

        grpc_event.set()

        heartbeat_thread = Thread(
            target=self.heartbeat_alive_check, args=(client_info,)
        )
        heartbeat_thread.start()

        stop_event.wait()
        client.loop_stop()

    def heartbeat_alive_check(self, client_info):
        heartbeat_interval_flag = Event()
        while not heartbeat_interval_flag.is_set():
            for client in client_info.keys():
                if client_info.get(f"{client}.is_active"):
                    if time.time() - client_info.get(f"{client}.heartbeat.timestamp")[
                        -1
                    ] >= (
                        (
                            self.max_heartbeats_miss_threshold
                            * self.mqtt_heartbeat_interval_s
                        )
                        + 2
                    ):
                        print(f"Removing client:{client} from active clients")
                        self.logger.warn(
                            "MQTT.server.heartbeat.delayed",
                            f"Removing client:{client} from active clients",
                        )
                        client_info.put(f"{client}.is_active", False)
                        client_info.put(f"{client}.is_training", False)
            heartbeat_interval_flag.wait(self.mqtt_heartbeat_interval_s)
