"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import uuid
from concurrent import futures
from threading import Event, Thread

import grpc
import torch
from grpc import Server

import proto.grpc_pb2_grpc as grpc_pb2_grpc
from client.client_file_manager import (
    get_available_datasets,
    get_available_models,
    setup_dir,
)
from client.client_grpc_manager import ClientGRPCManager
from client.client_mqtt_manager import ClientMQTTManager
from client.utils.ip import get_ip_address, get_ip_address_docker
from client.utils.port_allocator import port_allocator
from utils.logger import FedLogger


class ClientManager:
    def __init__(self, client_id: int, client_config: dict, client_info: dict):
        self.client_id: str = client_id
        self.session_id: str = str(uuid.uuid4())

        self.client_info = client_info

        if client_config["general_config"]["use_gpu"]:
            self.torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.torch_device = torch.device("cpu")
        if client_config["general_config"]["use_gpu"] and self.torch_device == "cpu":
            self.logger.warn(
                f"WARNING: GPU not available on Client{self.client_id}.",
                "Setting torch_device as 'cpu'",
            )

        try:
            ev = os.environ["DOCKER_RUNNING"]
            print("RUNNING INSIDE DOCKER")
        except KeyError:
            print("RUNNING ON BARE METAL")
            ev = False

        self.ip: str = get_ip_address_docker() if ev else get_ip_address()
        print(self.ip)

        self.grpc_config: dict = client_config["comm_config"]["grpc"]
        self.mqtt_config: dict = client_config["comm_config"]["mqtt"]

        self.grpc_workers: int = self.grpc_config["workers"]
        self.init_grpc_port: int = int(self.grpc_config["sync_port"])

        self.grpc_port: int = port_allocator(self.ip, self.init_grpc_port)
        self.grpc_ep: str = f"{self.ip}:{self.grpc_port}"

        self.opts: list = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
            ("grpc.so_reuseport", 0),
            ("grpc.so_reuseaddr", 0),
        ]

        # setting up necessary directories
        self.temp_dir_path = client_config["general_config"]["temp_dir_path"]
        self.datasets_dir_path = client_config["dataset_config"]["datasets_dir_path"]
        setup_dir(dir_path=self.temp_dir_path)

        # get available datasets and models
        # self.models_available = get_available_models(self.temp_dir_path)
        self.dataset_details, self.dataset_paths = get_available_datasets(
            self.datasets_dir_path
        )

        # saving the cleanup options
        self.cleanup_session_on_exit: bool = client_config["general_config"][
            "cleanup_session_on_exit"
        ]
        self.cleanup_model_cache_on_exit: bool = client_config["general_config"][
            "cleanup_model_cache_on_exit"
        ]
        self.cleanup_temp_on_exit: bool = client_config["general_config"][
            "cleanup_temp_on_exit"
        ]

        # setting up client logger
        self.logger = FedLogger(id=self.client_id, loggername="CLIENT_MANAGER")

    def mqtt_init(self, stop_event: Event) -> Thread:
        """
        Function to start the client's MQTT service
        """

        self.logger.info("MQTT.client.init", "")
        mqtt_client = ClientMQTTManager(
            id=self.client_id,
            mqtt_config=self.mqtt_config,
            grpc_config=self.grpc_config,
            temp_dir_path=self.temp_dir_path,
            dataset_details=self.dataset_details,
            client_info=self.client_info,
        )
        mqtt_task = Thread(target=mqtt_client.mqtt_sub, args=(stop_event,))
        mqtt_task.start()
        return mqtt_task

    def grpc_init(self, stop_event: Event) -> Server:
        """
        Function to start the client's gRPC service
        """

        sync_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.grpc_workers), options=self.opts
        )
        grpc_pb2_grpc.add_EdgeServiceServicer_to_server(
            ClientGRPCManager(
                client_id=self.client_id,
                temp_dir_path=self.temp_dir_path,
                torch_device=self.torch_device,
                dataset_paths=self.dataset_paths,
                client_info=self.client_info,
            ),
            sync_server,
        )
        sync_server.add_insecure_port(f"{self.ip}:{self.grpc_port}")
        sync_server.start()
        self.logger.info("fedclient_gRPC.init", "")
        return sync_server

    def run(self) -> None:
        """
        Function that starts the client's MQTT and gRPC services, and waits for kill signals.
        The function also implements the sequence of events that happens once the client receives
        a KeyBoardInterrupt.
        """
        try:
            print(f"client id: {self.client_id}")
            stop_event = Event()

            self.logger.info("fedclient.init", "")

            grpc_sync_server = self.grpc_init(stop_event)

            mqtt_server = self.mqtt_init(stop_event)

            stop_event.wait()

        except KeyboardInterrupt:
            self.logger.info(
                "fedclient.Keyboard_interrupt",
                "Received KeyboardInterrupt starting exit procedure",
            )
            self.exit_procedure(stop_event, grpc_sync_server, mqtt_server)
            self.logger.info("fedclient.Keyboard_interrupt.exit", "")

    def exit_procedure(self, stop_event, grpc_sync_server, mqtt_server):
        grpc_sync_server.stop(grace=None)
        stop_event.set()
        mqtt_server.join()
