"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import threading
import time
import os
from threading import Event

from server.server_mqtt_manager import MQTTManager
from server.server_session_manager import FloSessionManager
from server.server_state_manager import StateManager
from utils.logger import FedLogger


class FlotillaServerManager:
    def __init__(self, server_config: dict):
        self.logger = FedLogger(id="0", loggername="SERVER_MANAGER")

        self.server_config = server_config
        self.state = self.server_config["state"]

        self.client_info = StateManager(
            loc=self.state["state_location"],
            name="client_info",
            host=self.state["state_hostname"],
            port=self.state["state_port"],
        )

        self.mqtt_config: dict = self.server_config["comm_config"]["mqtt"]
        self.mqtt_stop_event = Event()
        self.mqtt_init_finish_event = Event()
        self.mqtt_obj = MQTTManager(self.mqtt_config)
        self.mqtt_task = threading.Thread(
            target=self.mqtt_obj.mqtt_ad,
            args=(self.client_info, self.mqtt_stop_event, self.mqtt_init_finish_event),
        )
        self.mqtt_task.name = "MQTT_Task_Thread"
        self.mqtt_task.start()

    async def run(
        self, id: str, train_config: dict, restore=False, revive=False, file=False
    ):
        session_run_time = time.time()

        self.logger.debug("fedserver.run.started", f"{id},{session_run_time}")
        session = FloSessionManager(
            id=id,
            client_info=self.client_info,
            mqtt_init_event=self.mqtt_init_finish_event,
            server_config=self.server_config,
            session_config=train_config,
            restore=restore,
            revive=revive,
            file=file,
        )
        await session.start_session()
        os.rename(f"logs/flotilla_{id}.log",f"logs/flotilla_{id}_{train_config['session_id']}")
        self.logger.debug(
            "fedserver.run.finished", f"{id},{time.time()-session_run_time}"
        )

    def get_active_clients(self):
        active_clients = [
            client_id
            for client_id in self.client_info.keys()
            if self.client_info.get(f"{client_id}.is_active")
        ]

        return active_clients
