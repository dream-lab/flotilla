"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import logging
import logging.config
import os


class FedLogger(object):
    def __init__(self, id: str, loggername: str = None) -> None:
        super().__init__()

        self.id: str = id
        self.loggername = loggername
        self.log_dir: str = "logs"
        if id == "0":
            log_name: str = "flotilla_server.log"
        else:
            log_name: str = f"flotilla_{self.id}.log"

        os.makedirs(os.path.join(self.log_dir), exist_ok=True)
        filepath = os.path.join(self.log_dir, log_name)
        logging.config.fileConfig(
            fname=os.path.join("config", "logger.conf"),
            defaults={"logfilename": filepath},
        )
        self._logger = logging.getLogger(self.loggername)

    def update_id(self, id: str):
        self.id = id
        if id == "0":
            log_name: str = "flotilla_server.log"
        else:
            log_name: str = f"flotilla_{self.id}.log"
        filepath = os.path.join(self.log_dir, log_name)
        logging.config.fileConfig(
            fname=os.path.join("config", "logger.conf"),
            defaults={"logfilename": filepath},
        )
        self._logger = logging.getLogger(self.loggername)

    def debug(self, event: str, msg: str) -> None:
        self._logger.debug(f"{self.id},{event},{msg}")

    def info(self, event: str, msg: str) -> None:
        self._logger.info(f"{self.id},{event},{msg}")

    def warn(self, event: str, msg: str) -> None:
        self._logger.warning(f"{self.id},{event},{msg}")

    def error(self, event: str, msg: str) -> None:
        self._logger.error(f"{self.id},{event},{msg}")

    def critical(self, event: str, msg: str) -> None:
        self._logger.critical(f"{self.id},{event},{msg}")
