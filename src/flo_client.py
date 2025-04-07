"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import argparse
import os
import uuid

from client.client_file_manager import OpenYaML
from client.client_manager import ClientManager
from client.utils.client_info import generate_client_info
from client.utils.monitor import Monitor


def main():
    pid = os.getpid()
    client_config = OpenYaML(os.path.join("config", "client_config.yaml"))
    temp_dir_path = os.path.join(client_config["general_config"]["temp_dir_path"])
    if os.path.isfile(os.path.join(temp_dir_path, "client_info.yaml")):
        client_info = OpenYaML(os.path.join(temp_dir_path, "client_info.yaml"))
        client_id: str = client_info["client_id"]
    else:
        client_id: str = str(uuid.uuid4())
        client_info = generate_client_info(client_id, temp_dir_path)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--monitor",
        action="store_true",
        default=False,
        help="Monitor CPU/RAM/Disk/Network usage.",
    )
    args = parser.parse_args()

    if args.monitor:
        Monitor(client_id, pid)

    client = ClientManager(client_id, client_config, client_info)
    client.run()


if __name__ == "__main__":
    main()
