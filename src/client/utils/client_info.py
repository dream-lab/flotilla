import os

import yaml


def generate_client_info(client_id, path):
    os.makedirs(path, exist_ok=True)
    client_info_path = os.path.join(path, "client_info.yaml")
    client_info: dict = {"client_id": client_id, "benchmark_info": dict()}
    with open(client_info_path, "w") as file:
        yaml.dump(client_info, file)
    return client_info
