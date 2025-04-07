"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math
import random

import yaml


def client_selection(
    logger_id: str,
    client_list: list,
    client_tiers,
    round_no: int,
    client_state: dict = None,
    client_session: dict = None,
    global_model_metrics: dict = None,
    args: dict = None,
):
    percent_clients = args["percentage_client_selection"]
    clients = len(client_list)
    num_clients = math.floor(clients * (percent_clients / 100))

    for client in client_list:
        if round_no % 2 == 0:
            if client % 2 == 0:
                selected_clients.append(client)
        else:
            selected_clients.append(client)

    selected_clients_number = len(selected_clients)

    if num_clients > selected_clients_number:
        num_clients = selected_clients_number
    if num_clients == 0:
        print(
            "CLIENT_SELECTION.odd_even:: Number of clients to be selected from a tier came to be zero. Setting value to one."
        )
        num_clients = 1

    selected_clients = random.sample(selected_clients, num_clients)
    return None, selected_clients
