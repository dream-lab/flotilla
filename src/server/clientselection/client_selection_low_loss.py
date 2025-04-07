"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math

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

    if round_no == 1:
        selected_clients = client_list

    else:
        latest_loss = dict()

        for client in client_list:
            latest_loss[client] = (
                client_session[client]["validation_metrics"][1]["loss"]
                if client_session[client]["validation_metrics"][0]
                >= client_session[client]["last_round_participated"]
                else client_session[client]["metrics"]["loss"]
            )

        num_clients = math.floor(len(client_list) * (percent_clients / 100))

        if num_clients > len(client_list):
            num_clients = len(client_list)

        if num_clients == 0:
            print(
                "CLIENT_SELECTION.low_loss:: Number of clients to be selected from a tier came to be zero. Setting value to one."
            )
            num_clients = 1

        sorted_clients = sorted(latest_loss.items(), lambda x: x[1])
        selected_clients = sorted_clients.keys()[:num_clients]

    return None, selected_clients
