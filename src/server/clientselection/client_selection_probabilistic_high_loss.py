"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math
import random

import numpy as np


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
        # selected_clients = client_list
        selected_clients = np.random.choice(
            a=client_list, size=num_clients, replace=False
        )
    else:
        latest_loss = []

        for client in client_list:
            latest_loss.append(client_session[client]["valdiation_metrics"][0])

        loss_sum = sum(latest_loss)
        client_probabilities = [item / loss_sum for item in latest_loss]

        num_clients = math.floor(len(client_list) * (percent_clients / 100))

        if num_clients > len(client_list):
            num_clients = len(client_list)

        if num_clients == 0:
            print(
                "CLIENT_SELECTION.probabilistic_high_loss:: Number of clients to be selected from a tier came to be zero. Setting value to one."
            )
            num_clients = 1

        selected_clients = np.random.choice(
            a=client_list, p=client_probabilities, size=num_clients, replace=False
        )

    return None, selected_clients
