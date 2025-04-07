"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np


def client_selection(
    selectable_clients: list,
    session_id: str,
    client_info: dict,
    training_state: dict,
    training_session: dict,
    aggregate_state: dict,
    client_selection_state: dict,
    args: dict = None,
):
    current_round = training_session.get(f"{session_id}.last_round_number")
    rng = np.random.default_rng()
    print("IN CLIENT SELECT")

    client_fraction = args["client_fraction"]
    print("CLIENT FRACTION", client_fraction)
    if current_round == 0:
        num_clients = max(1, int(client_fraction * len(selectable_clients)))
        selected_clients = rng.choice(
            a=selectable_clients, size=num_clients, replace=False
        )
        for client in selected_clients:
            client_selection_state.put(f"{client}", current_round)

        print("EXITING CLIENT SELECT")
        return selected_clients, None
    else:
        print("SELECTABLE CLIENTS", selectable_clients)
        selected_client = rng.choice(a=selectable_clients, size=1, replace=False)
        print("CS_BEFORE PUT CLIENT_SELECTION_STATE = ", client_selection_state.keys())
        client_selection_state.put(f"{selected_client[0]}", current_round)
        print("CS_END CLIENT_SELECTION_STATE = ", client_selection_state.keys())
        return selected_client, None
