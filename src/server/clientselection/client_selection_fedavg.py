"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math

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
    print("CLIENT SELECTION CALLED!")
    if len(aggregate_state.keys()) == 0:
        C = args["client_fraction"]
        M = max(1, int(C * len(selectable_clients)))
        rng = np.random.default_rng()
        if len(selectable_clients) >= M:
            selected_clients = rng.choice(
                a=selectable_clients, size=M, replace=False
            ).tolist()
        else:
            return None, None
        client_selection_state.put("selected_clients", selected_clients)
        return selected_clients, None
    else:
        return None, None
