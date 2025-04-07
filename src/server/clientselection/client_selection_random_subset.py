"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np

from utils.logger import FedLogger


def client_selection(
    active_clients: list,
    session_id: str,
    client_info: dict,
    training_state: dict,
    training_session: dict,
    aggregate_state: dict,
    client_selection_state: dict,
    args: dict = None,
):
    logger = FedLogger(id=session_id, loggername="CLIENT_SELECTION")
    print(len(aggregate_state.keys()), "AGG_STATE")
    if (
        training_session.get(f"{session_id}.last_round_number") == 0
        or len(aggregate_state.keys()) == 0
    ):
        C = args["client_fraction"]
        M = max(1, int(C * len(active_clients)))
        rng = np.random.default_rng()
        selected_clients = rng.choice(a=active_clients, size=M, replace=False)
        # selected_clients = active_clients
        client_selection_state.put(f"{session_id}.selected_clients", selected_clients)
        print(selected_clients)
        return selected_clients, None
    else:
        return None, None
