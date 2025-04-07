from collections import OrderedDict
from typing import OrderedDict

import numpy as np
from torch import zeros

from utils.logger import FedLogger


def aggregate(
    session_id,
    client_id,
    client_active,
    client_local_weights,
    client_info,
    training_state,
    training_session,
    aggregator_state,
    client_selection_state,
    args,
):
    logger = FedLogger(id=session_id, loggername="AGGREGATOR")
    print("CALLING FEDAVG")
    print("CLIENT ACTIVE", client_active)
    print("AGGREGATOR STATE", aggregator_state.keys())
    if client_active:
        aggregator_state.put(f"{client_id}.client_local_weights", client_local_weights)

    finished_clients = aggregator_state.keys()
    print("FINISHED CLIENTS", finished_clients)

    active_clients = [
        c for c in client_info.keys() if client_info.get(f"{c}.is_active")
    ]

    selected_clients = client_selection_state.get("selected_clients")

    if client_active == False:
        try:
            selected_clients.remove(client_id)
            print(selected_clients)
            client_selection_state.put(f"selected_clients", selected_clients)
        except Exception as e:
            print("EXCEPTION E", e)

    clients_to_wait_for = [c for c in selected_clients if c in active_clients]

    if len(finished_clients) > 0 and all(
        c in finished_clients for c in clients_to_wait_for
    ):
        try:
            print("AGGREGATOR:: Aggregating clients - ", finished_clients)
            N = 0
            global_model = OrderedDict()
            temp_model = aggregator_state.get(
                f"{finished_clients[0]}.client_local_weights"
            )

            for layer in temp_model:
                shape = temp_model[layer].shape
                global_model[layer] = zeros(shape)

            del temp_model

            client_weights = list()

            N_k = np.array(list())
            for client_id in finished_clients:
                N_k = np.append(
                    N_k,
                    training_state.get(f"{client_id}.current_dataset_detail")[
                        "metadata"
                    ]["num_items"],
                )
                client_weights.append(
                    aggregator_state.get(f"{client_id}.client_local_weights")
                )

            N_k = N_k / sum(N_k)
            print("N_k", N_k)

            for i, weights in enumerate(client_weights):
                for layer in weights.keys():
                    # fmt: off
                    global_model[layer] += (weights[layer] * N_k[i])
                    # fmt: on

            aggregator_state.clear()
            print("RETURNING AGGREGATED MODEL")
            return global_model
        except Exception as e:
            aggregator_state.clear()
            print("AGGREGATOR.FEDAVG:: EXCEPTION = ", e)
            return None
    else:
        return None
