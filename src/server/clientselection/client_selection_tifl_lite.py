"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""
import math

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from Utils.logger import FedLogger


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
    logger = FedLogger(logger_id, loggername="CLIENT_SELECTION")
    percent_clients = args["percentage_client_selection"]
    try:
        num_tiers = args["num_tiers"]
    except KeyError:
        num_tiers = 1

    if round_no == 1:
        client_tiers = list()
        selected_clients = client_list
    else:
        benchmark_metrics = list()

        for client in client_list:
            benchmark_metrics.append(
                client_state[client]["benchmark"]["num_mini_batches"]
            )

        sorted_benchmark_metrics_index = np.argsort(benchmark_metrics)
        sorted_benchmark_metrics = np.sort(benchmark_metrics).reshape(-1, 1)

        sorted_clients = [client_list[i] for i in sorted_benchmark_metrics_index]

        if len(client_list) < num_tiers:
            print(
                "CLIENT_SELECTION.client_selection_tifl_lite:: More num_tiers set than available clients. Setting number of tiers equal to available clients"
            )
            num_tiers = len(client_list)

        agglomerative = AgglomerativeClustering(
            n_clusters=num_tiers, metric="euclidean"
        )

        agglomerative_tiers = agglomerative.fit_predict(sorted_benchmark_metrics)
        client_tiers = [list() for i in range(num_tiers)]
        for tier_id, client_id in zip(agglomerative_tiers, sorted_clients):
            client_tiers[tier_id].append(client_id)

        print(f"\nTIFL_LITE:: client_tiers={client_tiers}, type={type(client_tiers)}")

        latest_loss = dict()

        for client in client_list:
            latest_loss.append(client_session[client]["validation_metrics"][0])

        tier_avg_loss = list()
        for i, tier in enumerate(client_tiers):
            loss_sum = 0.0
            for _, client in enumerate(tier):
                loss_sum += latest_loss[client]
            assert len(client_tiers[i]) != 0  # asserting that a tier is not empty
            tier_avg_loss.append(loss_sum / len(client_tiers[i]))

        sorted_tier_index = (-np.array(tier_avg_loss)).argsort()

        tier_probabilities = list()

        if num_tiers > 1:
            D = num_tiers * (num_tiers - 1) // 2
            for i in range(1, num_tiers + 1):
                tier_probabilities.append((num_tiers - i) / D)
        else:
            tier_probabilities.append(1)
        print(f"TIFL_LITE:: sorted_tiers: {sorted_tier_index}")
        print(f"TIFL_LITE:: tier_probabilities:{tier_probabilities}")

        chosen_tier = np.random.choice(sorted_tier_index, p=tier_probabilities, size=1)[
            0
        ]
        print(f"TIFL_LITE:: chosen_tier={chosen_tier}")
        logger.info(f"fedserver.ClientSelect.TIFL_lite.chosen_tier", f"{chosen_tier}")

        num_clients = math.floor(
            len(client_tiers[chosen_tier]) * (percent_clients / 100)
        )

        if num_clients > len(client_tiers[chosen_tier]):
            num_clients = len(client_tiers[chosen_tier])
        if num_clients == 0:
            print(
                "CLIENT_SELECTION.tifl_lite:: Number of clients to be selected from a tier came to be zero. Setting value to one."
            )
            num_clients = 1

        selected_clients = np.random.choice(
            client_tiers[chosen_tier], size=num_clients, replace=False
        )

    return client_tiers, selected_clients
