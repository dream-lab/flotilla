"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

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
    print("TIERED")
    try:
        num_tiers = args["num_tiers"]
        percent_clients = args["percentage_client_selection"]
    except KeyError:
        num_tiers = 1
        percent_clients = 100

    if round_no == 1:
        client_tiers = dict()
        client_tiers["next_tier"] = 0
        selected_clients = client_list
    else:
        benchmark_metrics = list()

        for client in client_list:
            benchmark_metrics.append(
                client_state[client]["benchmark"]["num_mini_batches"]
            )

        sorted_benchmark_metrics_index = np.argsort(benchmark_metrics)
        sorted_benchmark_metrics = np.sort(benchmark_metrics)

        sorted_clients = [client_list[i] for i in sorted_benchmark_metrics_index]

        benchmark_data = np.array(benchmark_metrics).reshape(-1, 1)
        print(benchmark_data, benchmark_metrics)
        agglomerative = AgglomerativeClustering(
            n_clusters=num_tiers, metric="euclidean"
        )
        agglomerative_tiers = agglomerative.fit_predict(benchmark_data)
        print(agglomerative_tiers)

        tiers = dict()
        for tier_id, client_id in zip(agglomerative_tiers, client_list):
            if tier_id not in tiers:
                tiers[tier_id] = []
            tiers[tier_id].append(client_id)

        client_tiers["tiers"] = tiers

        chosen_tier = client_tiers["next_tier"]
        print(f"TIERED_RR:: client_tiers={client_tiers}, type={type(client_tiers)}")
        num_clients = int(
            (len(client_tiers["tiers"][chosen_tier]) * percent_clients) // 100
        )
        if num_clients == 0:
            num_clients = 1

        if num_clients > len(client_tiers["tiers"][chosen_tier]):
            num_clients = len(client_tiers["tiers"][chosen_tier])

        selected_clients = client_tiers["tiers"][chosen_tier][:num_clients]
        client_tiers["next_tier"] = (chosen_tier + 1) % num_tiers

    return client_tiers, selected_clients
