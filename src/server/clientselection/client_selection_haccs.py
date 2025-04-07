"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math
import random

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def client_selection(
    selectable_clients: dict,
    session_id: str,
    client_info: dict,
    training_state: dict,
    training_session: dict,
    aggregate_state: dict,
    client_selection_state: dict,
    args: dict = None,
):
    if len(selectable_clients) == 0:
        return None, None

    def get_client_clusters(num_clusters):
        client_histograms = dict()
        unique_labels = []
        for c in selectable_clients:
            client_histograms[c] = training_state.get(f"{c}.current_dataset_detail")[
                "metadata"
            ]["label_distribution"]
            unique_labels.extend(client_histograms[c].keys())

        print("CLIENT_HISTOGRAMS = ", client_histograms)
        unique_labels = list(set(unique_labels))

        client_label_histograms = []
        for c in selectable_clients:
            client_label_histograms.append(
                [
                    client_histograms[c][x] if x in client_histograms[c].keys() else 0.0
                    for x in unique_labels
                ]
            )
        print("CLIENT_LABEL_HISTOGRAMS = ", client_label_histograms)
        agglomerative = AgglomerativeClustering(
            n_clusters=num_clusters, metric="euclidean"
        )
        agglomerative.fit(client_label_histograms)
        return agglomerative.labels_

    current_round = training_session.get(f"{session_id}.last_round_number")
    print("CURRENT_ROUND = ", current_round)

    if len(aggregate_state.keys()) == 0:
        if current_round == 0:
            try:
                num_clusters = args["num_tiers"]
            except:
                num_clusters = 1
                print("CLIENT_SELECTION:: num_clusters = 1")

            cluster_labels = get_client_clusters(num_clusters=num_clusters)
            print("CLUSTER_LABELS", cluster_labels)

            client_to_cluster_dict = dict()
            for i, client_id in enumerate(selectable_clients):
                client_to_cluster_dict[client_id] = cluster_labels[i]
            print("CLIENT_TO_CLUSTER_DICT =", client_to_cluster_dict)
            client_latencies = dict()

            model_id = training_state.get(f"{selectable_clients[0]}.current_model_id")
            for client in selectable_clients:
                client_benchmark_info = client_info.get(f"{client}.benchmark_info")[
                    model_id
                ]
                client_latency = (
                    client_benchmark_info["time_taken_s"]
                    / client_benchmark_info["num_mini_batches"]
                ) * 100
                client_latencies[client] = client_latency

            print("CLIENT_LATENCIES", client_latencies)
            client_selection_state.put("client_to_cluster_dict", client_to_cluster_dict)
            client_selection_state.put("client_latencies", client_latencies)

            client_selection_state.put("selected_clients", selectable_clients)

            return selectable_clients, None

        try:
            client_fraction = args["client_fraction"]
        except Exception as e:
            client_fraction = 0.1
            print(
                f"CLIENT_SELECTION.HACCS:: Exception - {e} \nSetting percentage_client_selected_per_round = 100"
            )
        try:
            loss_latency_tradeoff = args["loss_latency_tradeoff_param"]
        except Exception as e:
            loss_latency_tradeoff = 0.5
            print(
                f"CLIENT_SELECTION.HACCS:: Exception - {e} \nSetting loss_latency_tradeoff_param = 0.5"
            )

        num_clients = max(1, int(client_fraction * len(selectable_clients)))

        client_to_cluster_dict = client_selection_state.get("client_to_cluster_dict")

        selectable_client_to_cluster_ids = {
            c: client_to_cluster_dict[c] for c in selectable_clients
        }

        cluster_ids = np.unique(list(selectable_client_to_cluster_ids.values()))

        # client_clusters = [
        #     list(selectable_clients[selectable_client_to_cluster_ids == id])
        #     for id in np.unique(selectable_client_to_cluster_ids)
        # ]
        client_clusters = [list() for i in cluster_ids]
        for client_id in selectable_client_to_cluster_ids:
            client_clusters[selectable_client_to_cluster_ids[client_id]].append(
                client_id
            )

        num_clusters = len(client_clusters)

        client_latencies = client_selection_state.get("client_latencies")
        print("CLIENT_LATENCIES = ", client_latencies)
        print("CLIENT_CLUSTERS = ", client_clusters)

        cluster_loss = []

        for cluster in client_clusters:
            avg_loss = 0.0
            for client in cluster:
                # print(client)
                # print(training_state.get(f"{client}.training_metrics")[training_state.get(f"{client}.last_round_participated")])
                avg_loss += training_state.get(f"{client}.training_metrics")[
                    training_state.get(f"{client}.last_round_participated")
                ]["loss"]
            cluster_loss.append(avg_loss / len(cluster))

        print("CLUSTER_LOSS = ", cluster_loss)
        cluster_latency = []
        for cluster in client_clusters:
            cluster_latency.append(
                max([client_latencies[client] for client in cluster])
            )
        print("CLUSTER_LATENCIES = ", cluster_latency)
        sorted_clusters_by_latencies = []
        for cluster in client_clusters:
            sorted_clusters_by_latencies.append(
                [
                    k
                    for k, v in sorted(
                        client_latencies.items(), key=lambda item: item[1]
                    )
                    if k in cluster
                ]
            )

        print("SORTED CLUSTERS = ", sorted_clusters_by_latencies)

        max_cluster_latency = max(cluster_latency)
        reduction_in_latency = [1 - x / max_cluster_latency for x in cluster_latency]

        sum_cluster_loss = sum(cluster_loss)
        # fmt:off
        cluster_weights = [
            (loss_latency_tradeoff * reduction_in_latency[i])
            + (1 - loss_latency_tradeoff) * cluster_loss[i] / sum_cluster_loss
            for i in range(num_clusters)
        ]
        # fmt:on
        print("CLUSTER_WEIGHTS = ", cluster_weights)

        cluster_choices = random.choices(
            range(num_clusters), weights=cluster_weights, k=num_clients
        )
        print(cluster_choices)

        selected_clients = []
        for cluster in cluster_choices:
            try:
                selected_clients.append(client_clusters[cluster].pop(0))
            except:
                print(
                    "WARNING!! No clients left in cluster to choose. Skipping choice."
                )

        client_selection_state.put("selected_clients", selected_clients)

        return selected_clients, None
    else:
        return None, None
