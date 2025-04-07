"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""
import copy
import math
import random

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def client_selection(
    logger_id: str,
    client_list: list,
    client_tiers,
    client_info,
    round_no: int,
    client_state: dict = None,
    client_session: dict = None,
    global_model_metrics: dict = None,
    args: dict = None,
):
    try:
        percent_clients = args["percentage_client_selection"]
        num_tiers = args["num_tiers"]
    except Exception as e:
        num_tiers = 1
        percent_clients = 100.0
        print(
            f"CLIENT_SELECTION.HACCS_LITE:: Exception - {e} \nSetting num_tiers = 1, percentage_client_selection = 100"
        )

    num_clients = math.floor(len(client_list) * (percent_clients / 100))
    dataset_id = client_session["current_dataset"]

    if num_clients == 0:
        num_clients = 1
        print(
            "CLIENT_SELECTION.HACCS_LITE:: num_clients was calculated to be zero. num_clients set to 1."
        )
    if num_clients > len(client_list):
        num_clients = len(client_list)
        print(
            "CLIENT_SELECTION.HACCS_LITE:: num_clients > total available clients. num_clients = total available clients."
        )

    client_histograms = dict()
    unique_labels = []
    for c in client_list:
        client_histograms[c] = client_state[c][dataset_id]["metadata"][
            "label_distribution"
        ]
        unique_labels.extend(client_histograms[c].keys())
    unique_labels = list(set(unique_labels))

    client_label_histograms = []
    for c in client_list:
        client_label_histograms.append(
            [
                client_histograms[c][x] if x in client_histograms[c].keys() else 0.0
                for x in unique_labels
            ]
        )

    # kmeans = KMeans(n_clusters=num_tiers, random_state=0, n_init="auto")
    agglomerative = AgglomerativeClustering(n_clusters=num_tiers, metric="euclidean")
    agglomerative_labels = agglomerative.fit_predict(client_label_histograms)
    cluster_labels = agglomerative_labels
    client_list = np.array(client_list)
    client_tiers = [list(client_list[cluster_labels == x]) for x in range(num_tiers)]

    print("CLIENT_SELECTION.HACCS_LITE:: client_tiers - ", client_tiers)

    selected_clients = []

    remaining = num_clients

    while remaining > 0:
        for i in range(num_tiers):
            if len(client_tiers[i]) > 0:
                c = np.random.choice(client_tiers[i], size=1, replace=False)[0]
                selected_clients.append(c)
                client_tiers[i].remove(c)
                remaining -= 1

            if remaining == 0:
                break

    print("CLIENT_SELECTION.HACCS_LITE:: selected clients - ", selected_clients)

    return client_tiers, selected_clients
