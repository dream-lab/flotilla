"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering

np.random.seed()


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

    if len(selectable_clients) == 0:
        return None, None

    if len(aggregate_state.keys()) == 0:
        try:
            val_interval = args["validation_round_interval"]
        except Exception as e:
            val_interval = 1
            print(
                f"CLIENT_SELECTION.TIFL:: Error {e}. Setting validation_round_interval = 1."
            )

        if current_round == 0 or (current_round + val_interval) % val_interval == 0:
            if current_round == 0:
                if client_selection_state.get("val_ongoing") is None:
                    client_selection_state.put("val_ongoing", False)
            print(
                "\n\n\nClient Validation:::", client_selection_state.get("val_ongoing")
            )
            if client_selection_state.get("val_ongoing") == False:
                print("Validation for round ", current_round, " starts")
                client_selection_state.put("val_ongoing", True)
                client_selection_state.put("client_ids_validating", selectable_clients)
                return None, selectable_clients
            else:
                # print("selectable_clients = ", selectable_clients)
                # print(
                #     "clients_validating = ",
                #     client_selection_state.get("client_ids_validating"),
                # )
                active_clients = [
                    c for c in client_info.keys() if client_info.get(f"{c}.is_active")
                ]

                clients_to_wait_for = [
                    c
                    for c in client_selection_state.get("client_ids_validating")
                    if c in active_clients
                ]
                if all([c in selectable_clients for c in clients_to_wait_for]):
                    client_selection_state.put("val_ongoing", False)
                    latest_loss = dict()
                    for client in selectable_clients:
                        latest_loss[client] = training_state.get(
                            f"{client}.validation_metrics"
                        )[current_round]["loss"]
                    client_selection_state.put("client_validation_losses", latest_loss)
                    print("Validation for round ", current_round, " ends")

        if client_selection_state.get("val_ongoing") == False:
            try:
                num_clients = args["num_clients"]
            except Exception as e:
                num_clients = 1
                print(
                    f"CLIENT_SELECTION.TIFL:: Error {e}. Setting client_fraction = 1."
                )

            if current_round == 0:
                try:
                    num_tiers = args["num_tiers"]
                except Exception as e:
                    num_tiers = 1
                    print(f"CLIENT_SELECTION.TIFL:: Error {e}. Setting num_tiers = 1.")

                try:
                    credits_per_tier = args["credits_per_tier"]
                except Exception as e:
                    credits_per_tier = 10
                    print(
                        f"CLIENT_SELECTION.TIFL:: Error {e}. Setting credits_per_tier = 10."
                    )

                model_id = training_state.get(
                    f"{selectable_clients[0]}.current_model_id"
                )

                print("Current model id = ", model_id)

                client_latencies = list()
                for client in selectable_clients:
                    print(
                        "Benchmark info = ", client_info.get(f"{client}.benchmark_info")
                    )
                    client_benchmark_info = client_info.get(f"{client}.benchmark_info")[
                        model_id
                    ]
                    client_latency = (
                        client_benchmark_info["time_taken_s"]
                        / client_benchmark_info["num_mini_batches"]
                    ) * 100
                    client_latencies.append(client_latency)

                print("CLIENT LATENCIES = ", client_latencies)

                reshaped_client_latencies = np.array(client_latencies).reshape(-1, 1)
                agglomerative = AgglomerativeClustering(
                    n_clusters=num_tiers, metric="euclidean"
                )
                client_tiers = agglomerative.fit_predict(reshaped_client_latencies)

                client_to_tier_id = dict()

                for i in range(len(selectable_clients)):
                    client_to_tier_id[selectable_clients[i]] = client_tiers[i]

                print(f"CLIENT_SELECTION:: client_tiers = ", client_to_tier_id)

                # tier_credits = {
                #     t_id: credits_per_tier for t_id in np.unique(client_tiers)
                # }
                print("CREDITS_PER_TIER = ", credits_per_tier)
                for i in range(num_tiers):
                    client_selection_state.put(f"tier_{i}_credits", credits_per_tier)

                # client_selection_state.put("tier_credits_dict", tier_credits)
                client_selection_state.put("client_to_tier_id_dict", client_to_tier_id)

            client_to_tier_dict = client_selection_state.get("client_to_tier_id_dict")
            latest_loss = client_selection_state.get("client_validation_losses")

            selectable_client_to_tier_id_dict = {
                c: client_to_tier_dict[c] for c in selectable_clients
            }

            selectable_tier_ids_list = np.unique(
                list(selectable_client_to_tier_id_dict.values())
            )
            selectable_num_tiers = len(selectable_tier_ids_list)

            selectable_tier_credits = [
                client_selection_state.get(f"tier_{x}_credits")
                for x in selectable_tier_ids_list
            ]

            tier_ids = [
                selectable_tier_ids_list[i]
                for i in range(selectable_num_tiers)
                if selectable_tier_credits[i] != 0
            ]

            current_num_tiers = len(tier_ids)

            if current_num_tiers == 0:
                print(f"CLIENT_SELECTION.TIFL:: No tier has credits left!")
                return None, None

            tier_credits = [selectable_tier_credits[x] for x in tier_ids]

            print("TIER = ", tier_ids)
            print("BEFORE SELECT TIER CREDITS = ", tier_credits)

            selectable_clients_to_tier_id_w_credits_dict = {
                c: selectable_client_to_tier_id_dict[c]
                for c in selectable_client_to_tier_id_dict.keys()
                if selectable_client_to_tier_id_dict[c] in tier_ids
            }

            print(
                "SELECTABLE CLIENTS WITH CREDITS = ",
                selectable_clients_to_tier_id_w_credits_dict,
                len(selectable_clients_to_tier_id_w_credits_dict),
            )

            client_tiers = [list() for i in tier_ids]
            for client_id in selectable_clients_to_tier_id_w_credits_dict:
                client_tiers[
                    tier_ids.index(
                        selectable_clients_to_tier_id_w_credits_dict[client_id]
                    )
                ].append(client_id)

            tier_avg_loss = list()

            print("CLIENT TIERS = ", client_tiers)
            for i, tier in enumerate(client_tiers):
                loss_sum = 0.0
                for _, client in enumerate(tier):
                    loss_sum += latest_loss[client]
                assert len(client_tiers[i]) != 0  # asserting that a tier is not empty
                tier_avg_loss.append(loss_sum / len(client_tiers[i]))

            sorted_tier_index = (-np.array(tier_avg_loss)).argsort()
            print("TIER AVG LOSS = ", tier_avg_loss)
            print("SORTED_TIER_INDEX = ", sorted_tier_index)
            tier_probs = list()

            if current_num_tiers > 1:
                D = current_num_tiers * (current_num_tiers - 1) // 2
                for i in range(1, current_num_tiers + 1):
                    tier_probs.append((current_num_tiers - i) / D)

                chosen_tier = None

                print("TIER PROBS = ", tier_probs)
                while True:
                    chosen_tier = tier_ids[
                        np.random.choice(a=sorted_tier_index, p=tier_probs, size=1)[0]
                    ]
                    print(f"Chosen tier for round {current_round} is = {chosen_tier}")
                    if tier_credits[chosen_tier] > 0:
                        break
            else:
                chosen_tier = tier_ids[0]

            num_clients = min(
                len(client_tiers[tier_ids.index(chosen_tier)]), num_clients
            )

            selected_clients = np.random.choice(
                client_tiers[tier_ids.index(chosen_tier)],
                size=num_clients,
                replace=False,
            ).tolist()

            # print("SELECTED CLIENTS = ", selectable_clients)

            client_selection_state.put(
                f"tier_{chosen_tier}_credits",
                tier_credits[tier_ids.index(chosen_tier)] - 1,
            )

            tier_credits = [
                client_selection_state.get(f"tier_{x}_credits") for x in tier_ids
            ]

            print("AFTER SELECT TIER CREDITS = ", tier_credits)
            # print(f"Client_to_tier_map = ", client_to_tier_dict)
            # print("Tier_ids", tier_ids)
            # print("Tier_credits", tier_credits_dict)
            # print(f"CLIENT_SELECTION:: actual_tier = {tier_ids[chosen_tier]} chosen_tier = {chosen_tier}, selected_clients = {selected_clients}")
            client_selection_state.put("selected_clients", selected_clients)
            return selected_clients, None

    return None, None
