import numpy as np
from sklearn.cluster import AgglomerativeClustering


def client_selection(
    selectable_clients,
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

    current_round = training_session.get(f"{session_id}.last_round_number")
    print("CURRENT_ROUND = ", current_round)
    # print(type(aggregate_state))
    if current_round == 0 and len(aggregate_state.keys()) == 0:
        try:
            num_tiers = args["num_tiers"]
        except Exception as e:
            print(f"CLIENT_SELECTION.TiFL:: Exception - {e} \nSetting num_tiers = 1")

        if len(selectable_clients) < num_tiers:
            print(
                "CLIENT_SELECTION.client_selection_tifl_lite:: More num_tiers set than available clients. Setting number of tiers equal to available clients"
            )
            num_tiers = len(selectable_clients)

        model_id = training_state.get(f"{selectable_clients[0]}.current_model_id")
        client_latencies = list()
        for client in selectable_clients:
            client_benchmark_info = client_info.get(f"{client}.benchmark_info")[
                model_id
            ]
            client_latency = (
                client_benchmark_info["time_taken_s"]
                / client_benchmark_info["num_mini_batches"]
            ) * 100
            client_latencies.append(client_latency)

        print("Client Latencies = ", client_latencies)

        reshaped_client_latencies = np.array(client_latencies).reshape(-1, 1)
        agglomerative = AgglomerativeClustering(
            n_clusters=num_tiers, metric="euclidean"
        )
        client_tiers_id = agglomerative.fit_predict(reshaped_client_latencies)
        client_to_tier_id = dict()

        for i in range(len(selectable_clients)):
            client_to_tier_id[selectable_clients[i]] = client_tiers_id[i]

        print("CLIENT_TO_TIER_ID = ", client_to_tier_id)

        selectable_client_to_tier_id = {
            c: client_to_tier_id[c] for c in selectable_clients
        }

        # print("SELECTABLE_CLIENT TO", selectable_client_to_tier_id)
        tier_ids = np.unique(list(selectable_client_to_tier_id.values()))
        client_tiers = [list() for i in tier_ids]

        for client_id in selectable_client_to_tier_id.keys():
            client_tiers[selectable_client_to_tier_id[client_id]].append(client_id)

        print("CLIENT_TIERS = ", client_tiers)

        try:
            num_clients = args["num_clients_selected_per_tier"]
        except Exception as e:
            num_clients = min([max(tier) for tier in client_tiers])
            print(
                f"CLIENT_SELECTION.TIFL:: Exception - {e} \nSetting num_clients_selected_per_tier = minimin tier size = {num_clients}"
            )

        selected_clients = []
        for i, tier in enumerate(client_tiers):
            num_clients = min(len(tier), num_clients)
            clients = np.random.choice(tier, size=num_clients, replace=False).tolist()

            print(f"SELECTED CLIENTS FROM TIER {i} = ", clients)
            selected_clients.extend(clients)
            client_selection_state.put(f"selected_clients_tier_{i}", clients)

        print("SELECTED_CLIENTS", selected_clients)

        global_model = training_session.get(f"{session_id}.global_model")
        for i in range(num_tiers):
            aggregate_state.put(f"update_count_tier_{i}", 0)
            aggregate_state.put(
                f"tier_model_tier_{i}",
                global_model,
            )
        client_selection_state.put("client_to_tier_id_dict", client_to_tier_id)
        return selected_clients, None

    else:
        client_to_tier_dict = client_selection_state.get("client_to_tier_id_dict")
        # active_client_to_tier_id = [client_to_tier_dict[c] for c in selectable_clients]

        selectable_client_to_tier_id = {
            c: client_to_tier_dict[c] for c in selectable_clients
        }
        tier_ids = np.unique(list(selectable_client_to_tier_id.values()))
        print("-------------------------- Client Selection -----------------------")
        print("AGGREGATION-STATE-IN-CS = ", aggregate_state.keys())
        print("SELECTABLLE_CLIENTS_TO_TIER_IDS = ", selectable_client_to_tier_id)
        for i in tier_ids:
            tier = client_selection_state.get(f"selected_clients_tier_{i}")
            print(f"selected clients tier {i} = ", tier)
            if len(tier) == 0:
                print("\n\nSELECTING NEW CLIENTS FOR TIER ", i)

                client_tier = [
                    c
                    for c in selectable_client_to_tier_id.keys()
                    if selectable_client_to_tier_id[c] == i
                ]

                print(f"CURRENT TIER {i} = ", client_tier)

                try:
                    num_clients = args["num_clients_selected_per_tier"]
                except Exception as e:
                    num_clients = 1
                    print(
                        f"CLIENT_SELECTION.FEDAT:: Exception - {e} \nSetting num_clients_selected_per_tier = length of tier {i} = {num_clients}"
                    )
                num_clients = min(len(client_tier), num_clients)

                selected_clients = np.random.choice(
                    client_tier, size=num_clients, replace=False
                ).tolist()

                print(
                    f"tier_id = {i}, client_tier = {client_tier}, num_clients = {num_clients}, selected_clients = {selected_clients}"
                )

                client_selection_state.put(
                    f"selected_clients_tier_{i}",
                    selected_clients,
                )
                return selected_clients, None

        return None, None
