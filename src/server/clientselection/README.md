This directory contains all available client selection strategies. Each `client_selection_<scheme>.py` file implements a function `client_selection()` function and returns two items, the first being a dictionary `client_tiers` and the second being a list of selected clients for the current round. The `client_tiers` dictionary is passed back to the client selection module by the Server as is. 

In order to implement a custom client selection strategy, the user must use the same function definition. The custom function must return a list of selected clients.

Server\load_client_selection.py would select the client selection method from here.

Please follow this naming convension to add a new client selection scheme.
    
    client_selection_<scheme>.py

To select a client selection scheme, please navigate to config/[training_config.yaml](..%2F..%2Fconfig%2Ftraining_config.yaml)
and update 

    train_config:
        client_selection: scheme
