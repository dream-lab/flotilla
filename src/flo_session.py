"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import argparse
import pprint
import sys
from uuid import uuid4

import requests
import yaml

parser = argparse.ArgumentParser(
    description="Submit a job to a Federated Learning Server"
)

# Argument for the path to the configuration file (required)
parser.add_argument(
    "config_path", type=str, help="Path to the Federated Learning configuration file"
)
parser.add_argument(
    "--file",
    action="store_true",
    default=False,
    help="If file flag is provided server will try to restore from a checkpoint file.",
)

parser.add_argument(
    "--restore",
    action="store_true",
    default=False,
    help="Restore session by providing the session id. If both restore and revive flags are provided, the server will try to first restore the session if all clients from the previous session are still online, otherwise revive",
)

parser.add_argument(
    "--revive",
    action="store_true",
    default=False,
    help="Revive session by providing the session id. If both restore and revive flags are provided, the server will try to first restore the session if all clients from the previous session are still online, otherwise revive",
)


if "--restore" in sys.argv or "--revive" in sys.argv or "--file" in sys.argv:
    parser.add_argument(
        "--session_id",
        type=str,
        required=True,
        help="The id of the session to be restored/revived",
    )

parser.add_argument(
    "--federated_server_endpoint",
    type=str,
    default="10.24.24.31:12345",
    help="Address of the Federated Learning Server",
)

args = parser.parse_args()
api_url = f"http://{args.federated_server_endpoint}/execute_command"

federated_learning_config = dict()
federated_learning_config["session_id"] = str(uuid4())

with open(args.config_path) as file:
    federated_learning_config["federated_learning_config"] = yaml.safe_load(file)
    pass

if args.restore:
    federated_learning_config["session_id"] = args.session_id
    federated_learning_config["restore"] = True
else:
    federated_learning_config["restore"] = False
if args.revive:
    federated_learning_config["session_id"] = args.session_id
    federated_learning_config["revive"] = True
else:
    federated_learning_config["revive"] = False
if args.file:
    federated_learning_config["session_id"] = args.session_id
    federated_learning_config["file"] = True
else:
    federated_learning_config["file"] = False

# Send a POST request with the dictionary as JSON data
try:
    pprint.pprint(federated_learning_config)
    response = requests.post(api_url, json=federated_learning_config)
    if response.status_code == 200:
        print("Request was successful!")
        print("Response JSON:", response.json())
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response JSON:", response.json())
except requests.exceptions.ConnectionError:
    print("Server closed connection without response")
