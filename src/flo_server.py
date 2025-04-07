"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import asyncio
from argparse import ArgumentParser
from os import getpid
from threading import Event
from uuid import uuid4

from flask import Flask, jsonify, request
from waitress import serve

from server.server_file_manager import OpenYaML
from server.server_manager import FlotillaServerManager
from utils.monitor import Monitor

app = Flask("flo_server")


process_id: int = getpid()
session_running = Event()
server_config = OpenYaML("./config/server_config.yaml")

parser = ArgumentParser()
parser.add_argument(
    "--monitor",
    action="store_true",
    default=False,
    help="Monitor CPU/RAM/Disk/Network IO",
)
args = parser.parse_args()
is_monitoring = args.monitor
if is_monitoring:
    monitor = Monitor("0", process_id)


def handle_request(
    session_id,
    session_config,
    restore=False,
    revive=False,
    file=False,
):
    session_running.set()
    if is_monitoring:
        monitor.set_session(session_id)
    print("Starting Session:", session_id)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            asyncio.gather(
                flo_server.run(session_id, session_config, restore, revive, file)
            )
        )
    except asyncio.CancelledError:
        print("Session Cancelled")
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt")
    except Exception as e:
        print(e)
        print("Exception in Gather loop")
    finally:
        session_running.clear()
        if is_monitoring:
            monitor.reset_session()
        return session_id


@app.route("/execute_command", methods=["POST"])
def execute_command():
    data = request.get_json()
    if session_running.is_set():
        print("Session Already Running")
        return jsonify({"message": "A session is already running"}), 400
    elif len(flo_server.get_active_clients()) == 0:
        print("No active clients")
        return jsonify({"message": "No active clients"}), 400
    elif data and "federated_learning_config" in data:
        session_config = data["federated_learning_config"]

        if (data["file"] or data["restore"] or data["revive"]) and data["session_id"]:
            restore_session_id = data["session_id"]
            session_id = handle_request(
                session_id=restore_session_id,
                session_config=session_config,
                restore=data["restore"],
                revive=data["revive"],
                file=data["file"],
            )
        elif data["session_id"]:
            session_id = data["session_id"]
            handle_request(session_id=session_id,session_config=session_config)

        return jsonify({"message": f"Session {session_id} finished"}), 200
    else:
        print("Received Invalid Request")
        return jsonify({"message": "Invalid request"}), 400


def main():
    print("Starting FLo_Server")
    global flo_server
    flo_server = FlotillaServerManager(server_config)
    serve(
        app,
        host=server_config["comm_config"]["restful"]["rest_hostname"],
        port=server_config["comm_config"]["restful"]["rest_port"],
    )


if __name__ == "__main__":
    main()
