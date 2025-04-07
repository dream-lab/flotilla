import asyncio
import io
import os
import pickle
import sys
import tarfile
from time import time

import grpc
from torch import device as torch_device
from torch.cuda import is_available
from typing_extensions import OrderedDict

import proto.grpc_pb2 as grpc_pb2
import proto.grpc_pb2_grpc as grpc_pb2_grpc
from server.load_aggregator import load_aggregator
from server.load_client_selection import load_client_selection
from server.server_file_manager import (
    OpenYaML,
    get_available_datasets,
    get_model_dir_hash,
)
from server.server_model_manager import ServerModelManager
from server.server_state_manager import StateManager
from utils.logger import FedLogger
from utils.plot import Plot


class FloSessionManager:
    def __init__(
        self,
        id,
        client_info,
        mqtt_init_event,
        server_config,
        session_config,
        restore,
        revive,
        file,
    ) -> None:
        self.id = id
        self.logger = FedLogger(id=self.id, loggername="SESSION_MANAGER")

        self.client_info = client_info
        self.mqtt_init_finish_event = mqtt_init_event

        validation_data_dir_path = server_config["validation_data_dir_path"]
        self.dataset_available = get_available_datasets(validation_data_dir_path)

        try:
            self.state = server_config["state"]
            self.state_location = self.state["state_location"]
            self.state_hostname = self.state["state_hostname"]
            self.state_port = self.state["state_port"]
        except KeyError:
            self.state_location = "inmemory"
            self.state_hostname = None
            self.state_port = None

        grpc_max_message_length: int = server_config["comm_config"]["grpc"][
            "max_message_length"
        ]
        self.grpc_opts: list = [
            ("grpc.max_send_message_lenght", grpc_max_message_length),
            ("grpc.max_receive_message_length", grpc_max_message_length),
            ("grpc.so_reuseport", 0),
            ("grpc.so_readdr", 0),
            ("grpc.enable_http_proxy", 0),
        ]
        self.grpc_timeout: int = server_config["comm_config"]["grpc"]["timeout_s"]
        self.grpc_chunk_size: int = server_config["comm_config"]["grpc"][
            "chunk_size_bytes"
        ]

        self.temp_dir_path: str = server_config["temp_dir_path"]
        self.checkpoint_dir_path: str = server_config["checkpoint_dir_path"]
        self.session_dir_path: str = os.path.join(self.temp_dir_path, self.id)

        self.training_session = StateManager(
            loc=self.state_location,
            name="training_session",
            host=self.state_hostname,
            port=self.state_port,
            state_id=self.id,
        )
        self.training_state = StateManager(
            loc=self.state_location,
            name="training_state",
            host=self.state_hostname,
            port=self.state_port,
            state_id=self.id,
        )
        self.client_selection_state = StateManager(
            loc=self.state_location,
            name="client_selection_state",
            host=self.state_hostname,
            port=self.state_port,
            state_id=self.id,
        )
        self.aggregator_state = StateManager(
            loc=self.state_location,
            name="aggregator_state",
            host=self.state_hostname,
            port=self.state_port,
            state_id=self.id,
        )

        if restore or revive:
            if file:
                session_config = self.restore_from_file(restore, revive)
            else:
                session_config = self.restore(restore, revive)
        else:
            results = dict()
            self.training_session.put(f"{self.id}.session_config", session_config)
            self.training_session.put(f"{self.id}.status", "running")
            self.training_session.put(f"{self.id}.last_round_number", 0)
            self.training_session.put(f"{self.id}.global_validation_metrics", results)

        if session_config["session_config"]["use_gpu"]:
            self.torch_device = torch_device("cuda" if is_available() else "cpu")
            self.logger.info("flo_session.torch_device", self.torch_device)
        else:
            self.torch_device = torch_device("cpu")
            self.logger.info("flo_session.torch_device", self.torch_device)

        self.aggregator = session_config["session_config"]["aggregator"]
        self.aggregator_args = session_config["session_config"]["aggregator_args"]
        self.aggregate = load_aggregator(self.id, self.aggregator).aggregate
        self.client_selection_strategy = session_config["session_config"][
            "client_selection"
        ]
        self.client_selection_args: dict = session_config["session_config"][
            "client_selection_args"
        ]
        self.client_selection = load_client_selection(
            self.id, self.client_selection_strategy
        ).client_selection

        self.checkpoint_interval = (
            session_config["session_config"]["checkpoint_interval"]
            if session_config["session_config"]["checkpoint_interval"]
            else None
        )

        self.server_validation_interval = session_config["session_config"][
            "validation_round_interval"
        ]
        self.generate_plots = session_config["session_config"]["generate_plots"]
        if self.generate_plots:
            plotting_obj = Plot(self.id)

        self.bench_config: dict = session_config["benchmark_config"]
        try:
            self.skip_bench: bool = session_config["benchmark_config"]["skip_benchmark"]
            if self.skip_bench:
                print("\n\n\nSKIPPING BENCHMARK\n\n\n")
        except KeyError:
            self.skip_bench = False

        self.train_config: dict = (
            session_config["server_training_config"]
            | session_config["client_training_config"]
        )

        self.model_config: dict = session_config["model_config"]
        if not self.model_config:
            model_config_path: str = os.path.join(
                self.train_config["model_dir"], "config.yaml"
            )
            self.model_config: dict = OpenYaML(model_config_path, self.logger)[
                "default_training_config"
            ]

        self.model_util = ServerModelManager(
            id=self.id,
            torch_device=self.torch_device,
            model_dir=self.train_config["model_dir"],
            model_class=self.train_config["model_class"],
            batch_size=self.train_config["global_model_validation_batch_size"],
            val_data_path=self.dataset_available[self.train_config["dataset"]][
                "dataset_details"
            ]["data_filename"],
            use_custom_dataloader=self.model_config["use_custom_dataloader"],
            custom_dataloader_args=self.model_config["custom_loader_args"],
            use_custom_validator=self.model_config["use_custom_validator"],
            custom_validator_args=self.model_config["custom_validator_args"],
            model_args=self.model_config["model_args"],
        )

        self.model_util.set_loss_fun(
            self.train_config["loss_function"],
            self.train_config["loss_function_custom"],
        )
        self.model_util.set_optimizer(
            self.train_config["learning_rate"],
            self.train_config["optimizer"],
            self.train_config["optimizer_custom"],
        )
        if restore or revive or file:
            self.model_util.set_model_weights(
                self.training_session.get(f"{self.id}.global_model")
            )
        else:
            print("INITIATING RANDOM MODEL")
            self.training_session.put(
                f"{self.id}.global_model", self.model_util.get_model_weights()
            )

    def restore(self, restore, revive):
        print("RECIEVED RESTORE FLAG")
        session_config = self.training_session.get(f"{self.id}.session_config")
        active_clients = self.get_active_clients()
        session_clients = self.training_state.keys()
        print(
            "TRAINING_ROUND::",
            self.training_session.get(f"{self.id}.last_round_number"),
        )
        restore_check = True if set(active_clients) == set(session_clients) else False
        if restore and restore_check:
            print("RESTORING SESSION")
            pass
        elif revive:
            self.training_state.clear()
            self.client_selection_state.clear()
            self.aggregator_state.clear()
            print("REVIVING SESSION")
        else:
            raise Exception("Session can't continue")
        return session_config

    def restore_from_file(self, restore, revive):
        print("RECIEVED RESTORE FLAG and FILE FLAG")
        self.training_session.clear()
        self.training_state.clear()
        self.client_selection_state.clear()
        self.aggregator_state.clear()
        active_clients = self.get_active_clients()
        with tarfile.open(
            os.path.join(self.checkpoint_dir_path, f"checkpoint_{self.id}.tar")
        ) as tf:
            training_session_bytearray = tf.extractfile(
                f"training_session_{self.id}"
            ).read()
            self.training_session.putall(pickle.loads(training_session_bytearray))
            training_state_bytearray = tf.extractfile(
                f"training_state_{self.id}"
            ).read()
            self.training_state.putall(pickle.loads(training_state_bytearray))
            session_clients = self.training_state.keys()
            session_config = self.training_session.get(f"{self.id}.session_config")

            restore_check = (
                True if set(active_clients) == set(session_clients) else False
            )
            if restore and restore_check:
                print("RESTORING FROM FILE")
                client_selection_bytearray = tf.extractfile(
                    f"client_selection_state_{self.id}"
                ).read()
                self.client_selection_state.putall(
                    pickle.loads(client_selection_bytearray)
                )
                aggregator_state_bytearray = tf.extractfile(
                    f"aggregator_state_{self.id}"
                ).read()
                self.aggregator_state.putall(pickle.loads(aggregator_state_bytearray))
            elif revive:
                print("REVIVING")
                self.training_state.clear()
            else:
                raise Exception("Session can't continue")
            return session_config

    async def start_session(self):
        self.logger.debug("session_id", str(self.id))
        self.mqtt_init_finish_event.wait()
        await self.echo()

        active_clients = self.get_active_clients()
        for client in active_clients:
            self.client_info.put(f"{client}.is_training", False)
        print(
            f"session_manager.run:::active_clients:{active_clients}\t{len(active_clients)}"
        )

        await self.train()

        results = self.training_session.get(f"{self.id}.global_validation_metrics")
        for key in results:
            self.logger.info(
                f"session.train.{key}", ",".join([str(x) for x in results[key]])
            )
        self.logger.info(
            "fedserver_session_finished_running", f"{self.id}.finished_running"
        )
        return

    async def grpc_echo(self, client_id: str) -> None:
        """
        Asynchronous function that implements a gRPC echo functionality
        with a Flotilla client
        """

        start_time = time()

        self.logger.info("fedserver_gRPC.echo.start", f"connecting_to,{client_id}")
        grpc_ep = self.client_info.get(f"{client_id}.grpc_ep")
        channel = grpc.aio.insecure_channel(f"{grpc_ep}", self.grpc_opts)

        try:
            stub = grpc_pb2_grpc.EdgeServiceStub(channel)
            response = await stub.Echo(
                grpc_pb2.echoMessage(text=f"{self.id}"), timeout=self.grpc_timeout
            )
        except AttributeError:
            self.logger.error("fedserver_gRPC.echo.invalid_channel", f"{client_id}")
            response = None
        except grpc.RpcError:
            self.logger.error("fedserver_gRPC.echo.timeout", f"timed_out,{client_id}")
            self.training_state.put(
                f"{client_id}.missed_deadline", (time(), self.grpc_timeout)
            )
            response = None

        if response:
            self.logger.info(
                "fedserver_gRPC.echo.response", f"client_replied,{client_id}"
            )
            self.training_state.put(f"{client_id}.missed_deadline", None)
        self.logger.info(
            "fedserver_gRPC.echo.client.finished",
            f"client_id - time_taken,{client_id},{time()-start_time}",
        )

    async def echo(self) -> None:
        start_time = time()
        self.logger.info(
            "fedserver_gRPC.echo.init", f"num_of_clients,{self.client_info.len()}"
        )

        await asyncio.gather(
            *(self.grpc_echo(client_id) for client_id in self.get_active_clients())
        )
        self.logger.info(
            "fedserver_gRPC.echo.finished", f"time_taken,{time()-start_time}"
        )

    async def grpc_send_model(
        self, client_id: str, model_id: str, model_hash, path: str
    ):
        """
        Asynchronous function that sends that sends all files passed to the
        varible "path" to the client with ID "client_id"
        """

        start = time()
        try:
            SEND_MODEL = True
            models_on_client: dict = self.client_info.get(f"{client_id}.models")
            for c_model_id, c_model_hash in models_on_client.items():
                if model_id == c_model_id:
                    if model_hash == c_model_hash:
                        SEND_MODEL = False
                    else:
                        SEND_MODEL = True
            if SEND_MODEL:
                print(f"SENDING MODEL {model_id} to client {client_id}")
                grpc_ep = self.client_info.get(f"{client_id}.grpc_ep")
                channel = grpc.aio.insecure_channel(f"{grpc_ep}", self.grpc_opts)
                stub = grpc_pb2_grpc.EdgeServiceStub(channel)
                if os.path.isdir(path):
                    for f in os.scandir(path):
                        if os.path.isfile(f.path):
                            response = await stub.StreamFile(
                                self.stream_file_chunk(model_id=model_id, path=f.path),
                                timeout=self.grpc_timeout,
                            )
                            self.logger.info(
                                "fedserver_gRPC.send_model.cache_miss",
                                f"{client_id},{response}",
                            )
                            models_on_client[model_id] = model_hash
                            self.client_info.put(
                                f"{client_id}.models", models_on_client
                            )
                else:
                    self.logger.error(
                        "fedserver_gRPC.send_model.invaid.path",
                        f"path doesn't exist,{path}",
                    )
            else:
                self.logger.debug(
                    "fedserver_gRPC.send_model.cache_hit",
                    f"Client:{client_id} has model {model_id},{client_id},{model_id}",
                )
        except sys.excepthook:
            self.logger.error(str(sys.excepthook), str(client_id))
        except Exception as e:
            self.logger.error("fedserver_gRPC.send_model.timeout", str(client_id))
            response = None

        self.logger.info(
            "fedserver_gRPC.send_model.client.finished",
            f"client_id - time_taken,{client_id},{time()-start}",
        )

    async def send_model(self, model_id: str, path: str, clients: list):
        """
        Asynchronous function that sends model with model ID passed to the
        function through the argument "model_id" to all active clients.
        """

        model_hash = get_model_dir_hash(path)
        start = time()
        self.logger.info("fedserver_gRPC.send_model.init", "")
        await asyncio.gather(
            *(
                self.grpc_send_model(client_id, model_id, model_hash, path)
                for client_id in clients
            )
        )
        self.logger.info(
            "fedserver_gRPC.send_model.finished", f"time_taken,{time()-start}"
        )

    async def grpc_benchmark(
        self,
        client_id: str,
        model_id: str,
        model_class: str,
        model_hash: str,
        dataset_id: str,
        batch_size: int,
        learning_rate: float,
        timeout_duration_s: float,
    ) -> None:
        """
        Asynchronous function that initiates a benchmark round with the client with whose ID
        is passed to it as an argument. The function then receives the
        output of the benchmark and updates client_info[client_id]["benchmark"]
        """
        start_time = time()
        try:
            self.client_info.put(f"{client_id}.is_training", True)
            self.logger.info("fedserver_gRPC.bench.connect", f"{client_id}")
            grpc_ep = self.client_info.get(f"{client_id}.grpc_ep")
            channel = grpc.aio.insecure_channel(f"{grpc_ep}", self.grpc_opts)
            stub = grpc_pb2_grpc.EdgeServiceStub(channel)

            self.logger.debug(
                "fedserver_gRPC.bench.config",
                f"model_id-model_class-dataset_id-batch_size-learning_rate-time_out are,{model_id},{model_class},{batch_size},{learning_rate},{timeout_duration_s}",
            )

            self.logger.info("fedserver_gRPC.bench.await.response", f"{client_id}")

            model_config = pickle.dumps(self.model_config)

            response_time = time()
            response = await stub.InitBench(
                grpc_pb2.InitBenchRequest(
                    model_id=model_id,
                    model_class=model_class,
                    model_config=model_config,
                    dataset_id=dataset_id,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    timeout_duration_s=timeout_duration_s,
                ),
                timeout=self.grpc_timeout,
            )

            self.logger.info(
                "fedserver_gRPC.bench.await.response_time",
                f"{client_id},{time()-response_time}",
            )

        except AttributeError as error:
            self.logger.error("fedserver_gRPC.bench.invalid_channel", f"{client_id}")
            response = None
            print(error)

        except grpc.RpcError as error:
            self.logger.error(
                "fedserver_gRPC.bench.connection_terminated", f"{client_id}"
            )
            response = None
            print(error)

        if response:
            self.client_info.put(
                f"{client_id}.benchmark_info",
                {
                    model_id: {
                        "time_taken_s": response.bench_duration_s,
                        "num_mini_batches": response.num_mini_batches,
                        "model_hash": model_hash,
                    }
                },
            )

            self.logger.debug(
                "fedserver_gRPC.bench.results",
                f"client_id-bench_time-mini_batches,{client_id},{response.bench_duration_s},{response.num_mini_batches}",
            )
            print(
                f"fedserver_gRPC.bench.results::",
                self.client_info.get(f"{client_id}.client_name"),
                ":",
                self.client_info.get(f"{client_id}.benchmark_info"),
                sep="",
            )
        else:
            self.logger.warn("fedserver_gRPC.bench.client_dropped", f"{client_id}")

        self.logger.info(
            "fedserver_gRPC.bench.client.finished",
            f"client_id and time,{client_id},{time()-start_time}",
        )

        self.client_info.put(f"{client_id}.is_training", False)

    async def benchmark(self, clients):
        """
        Asynchronous function that initiates a benchmark round for all active
        clients.
        """
        self.logger.info("fedserver_gRPC.bench.init", "")
        start_time = time()

        model_id = self.bench_config["model_id"]
        model_dir = self.bench_config["model_dir"]
        model_class = self.bench_config["model_class"]
        dataset_id = self.bench_config["dataset"]
        batch_size = self.bench_config["batch_size"]
        learning_rate = self.bench_config["learning_rate"]
        timeout_duration_s = self.bench_config["timeout_duration_s"]
        model_hash = get_model_dir_hash(model_dir)

        await self.send_model(model_id=model_id, path=model_dir, clients=clients)

        await asyncio.gather(
            *(
                self.grpc_benchmark(
                    client_id=client_id,
                    model_id=model_id,
                    model_class=model_class,
                    model_hash=model_hash,
                    dataset_id=dataset_id,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    timeout_duration_s=timeout_duration_s,
                )
                for client_id in clients
            )
        )
        self.logger.info(
            "fedserver_gRPC.bench.finished", f"time_taken,{time()-start_time}"
        )

    async def async_grpc_train(
        self,
        client_id: str,
        session_id: str,
        model_id: str,
        model_class: str,
        model_wts: OrderedDict,
        dataset_id: str,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        round_no: int,
        timeout_duration_s: float,
        loss,
        optimizer,
        model_updated_event,
        model_updated_condition,
    ) -> None:
        """
        Asynchronous function that initiates a training round of round number "round_no"
        with whose ID is passed to it as the argument "client_id".
        """
        train_start_time = time()
        self.logger.info("fedserver_gRPC.train.connect", f"connecting to,{client_id}")
        try:
            grpc_ep = self.client_info.get(f"{client_id}.grpc_ep")
            channel = grpc.aio.insecure_channel(f"{grpc_ep}", self.grpc_opts)
            stub = grpc_pb2_grpc.EdgeServiceStub(channel)

            self.logger.info("fedserver_gRPC.train.await.response", f"{client_id}")
            model_config = pickle.dumps(self.model_config)
            weights_time = time()
            serialized_model_wts: bytes = pickle.dumps(model_wts)
            self.logger.info(
                "fedserver_gRPC.train.round.model_wts.pickle.time",
                f"client_id-round_no-time_taken,{client_id},{round_no},{time() - weights_time}",
            )

            loss_time = time()
            serialized_loss_fun: bytes = pickle.dumps(loss)
            self.logger.info(
                "fedserver_gRPC.train.round.loss_function.pickle.time",
                f"client_id - round_no - time_taken,{client_id},{round_no},{time() - loss_time}",
            )

            optimizer_time = time()
            serialized_optimizer: bytes = pickle.dumps(optimizer)
            self.logger.info(
                "fedserver_gRPC.train.round.optimizer.pickle.time",
                f"client_id - round_no - time_taken,{client_id},{round_no},{time() - optimizer_time}",
            )

            response_time = time()
            response = await stub.StartTraining(
                grpc_pb2.InitTrainRequest(
                    session_id=session_id,
                    model_id=model_id,
                    model_class=model_class,
                    model_config=model_config,
                    model_wts=serialized_model_wts,
                    dataset_id=dataset_id,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    round_idx=round_no,
                    timeout_duration_s=timeout_duration_s,
                    loss_function=serialized_loss_fun,
                    optimizer=serialized_optimizer,
                ),
                timeout=self.grpc_timeout,
            )

            self.logger.info(
                "fedserver_gRPC.train.round.await.response_time",
                f"{client_id},{round_no},{time()-response_time}",
            )

            self.logger.info(
                "fedserver_gRPC.train.round.client.finished",
                f"client_id-round_no-time_taken,{client_id},{round_no},{time() - train_start_time}",
            )

        except AttributeError as e:
            self.logger.error(
                "fedserver_gRPC.train.connection_terminated", f"{client_id},{e}"
            )
            response = None
        except grpc.RpcError as e:
            self.logger.error(
                "fedserver_gRPC.train.invalid_channel", f"{client_id},{e}"
            )
            print(e)
            response = None
        finally:
            await model_updated_condition.acquire()
            self.client_info.put(f"{client_id}.is_training", False)
            print("BEFORE TRAIN CALLBACK")
            round_no = self.grpc_train_callback(
                client_id=client_id,
                start_time=train_start_time,
                response=response,
            )
            print("AFTER TRAIN CALLBACK")
            print("ROUND NO = ", round_no)
            model_updated_event.set()
            print(model_updated_event)

    def grpc_train_callback(self, client_id, start_time, response):
        if response:
            metrics = pickle.loads(response.metrics)
            local_model_wts = pickle.loads(response.model_weights)
            round_no = response.round_idx

            log_str_keys = "-".join(metrics.keys())
            log_str_vals = "-".join([str(x) for x in metrics.values()])

            log_str = ",".join(
                [str(round_no), str(client_id), log_str_keys, log_str_vals]
            )

            print(f"{client_id} has returned {log_str_keys} and {log_str_vals}")

            self.logger.info(
                "fedserver.train.round.results",
                log_str,
            )
            self.logger.info(
                "fedserver.train.round.client.finished",
                f"client_id-round_no-time_taken,{client_id},{round_no},{time()-start_time}",
            )

            self.training_state.put(f"{client_id}.last_round_participated", round_no)
            self.training_state.put(f"{client_id}.weights", local_model_wts)

            training_metrics = self.training_state.get(f"{client_id}.training_metrics")
            if training_metrics is None:
                self.training_state.put(
                    f"{client_id}.training_metrics", {round_no: metrics}
                )
            else:
                training_metrics[round_no] = metrics
                self.training_state.put(
                    f"{client_id}.training_metrics", training_metrics
                )

            aggregate_start_time = time()
            aggregated_model = self.aggregate(
                session_id=self.id,
                client_id=client_id,
                client_active=True,
                client_local_weights=local_model_wts,
                client_info=self.client_info,
                training_state=self.training_state,
                training_session=self.training_session,
                aggregator_state=self.aggregator_state,
                client_selection_state=self.client_selection_state,
                args=self.aggregator_args,
            )
        elif response == None:
            self.logger.warn("fedserver.train.client_dropped", f"{client_id}")
            print("CLIENT DIED")
            print(client_id, " TRAIN RESPONSE EMPTY")
            round_no = int(self.training_session.get(f"{self.id}.last_round_number"))
            aggregated_model = self.aggregate(
                session_id=self.id,
                client_id=client_id,
                client_active=False,
                client_local_weights=None,
                client_info=self.client_info,
                training_state=self.training_state,
                training_session=self.training_session,
                aggregator_state=self.aggregator_state,
                client_selection_state=self.client_selection_state,
                args=self.aggregator_args,
            )
            print("AGGREGATED MODEL FROM RESPONSE NONE = ", aggregated_model)

        if aggregated_model:
            print("GOT AGGREGATED MODEL", client_id)
            aggregate_end_time = time() - aggregate_start_time
            round_no = int(self.training_session.get(f"{self.id}.last_round_number"))
            self.training_session.put(f"{self.id}.global_model", aggregated_model)
            self.model_util.set_model_weights(aggregated_model)
            if round_no % self.server_validation_interval == 0:
                server_validation_time = time()
                global_validation_metrics = self.model_util.validate_model(
                    round_no=round_no
                )
                self.logger.info(
                    "fedserver.train_callback.server_validation_time",
                    f"{time()-server_validation_time}",
                )
                results = self.training_session.get(
                    f"{self.id}.global_validation_metrics"
                )
                for key in global_validation_metrics.keys():
                    if key in results:
                        results[key].append(global_validation_metrics[key])
                    else:
                        results[key] = [global_validation_metrics[key]]
                self.training_session.put(
                    f"{self.id}.global_validation_metrics", results
                )

                self.logger.info(
                    "fedserver.train_callback",
                    f"round_no,{','.join(list(global_validation_metrics.keys()))},{round_no+1},{','.join([str(i) for i in global_validation_metrics.values()])}",
                )
                self.logger.info(
                    "fedserver.train_callback.aggregate_time",
                    f"{round_no},{aggregate_end_time}",
                )
            self.training_session.put(f"{self.id}.last_round_number", round_no + 1)

            if (
                self.checkpoint_interval
                and (round_no + 1) % self.checkpoint_interval == 0
            ):
                self.checkpoint(round_no)

            self.logger.info(
                "fedserver.train.server_round_time",
                f"{round_no},{time()-self.round_start_time}",
            )
            self.round_start_time = time()
            print("RETURNING ROUND RESULTS", client_id)
        print("ROUND NO = ", round_no)
        return round_no

    def checkpoint(self, round_no):
        checkpoint_start_time = time()
        print("CHECKPOINTING", round_no)
        training_session_bytearray = pickle.dumps(self.training_session.getall())
        training_session_source_file = io.BytesIO(
            initial_bytes=training_session_bytearray
        )
        training_session_info = tarfile.TarInfo(f"training_session_{self.id}")
        training_session_info.size = len(training_session_bytearray)

        training_state_bytearray = pickle.dumps(self.training_state.getall())
        training_state_source_file = io.BytesIO(initial_bytes=training_state_bytearray)
        training_state_info = tarfile.TarInfo(f"training_state_{self.id}")
        training_state_info.size = len(training_state_bytearray)

        client_selection_state_bytearray = pickle.dumps(
            self.client_selection_state.getall()
        )
        client_selection_state_source_file = io.BytesIO(
            initial_bytes=client_selection_state_bytearray
        )
        client_selection_state_info = tarfile.TarInfo(
            f"client_selection_state_{self.id}"
        )
        client_selection_state_info.size = len(client_selection_state_bytearray)

        aggregator_state_bytearray = pickle.dumps(self.aggregator_state.getall())
        aggregator_state_source_file = io.BytesIO(
            initial_bytes=aggregator_state_bytearray
        )
        aggregator_state_info = tarfile.TarInfo(f"aggregator_state_{self.id}")
        aggregator_state_info.size = len(aggregator_state_bytearray)

        os.makedirs(self.checkpoint_dir_path, exist_ok=True)
        with tarfile.open(
            os.path.join(self.checkpoint_dir_path, f"checkpoint_{self.id}.tar"), "w"
        ) as tf:
            tf.addfile(training_session_info, training_session_source_file)
            tf.addfile(training_state_info, training_state_source_file)
            tf.addfile(client_selection_state_info, client_selection_state_source_file)
            tf.addfile(aggregator_state_info, aggregator_state_source_file)

        self.logger.info(
            "fedserver.train.checkpoint", f"{round_no},{time()-checkpoint_start_time}"
        )

    async def async_grpc_validation(
        self,
        client_id: str,
        session_id: str,
        model_id: str,
        model_class: str,
        dataset_id: str,
        model_wts: OrderedDict,
        batch_size: int,
        round_no: int,
        loss,
        optimizer,
        model_updated_event,
        model_updated_condition,
    ) -> None:
        """
        Asynchronous function that initiates a validation round of round number "round_no"
        with clients whose ID is passed to it as the argument "client_id".
        """
        validation_start_time = time()

        self.logger.info(
            "fedserver_gRPC.validation.connect", f"connecting to,{client_id}"
        )
        try:
            grpc_ep = self.client_info.get(f"{client_id}.grpc_ep")
            channel = grpc.aio.insecure_channel(f"{grpc_ep}", self.grpc_opts)
            stub = grpc_pb2_grpc.EdgeServiceStub(channel)

            self.logger.info("fedserver_gRPC.validation.await.response", f"{client_id}")

            weights_time = time()
            serialized_model_wts: bytes = pickle.dumps(model_wts)
            self.logger.info(
                "fedserver_gRPC.validation.round.loss_function.pickle.time",
                f"client_id - round_no - time_taken,{client_id},{round_no},{time() - weights_time}",
            )

            loss_time = time()
            serialized_loss_fun: bytes = pickle.dumps(loss)
            self.logger.info(
                "fedserver_gRPC.validation.round.loss_function.pickle.time",
                f"client_id - round_no - time_taken,{client_id},{round_no},{time() - loss_time}",
            )

            optimizer_time = time()
            serialized_optimizer: bytes = pickle.dumps(optimizer)
            self.logger.info(
                "fedserver_gRPC.train.round.optimizer.pickle.time",
                f"client_id - round_no - time_taken,{client_id},{round_no},{time() - optimizer_time}",
            )

            response_time = time()
            response = await stub.StartValidation(
                grpc_pb2.InitValidationRequest(
                    session_id=session_id,
                    model_id=model_id,
                    model_class=model_class,
                    model_config=pickle.dumps(self.model_config),
                    dataset_id=dataset_id,
                    model_wts=serialized_model_wts,
                    batch_size=batch_size,
                    round_idx=round_no,
                    loss_function=serialized_loss_fun,
                    optimizer=serialized_optimizer,
                ),
                timeout=self.grpc_timeout,
            )

            self.logger.info(
                "fedserver_gRPC.validation.round.await.response_time",
                f"{client_id},{round_no},{time()-response_time}",
            )

            self.logger.info(
                "fedserver_gRPC.validation.round.client.finished",
                f"client_id - round_no - time_taken,{client_id},{round_no},{time() - validation_start_time}",
            )

        except AttributeError as e:
            self.logger.error(
                "fedserver_gRPC.validation.connection_terminated", f"{client_id}"
            )
            response = None

        except grpc.RpcError as e:
            self.logger.error(
                "fedserver_gRPC.validation.invalid_channel", f"{client_id}"
            )
            print("SERVER_MANAGER.grpc_validation:: Error = ", e)
            response = None

        finally:
            await model_updated_condition.acquire()
            self.client_info.put(f"{client_id}.is_training", False)
            self.grpc_validation_callback(
                client_id=client_id,
                round_no=round_no,
                start_time=validation_start_time,
                response=response,
            )
            model_updated_event.set()
            print(model_updated_condition, model_updated_event)

    def grpc_validation_callback(self, client_id, round_no, start_time, response):
        if not response:
            print(client_id, " VALIDATION RESPONSE EMPTY")
            return
        metrics = pickle.loads(response.metrics)

        log_str_keys = "-".join(metrics.keys())
        log_str_vals = "-".join([str(x) for x in metrics.values()])

        log_str = ",".join([str(round_no), str(client_id), log_str_keys, log_str_vals])

        print(f"Validation::{client_id} returned {log_str_keys}, {log_str_vals}")

        self.logger.info(
            "fedserver_gRPC.validation.round.results",
            log_str,
        )
        self.logger.info(
            "fedserver_gRPC.validation.round.client.finished",
            f"client_id-round_no-time_taken,{client_id},{round_no},{time()-start_time}",
        )

        self.client_info.put(f"{client_id}.is_training", False)
        try:
            res = self.training_state.get(f"{client_id}.validation_metrics")
            res[round_no] = metrics
            self.training_state.put(f"{client_id}.validation_metrics", res)
        except Exception as e:
            self.training_state.put(
                f"{client_id}.validation_metrics", {round_no: metrics}
            )

    async def train(self):
        start_time = time()

        model_id: str = self.train_config["model_id"]
        bench_model_id: str = self.bench_config["model_id"]
        model_dir: str = self.train_config["model_dir"]
        bench_model_dir: str = self.bench_config["model_dir"]
        model_class: str = self.train_config["model_class"]
        dataset_id: str = self.train_config["dataset"]
        training_rounds: int = self.train_config["num_training_rounds"]
        epochs: int = self.train_config["epochs"]
        batch_size: int = self.train_config["batch_size"]
        loss_fun: str = self.train_config["loss_function"]
        optimizer: str = self.train_config["optimizer"]
        lr: float = self.train_config["learning_rate"]
        timeout: float = self.train_config["train_timeout_duration_s"]
        bench_model_hash: str = get_model_dir_hash(bench_model_dir)

        self.logger.info("fedserver_gRPC.train_session.init", str(start_time))
        self.logger.info(
            "fedserver_gRPC.train_session.config",
            f"model-dataset_id-training_rounds-epochs-batch_size-loss_fun-optimizer-lr-timeout-benchmark-model-aggregator-aggregator_args-client_selection-client_selection_args,{model_class},{dataset_id},{training_rounds},{epochs},{batch_size},{loss_fun},{optimizer},{lr},{timeout},{bench_model_id},{self.aggregator},{self.aggregator_args},{self.client_selection_strategy},{self.client_selection_args}",
        )

        self.logger.info("fedserver_gRPC.train.rounds", str(training_rounds))

        for client in self.training_state.keys():
            self.training_state.put(f"{client}.current_dataset", dataset_id)
            data_distribution = self.client_info.get(f"{client}.dataset_details")
            self.training_state.put(
                f"{client}.current_dataset_detail", data_distribution[dataset_id]
            )
            self.training_state.put(f"{client}.current_model_id", model_id)

        model_updated_condition = asyncio.Condition()
        model_updated_event = asyncio.Event()
        model_updated_event.set()
        await model_updated_condition.acquire()
        print(model_updated_condition)
        self.round_start_time = time()
        while (
            self.training_session.get(f"{self.id}.last_round_number") < training_rounds
        ):
            await model_updated_event.wait()
            if self.skip_bench == False:
                benchmark_overhead_time = time()
                benchmark_clients = list()
                for client in self.get_active_clients():
                    benchmark_info = self.client_info.get(f"{client}.benchmark_info")
                    # print(f"BENCHMARK INFO FOR CLIENT {client} = ", benchmark_info)
                    if bench_model_id not in benchmark_info.keys() or (
                        benchmark_info[bench_model_id]
                        and benchmark_info[bench_model_id]["model_hash"]
                        != bench_model_hash
                    ):
                        benchmark_clients.append(client)

                # print("CLIENTS TO BENCHMARK = ", benchmark_clients)
                if len(benchmark_clients) > 0:
                    await self.benchmark(benchmark_clients)
                self.logger.info(
                    "train.benchmark_overhead.time", f"{time()-benchmark_overhead_time}"
                )

            candidate_clients = [
                client
                for client in self.client_info.keys()
                if self.client_info.get(f"{client}.is_active")
                and not self.client_info.get(f"{client}.is_training")
            ]
            print("IN WHILE LOOP = candidate clients = ", candidate_clients)
            client_selection_time = time()
            training_clients, validation_clients = self.client_selection(
                selectable_clients=candidate_clients,
                session_id=self.id,
                client_info=self.client_info,
                training_state=self.training_state,
                training_session=self.training_session,
                aggregate_state=self.aggregator_state,
                client_selection_state=self.client_selection_state,
                args=self.client_selection_args,
            )
            print(
                f"IN WHILE LOOP clients selected = {training_clients}, validation clients = {validation_clients}"
            )
            self.logger.info(
                "train.client_selection.time_taken", f"{time()-client_selection_time}"
            )

            training_clients = (
                set(training_clients) if training_clients is not None else set()
            )
            validation_clients = (
                set(validation_clients) if validation_clients is not None else set()
            )
            for client in training_clients.union(validation_clients):
                self.client_info.put(f"{client}.is_training", True)

            assert len(training_clients.intersection(validation_clients)) == 0

            currently_training_clients = [
                client
                for client in self.client_info.keys()
                if self.client_info.get(f"{client}.is_training")
            ]

            print(f"CURRENTLY TRAINING CLIENTS::{currently_training_clients}")

            if training_clients and len(training_clients) > 0:
                model_wts = self.model_util.get_model_weights()
                loss = self.model_util.get_loss_fun()
                optimizer = self.model_util.get_optimizer()
                round_no = self.training_session.get(f"{self.id}.last_round_number")
                self.logger.debug(
                    "fedserver_gRPC.train.round.init",
                    f"round_no-num_clients-clients,{round_no},{len(training_clients)},{','.join([str(x) for x in training_clients])}",
                )
                await self.send_model(model_id, model_dir, training_clients)
                asyncio.gather(
                    *(
                        self.async_grpc_train(
                            client_id=client_id,
                            session_id=self.id,
                            model_id=model_id,
                            model_class=model_class,
                            model_wts=model_wts,
                            dataset_id=dataset_id,
                            batch_size=batch_size,
                            learning_rate=lr,
                            num_epochs=epochs,
                            round_no=round_no,
                            timeout_duration_s=timeout,
                            loss=loss,
                            optimizer=optimizer,
                            model_updated_event=model_updated_event,
                            model_updated_condition=model_updated_condition,
                        )
                        for client_id in training_clients
                    )
                )

            if validation_clients and len(validation_clients) > 0:
                model_wts = self.model_util.get_model_weights()
                loss = self.model_util.get_loss_fun()
                optimizer = self.model_util.get_optimizer()
                round_no = self.training_session.get(f"{self.id}.last_round_number")
                self.logger.debug(
                    "fedserver_gRPC.validation.round.init",
                    f"round_no-num_clients-clients,{round_no},{len(validation_clients)},{','.join([str(x) for x in validation_clients])}",
                )
                await self.send_model(model_id, model_dir, validation_clients)
                asyncio.gather(
                    *(
                        self.async_grpc_validation(
                            client_id=client_id,
                            session_id=self.id,
                            model_id=model_id,
                            model_class=model_class,
                            model_wts=model_wts,
                            dataset_id=dataset_id,
                            batch_size=batch_size,
                            round_no=round_no,
                            loss=loss,
                            optimizer=optimizer,
                            model_updated_event=model_updated_event,
                            model_updated_condition=model_updated_condition,
                        )
                        for client_id in validation_clients
                    )
                )

            model_updated_event.clear()
            model_updated_condition.release()

        self.logger.info("fedserver.session.loop_runtime", f"{time()-start_time}")
        print(f"Training Ends.")
        return

    def get_active_clients(self):
        active_clients = [
            client_id
            for client_id in self.client_info.keys()
            if self.client_info.get(f"{client_id}.is_active")
        ]

        return active_clients

    def stream_file_chunk(self, model_id, path):
        filename = path.split(os.sep)[-1]
        try:
            metadata = grpc_pb2.MetaData(model_id=model_id, file_name=filename)
            yield grpc_pb2.UploadFile(metadata=metadata)
            with open(path, mode="rb") as f:
                while True:
                    chunk = f.read(self.grpc_chunk_size)
                    if chunk:
                        yield grpc_pb2.UploadFile(chunk_data=chunk)
                    else:
                        return
        except sys.excepthook:
            print(sys.excepthook)

    def exit_procedure(self, mqtt_stop_event, mqtt_task):
        self.logger.info("fedserver.keyboard_interrupt", "received keyboard interrupt")
        mqtt_stop_event.set()
        mqtt_task.join()
