"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import sys
from os.path import join
from pickle import dumps as p_dumps
from pickle import loads as p_loads
from time import time

from typing_extensions import OrderedDict

import proto.grpc_pb2 as grpc_pb2
import proto.grpc_pb2_grpc as grpc_pb2_grpc
from client.client import Client
from client.client_file_manager import setup_model_dir
from utils.logger import FedLogger


class ClientGRPCManager(grpc_pb2_grpc.EdgeServiceServicer):
    def __init__(
        self,
        client_id: str,
        temp_dir_path: str,
        torch_device: str,
        dataset_paths: str,
        client_info: dict,
    ) -> None:
        self.logger = FedLogger(id=client_id, loggername="CLIENT_GRPC_MANAGER")
        self.temp_dir_path = temp_dir_path
        self.client_id = client_id

        self.client = Client(
            client_id=self.client_id,
            torch_device=torch_device,
            temp_dir_path=temp_dir_path,
            dataset_paths=dataset_paths,
            client_info=client_info,
        )

    def Echo(self, request, context) -> grpc_pb2.echoMessage:
        self.logger.debug(
            "fedclient.gRPC.echo.request", f"Received message:{request.text}"
        )
        print("fedclient.gRPC.echo.request:: Complete\n")
        if context.is_active():
            return grpc_pb2.echoMessage(text=request.text)
        else:
            self.logger.error("fedclient.gRPC.echo.request", f"fedserver not active")

    def StreamFile(self, request_iterator, context) -> None:
        model_id = str()
        file_name = str()
        data = bytearray()

        try:
            self.logger.debug("fedclient.gRPC.download.model.init", "")
            for request in request_iterator:
                if request.metadata.model_id and request.metadata.file_name:
                    model_id = request.metadata.model_id
                    file_name = request.metadata.file_name
                    self.logger.debug(
                        "fedclient.gRPC.download.model.received",
                        f"{model_id},{file_name}",
                    )
                data.extend(request.chunk_data)

            setup_model_dir(temp_dir_path=self.temp_dir_path, model_id=model_id)
            file_path = join(self.temp_dir_path, "model_cache", model_id, file_name)
            with open(file_path, "wb") as f:
                f.write(data)
        except sys.excepthook:
            print("Exception at fedclient.gRPC.StreamFile::", sys.excepthook)

        return grpc_pb2.StringResponse(
            text=f"{self.client_id} successfully received {model_id}/{file_name}"
        )

    def InitBench(self, request, context) -> grpc_pb2.InitBenchResponse:
        self.logger.info("fedclient.gRPC.benchmark.init", "")
        print("fedclient.gRPC.InitBench:: Benchmark Round Initiated")

        model_id: str = request.model_id
        model_class: str = request.model_class
        model_config: dict = p_loads(request.model_config)
        dataset_id: str = request.dataset_id
        batch_size: int = request.batch_size
        learning_rate: float = request.learning_rate

        optimizer = None
        loss_function = None
        timeout_duration_s = None
        max_mini_batches = None

        if request.loss_function:
            loss_function = p_loads(request.loss_function)
        if request.optimizer:
            optimizer = p_loads(request.optimizer)
        if request.timeout_duration_s:
            timeout_duration_s = request.timeout_duration_s
        if request.max_mini_batch_count:
            max_mini_batches = request.max_mini_batch_count

        if not context.is_active():
            self.logger.error("fedclient.gRPC.InitBench", f"fedserver not active")
            return
        result = self.client.Benchmark(
            model_id=model_id,
            model_class=model_class,
            model_config=model_config,
            dataset_id=dataset_id,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_function=loss_function,
            optimizer=optimizer,
            timeout_duration_s=timeout_duration_s,
            max_mini_batches=max_mini_batches,
        )

        log_str_keys = ",".join([str(key) for key in result.keys()])
        log_str_values = ",".join([str(value) for value in result.values()])

        log_string = ",".join([log_str_keys, log_str_values])
        self.logger.info("fedclient.gRPC.benchmark.results", log_string)
        self.logger.info("fedclient.gRPC.benchmark.finish", "")

        response = grpc_pb2.InitBenchResponse(
            model_id=model_id,
            num_mini_batches=result["total_mini_batches"],
            bench_duration_s=result["time_taken_s"],
        )

        print("fedclient.gRPC.InitBench:: Benchmark Round Finished")
        if context.is_active():
            return response
        else:
            self.logger.error("fedclient.gRPC.InitBench", f"fedserver not active")

    def StartTraining(self, request, context) -> grpc_pb2.InitTrainResponse:
        self.logger.info("fedclient.gRPC.train.init", "")
        grpc_train_time = time()

        model_id: str = request.model_id
        model_class: str = request.model_class
        model_config: dict = p_loads(request.model_config)
        dataset_id: str = request.dataset_id
        model_wts: OrderedDict = p_loads(request.model_wts)
        batch_size: int = request.batch_size
        learning_rate: float = request.learning_rate
        num_epochs: int = request.num_epochs
        round_id: int = request.round_idx
        timeout_duration_s = None
        loss_function = p_loads(request.loss_function)
        optimizer = p_loads(request.optimizer)

        if request.timeout_duration_s:
            max_mini_batches = None
            timeout_duration_s = request.timeout_duration_s
            max_epochs = None
        elif request.max_epochs:
            max_mini_batches = None
            timeout_duration_s = None
            max_epochs = request.max_epochs
        else:
            max_mini_batches = request.max_mini_batches
            timeout_duration_s = None
            max_epochs = None

        self.logger.debug("fedclient.gRPC.train.round.model", model_id)
        print(f"\nfedclient.gRPC.train.round:: Training Round:{round_id}")

        if not context.is_active():
            self.logger.error("fedclient.gRPC.train", f"fedserver not active")
            return
        result, model_weights = self.client.Train(
            model_id=model_id,
            model_class=model_class,
            model_config=model_config,
            dataset_id=dataset_id,
            model_wts=model_wts,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            loss_function=loss_function,
            optimizer=optimizer,
            timeout_duration_s=timeout_duration_s,
            max_epochs=max_epochs,
            max_mini_batches=max_mini_batches,
        )

        pickle_time = time()
        model_weights = p_dumps(model_weights)
        metrics = p_dumps(result)
        self.logger.info(
            "fedclient.gRPC.train.round.pickle.weights", f"{time()-pickle_time}"
        )

        response = grpc_pb2.InitTrainResponse(
            model_id=model_id,
            model_weights=model_weights,
            client_id=self.client_id,
            round_idx=round_id,
            metrics=metrics,
        )

        self.logger.info("fedclient.gRPC.train.round.complete", "")

        response_time = time()
        try:
            if context.is_active():
                return response
            else:
                self.logger.error("fedclient.gRPC.train", f"fedserver not active")
        finally:
            print("fedclient.gRPC.StartTraining:: Training Round Finished")
            self.logger.info("fedclient.gRPC.e2e.time", f"{time()-grpc_train_time}")
            self.logger.info(
                "fedclient.gRPC.train.response.time", f"{time()-response_time}"
            )

    def StartValidation(self, request, context) -> grpc_pb2.InitValidationResponse:
        self.logger.info("fedclient.gRPC.validation.round.init", "")
        grpc_validation_time = time()

        model_id: str = request.model_id
        model_class: str = request.model_class
        model_config = p_loads(request.model_config)
        dataset_id: str = request.dataset_id
        model_wts: OrderedDict = p_loads(request.model_wts)
        batch_size: int = request.batch_size
        round_id: int = request.round_idx
        loss_function = p_loads(request.loss_function)
        optimizer = p_loads(request.optimizer)

        self.logger.debug("fedclient.gRPC.validate.round.model", model_id)
        print(f"\nfedclient.gRPC.validate.round:: Validation Round:{round_id}")

        if not context.is_active():
            self.logger.error("fedclient.gRPC.validation", f"fedserver not active")
            return
        result = self.client.Validate(
            model_id=model_id,
            model_class=model_class,
            model_config=model_config,
            dataset_id=dataset_id,
            model_wts=model_wts,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        pickle_time = time()
        metrics = p_dumps(result)
        self.logger.info(
            "fedclient.gRPC.train.round.pickle.weights", f"{time()-pickle_time}"
        )

        response = grpc_pb2.InitValidationResponse(
            model_id=model_id,
            client_id=self.client_id,
            round_idx=round_id,
            metrics=metrics,
        )

        self.logger.info("fedclient.gRPC.validation.round.complete", "")

        return_time = time()
        try:
            if context.is_active():
                return response
            else:
                self.logger.error("fedclient.gRPC.validation", f"fedserver not active")
        finally:
            print("fedclient.gRPC.StartValidation:: Validation Round Finished")
            self.logger.info(
                "fedclient.gRPC.e2e.time", f"{time()-grpc_validation_time}"
            )
            self.logger.info(
                "fedclient.gRPC.validation.response.time", f"{time()-return_time}"
            )
