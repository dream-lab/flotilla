# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto.grpc_pb2 as grpc__pb2


class EdgeServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Echo = channel.unary_unary(
                '/EdgeService/Echo',
                request_serializer=grpc__pb2.echoMessage.SerializeToString,
                response_deserializer=grpc__pb2.echoMessage.FromString,
                )
        self.InitBench = channel.unary_unary(
                '/EdgeService/InitBench',
                request_serializer=grpc__pb2.InitBenchRequest.SerializeToString,
                response_deserializer=grpc__pb2.InitBenchResponse.FromString,
                )
        self.StartTraining = channel.unary_unary(
                '/EdgeService/StartTraining',
                request_serializer=grpc__pb2.InitTrainRequest.SerializeToString,
                response_deserializer=grpc__pb2.InitTrainResponse.FromString,
                )
        self.StreamFile = channel.stream_unary(
                '/EdgeService/StreamFile',
                request_serializer=grpc__pb2.UploadFile.SerializeToString,
                response_deserializer=grpc__pb2.StringResponse.FromString,
                )
        self.StartValidation = channel.unary_unary(
                '/EdgeService/StartValidation',
                request_serializer=grpc__pb2.InitValidationRequest.SerializeToString,
                response_deserializer=grpc__pb2.InitValidationResponse.FromString,
                )


class EdgeServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Echo(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def InitBench(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartTraining(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamFile(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartValidation(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EdgeServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Echo': grpc.unary_unary_rpc_method_handler(
                    servicer.Echo,
                    request_deserializer=grpc__pb2.echoMessage.FromString,
                    response_serializer=grpc__pb2.echoMessage.SerializeToString,
            ),
            'InitBench': grpc.unary_unary_rpc_method_handler(
                    servicer.InitBench,
                    request_deserializer=grpc__pb2.InitBenchRequest.FromString,
                    response_serializer=grpc__pb2.InitBenchResponse.SerializeToString,
            ),
            'StartTraining': grpc.unary_unary_rpc_method_handler(
                    servicer.StartTraining,
                    request_deserializer=grpc__pb2.InitTrainRequest.FromString,
                    response_serializer=grpc__pb2.InitTrainResponse.SerializeToString,
            ),
            'StreamFile': grpc.stream_unary_rpc_method_handler(
                    servicer.StreamFile,
                    request_deserializer=grpc__pb2.UploadFile.FromString,
                    response_serializer=grpc__pb2.StringResponse.SerializeToString,
            ),
            'StartValidation': grpc.unary_unary_rpc_method_handler(
                    servicer.StartValidation,
                    request_deserializer=grpc__pb2.InitValidationRequest.FromString,
                    response_serializer=grpc__pb2.InitValidationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'EdgeService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class EdgeService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Echo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EdgeService/Echo',
            grpc__pb2.echoMessage.SerializeToString,
            grpc__pb2.echoMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def InitBench(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EdgeService/InitBench',
            grpc__pb2.InitBenchRequest.SerializeToString,
            grpc__pb2.InitBenchResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StartTraining(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EdgeService/StartTraining',
            grpc__pb2.InitTrainRequest.SerializeToString,
            grpc__pb2.InitTrainResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StreamFile(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/EdgeService/StreamFile',
            grpc__pb2.UploadFile.SerializeToString,
            grpc__pb2.StringResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StartValidation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EdgeService/StartValidation',
            grpc__pb2.InitValidationRequest.SerializeToString,
            grpc__pb2.InitValidationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
