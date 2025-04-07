python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./grpc.proto
sed -i 's/grpc_pb2/proto.grpc_pb2/1' grpc_pb2_grpc.py
