python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./client.proto
sed -i 's/client_pb2/proto.client_pb2/1' client_pb2_grpc.py
