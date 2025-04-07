import socket


def port_allocator(ip: str, grpc_port: int) -> int:
    if is_port_in_use(ip, grpc_port):
        while is_port_in_use(ip, grpc_port) and grpc_port < 65535:
            grpc_port += 1
    return grpc_port


def is_port_in_use(ip, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((ip, port)) == 0
