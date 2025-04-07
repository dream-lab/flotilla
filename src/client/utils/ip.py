import socket


def get_ip_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def get_ip_address_docker() -> str:
    return socket.gethostbyname(socket.gethostname())
