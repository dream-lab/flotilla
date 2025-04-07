# Utils Module

The `utils` module in the client directory contains two Python files, `ip.py` and `port_allocator.py`, which provide helpful functions for handling IP addresses and port allocation for the gRPC communication configuration. Below is a detailed explanation of each file:

## 1. [ip.py](ip.py)

This file, `ip.py`, contains a Python function named `get_ip_address`. The function is responsible for obtaining the IP address of the client machine. This is useful when configuring communication settings that require the client's IP address, such as gRPC communication.

## 2. [port_allocator.py](port_allocator.py)

This file, `port_allocator.py`, contains a Python function named `port_allocator`. The function is designed to check the availability of a specific port for gRPC communication. If the specified port is unavailable (in use), the function returns an alternative port that is free and can be used for gRPC communication.

# Note

These utility functions in the `utils` module provide essential support for configuring communication settings in the client, making it easier to manage IP address retrieval and port allocation for gRPC communication. The module's purpose is to handle network-related functionalities and assist the client application in communicating with other components efficiently.

