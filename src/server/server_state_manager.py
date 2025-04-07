from importlib import import_module
from uuid import uuid4

from utils.logger import FedLogger


class StateManager:
    def __init__(
        self, loc: str, name: str, host: str, port: int, state_id: str = None
    ) -> None:
        self.state_id = state_id if state_id else str(uuid4())
        self.name = f"{name}_{self.state_id}"
        self.logger = FedLogger("0", "STATE_MANAGER")
        try:
            module = import_module(f"server.state_manager.{loc}")
            kvstore = module.StateManager(name=self.name, host=host, port=port)
        except Exception as e:
            self.logger.error(
                "fedserver.state_manager", f"{e}\tFalling back to inmemory store"
            )
            module = import_module(f"server.state_manager.inmemory")
            kvstore = module.StateManager(name=self.name)

        self.get = kvstore.get
        self.get_large = kvstore.get
        self.put = kvstore.put
        self.put_large = kvstore.put
        self.keys = kvstore.keys
        self.len = kvstore.len
        self.clear = kvstore.clear
        self.deletebykey = kvstore.deletebykey
        self.getall = kvstore.getall
        self.putall = kvstore.putall

    def get(self, key):
        raise NotImplementedError

    def get_large(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def put_large(self, key, value):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def len(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def deletebykey(self, key):
        raise NotImplementedError

    def getall(self):
        raise NotImplementedError

    def putall(self):
        raise NotImplementedError


class ReadOnlyState:
    def __init__(self, loc: str, name: str, host: str, port: int) -> None:
        self.state_id = str(uuid4())
        module = import_module(f"server.state_manager.{loc}")
        kvstore = module.StateManager(
            name=f"{name}_{self.state_id}", host=host, port=port
        )
        self.get = kvstore.get
        self.get_large = kvstore.get
        self.keys = kvstore.keys
        self.len = kvstore.len

    def get(self, key):
        raise NotImplementedError

    def get_large(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def len(self):
        raise NotImplementedError
