from functools import reduce


class StateManager:
    def __init__(self, name: str, host=None, port=None) -> None:
        self.state = dict()
        self.name = name

    def get(self, key):
        getter = self.state
        value = reduce(
            lambda d, key: d.get(key) if isinstance(d, dict) else None,
            key.split("."),
            getter,
        )
        return value

    def put(self, key, value) -> None:
        setter = self.state
        keys = key.split(".")
        for k in keys[:-1]:
            if k not in setter:
                setter[k] = dict()
            setter = setter[k]
        setter[keys[-1]] = value

    def keys(self):
        return self.state.keys()

    def len(self):
        return len(self.state)

    def clear(self):
        self.state.clear()

    def deletebykey(self, key):
        deleter = self.state
        keys = key.split(".")
        for k in keys[-1]:
            if k not in deleter:
                return
            deleter = deleter[k]
        del deleter[keys[-1]]

    def getall(self):
        return self.state

    def putall(self, data: dict):
        self.state = data
