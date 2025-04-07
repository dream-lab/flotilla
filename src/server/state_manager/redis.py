from pickle import PicklingError
from pickle import dumps as p_dumps
from pickle import loads as p_loads

from redis import Redis
from redis import exceptions as redis_exceptions

from utils.logger import FedLogger


class StateManager:
    def __init__(self, name: str, host: str = "localhost", port: int = 6379) -> None:
        self.logger = FedLogger("0", "STATE_MANAGER")
        self.redis = Redis(host=host, port=port)
        self.name = name
        # self.redis.flushdb()

    def get(self, key):
        try:
            value = self.redis.hget(self.name, key)
            if value == None:
                return None
            deserialized_value = p_loads(value)
        except redis_exceptions.ConnectionError as e:
            print("GET ERROR")
            self.logger.error("fedserver.redis", "-".join(e.args))
        return deserialized_value

    def put(self, key, value):
        client_id = key.split(".")[0]
        try:
            serialized_value = p_dumps(value)
        except PicklingError:
            self.logger.error("fedserver.redis", f"{value} cannot be pickled")
        try:
            self.redis.hset(self.name, key, serialized_value)
            self.redis.sadd(f"keys_{self.name}", client_id)
        except redis_exceptions.ConnectionError as e:
            self.logger.error("fedserver.redis", "-".join(e.args))
        except redis_exceptions.DataError:
            self.logger.error("fedserver.redis", f"Invalid input type")

    def keys(self):
        try:
            return [
                i.decode(encoding="utf-8")
                for i in self.redis.smembers(f"keys_{self.name}")
            ]
        except redis_exceptions.ConnectionError as e:
            self.logger.error("fedserver.redis", "-".join(e.args))

    def len(self):
        try:
            return self.redis.scard(f"keys_{self.name}")
        except redis_exceptions.ConnectionError as e:
            self.logger.error("fedserver.redis", "-".join(e.args))

    def clear(self):
        try:
            self.redis.delete(f"keys_{self.name}")
            keys = [i.decode() for i in self.redis.hkeys(self.name)]
            for key in keys:
                self.redis.hdel(self.name, key)
        except redis_exceptions.ConnectionError as e:
            self.logger.error("fedserver.redis", "-".join(e.args))

    def deletebykey(self, key):
        try:
            self.redis.srem(f"keys_{self.name}", key)
            self.redis.hdel(self.name, key)
        except redis_exceptions.ConnectionError as e:
            self.logger.error("fedserver.redis", "-".join(e.args))

    def getall(self):
        return self.redis.hgetall(self.name)

    def putall(self, data: dict):
        for key, value in data.items():
            key = key.decode(encoding="utf-8")
            client_id = key.split(".")[0]
            try:
                self.redis.hset(self.name, key, value)
                self.redis.sadd(f"keys_{self.name}", client_id)
            except redis_exceptions.ConnectionError as e:
                self.logger.error("fedserver.redis", "-".join(e.args))
            except redis_exceptions.DataError:
                self.logger.error("fedserver.redis", f"Invalid input type")
