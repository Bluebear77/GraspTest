import dbm
import json


class Cache:
    def __init__(self, db: "dbm._Database") -> None:
        self.db = db

    @staticmethod
    def load(cache_path: str):
        db = dbm.open(cache_path, "r")
        return Cache(db)

    def get(self, identifier: str) -> dict | None:
        raw = self.db.get(identifier)
        if raw is None:
            return None

        return json.loads(raw)
