"""Lưu kết quả nhận diện: MongoDB nếu có MONGO_URI, ngược lại giữ trong bộ nhớ."""

import logging
import os
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


class DetectionStore:
    def __init__(self, collection=None, max_memory=200):
        self._collection = collection
        self._memory = deque(maxlen=max_memory)
        self._lock = Lock()

    @classmethod
    def from_env(cls):
        uri = os.getenv("MONGO_URI")
        if not uri:
            logger.warning("MONGO_URI chưa được đặt, dùng bộ nhớ tạm.")
            return cls()

        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client[os.getenv("MONGO_DB", "parking")]
        collection = db[os.getenv("MONGO_COLLECTION", "detections")]
        collection.create_index([("created_at", -1)])
        return cls(collection=collection)

    @property
    def persistent(self):
        return self._collection is not None

    def save(self, record):
        with self._lock:
            self._memory.appendleft(dict(record))

        if self._collection is not None:
            # insert_one thêm _id vào dict truyền vào, nên truyền bản sao.
            self._collection.insert_one(dict(record))

    def latest(self):
        items = self.history(limit=1)
        return items[0] if items else None

    def history(self, limit=50):
        if self._collection is not None:
            cursor = (
                self._collection.find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            return list(cursor)

        with self._lock:
            return list(self._memory)[:limit]
