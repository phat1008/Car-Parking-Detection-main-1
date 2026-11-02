"""Lưu kết quả nhận diện: MongoDB nếu có MONGO_URI, ngược lại giữ trong bộ nhớ."""

import logging
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Lock

logger = logging.getLogger(__name__)

VN_TZ = timezone(timedelta(hours=7))


def _to_bson_date(value):
    """Đổi chuỗi ISO UTC (...Z) thành datetime để Atlas hiển thị dạng ngày giờ."""
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


def _to_iso_utc(value):
    """Đổi datetime đọc từ Mongo về chuỗi ISO UTC để giao diện xử lý như cũ."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return value


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
            doc = dict(record)
            doc["created_at"] = _to_bson_date(doc.get("created_at"))
            doc["created_at_vn"] = _to_bson_date(doc["created_at"]).astimezone(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
            self._collection.insert_one(doc)

    def latest(self):
        items = self.history(limit=1)
        return items[0] if items else None

    def history(self, limit=50):
        if self._collection is not None:
            cursor = self._collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
            items = list(cursor)
            for item in items:
                if "created_at" in item:
                    item["created_at"] = _to_iso_utc(item["created_at"])
            return items

        with self._lock:
            return list(self._memory)[:limit]

