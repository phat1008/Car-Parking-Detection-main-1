"""Báo cáo: xuất CSV từ dữ liệu đã lưu trong MongoDB."""

import csv
import io

REPORT_COLUMNS = [
    "created_at",
    "source",
    "video_name",
    "time_sec",
    "configured",
    "empty",
    "occupied",
    "unknown",
    "invalid",
    "availability",
]


def _availability(empty, occupied):
    total = empty + occupied
    return round(empty / total, 4) if total else ""


def _row(record, time_sec, counts):
    empty = counts.get("empty", 0)
    occupied = counts.get("occupied", 0)
    return {
        "created_at": record.get("created_at", ""),
        "source": record.get("source", "image"),
        "video_name": record.get("video_name", ""),
        "time_sec": time_sec,
        "configured": counts.get("configured", 0),
        "empty": empty,
        "occupied": occupied,
        "unknown": counts.get("unknown", 0),
        "invalid": counts.get("invalid", 0),
        "availability": _availability(empty, occupied),
    }


def build_history_csv(records, delimiter=","):
    """Mỗi bản ghi ảnh thành 1 dòng; mỗi khung hình video thành 1 dòng.

    delimiter=";" dùng cho Excel cài tiếng Việt (ngăn cột bằng dấu chấm phẩy).
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REPORT_COLUMNS, delimiter=delimiter)
    writer.writeheader()

    for record in records:
        if record.get("timeline"):
            for item in record["timeline"]:
                writer.writerow(_row(record, item.get("time_sec", ""), item.get("counts", {})))
        else:
            writer.writerow(_row(record, "", record.get("counts", {})))

    return buffer.getvalue()
