"""Báo cáo: xuất CSV từ dữ liệu đã lưu và đọc run mới nhất trong reports/runs/."""

import csv
import io
import json
from pathlib import Path

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


def find_latest_run(base_dir):
    """Trả về thư mục run mới nhất có summary.json, hoặc None nếu không có."""
    runs_dir = Path(base_dir) / "reports" / "runs"
    if not runs_dir.is_dir():
        return None
    runs = sorted(
        (p for p in runs_dir.iterdir() if (p / "summary.json").is_file()),
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[0] if runs else None


def load_run_report(run_dir):
    """Đọc summary.json và dòng cuối của frame_statistics.csv (nếu có)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    frame_stats = run_dir / "frame_statistics.csv"
    last_frame = None
    frame_count = 0
    if frame_stats.is_file():
        with frame_stats.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        frame_count = len(rows)
        last_frame = rows[-1] if rows else None

    return {
        "run": run_dir.name,
        "summary": summary,
        "frame_statistics_rows": frame_count,
        "last_frame": last_frame,
    }
