"""Build dashboard notices from evaluation and inference report files."""

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score")
CLASS_NAMES = {"empty": "empty", "occupied": "occupied", "not_empty": "occupied"}
COUNT_NAMES = ("configured", "valid", "empty", "occupied", "unknown", "invalid")


def _read_class_report(path):
    supports = {}
    metrics = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 5 or parts[0] not in CLASS_NAMES:
                continue
            try:
                precision, recall, f1_score = (float(value) for value in parts[1:4])
                support = int(parts[4])
            except ValueError:
                continue
            class_name = CLASS_NAMES[parts[0]]
            supports[class_name] = support
            metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }
    return metrics, supports


def _load_model_evaluation(metrics_path):
    if not metrics_path.is_file():
        return None, None
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            raw_metrics = json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Không đọc được báo cáo đánh giá CNN: %s", metrics_path)
        return None, "Không đọc được báo cáo đánh giá CNN; hãy chạy lại bước đánh giá."

    if not isinstance(raw_metrics, dict):
        return None, "Báo cáo đánh giá CNN không đúng định dạng."

    metrics = {}
    for name in METRIC_NAMES:
        value = raw_metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            return None, f"Báo cáo đánh giá CNN thiếu chỉ số hợp lệ: {name}."
        metrics[name] = float(value)

    class_report_path = metrics_path.with_name("classification_report.txt")
    try:
        class_metrics, supports = _read_class_report(class_report_path)
    except OSError:
        logger.exception("Không đọc được classification report: %s", class_report_path)
        class_metrics, supports = {}, {}

    evaluation = {
        **metrics,
        "classes": class_metrics,
        "supports": supports,
    }

    if not supports:
        return evaluation, (
            "Không xác định được số mẫu của từng lớp trong báo cáo; "
            "cần kiểm tra lại báo cáo đánh giá."
        )
    missing_classes = [
        label
        for name, label in (("empty", "ô trống"), ("occupied", "ô có xe"))
        if supports.get(name, 0) == 0
    ]
    if missing_classes:
        return evaluation, (
            f"Báo cáo đánh giá không có mẫu lớp {', '.join(missing_classes)}. "
            "Accuracy hiện tại có thể gây hiểu nhầm; hãy đánh giá lại với tập test "
            "có đủ ảnh của cả hai lớp."
        )
    return evaluation, None


def _load_latest_run(runs_dir):
    try:
        run_dirs = sorted(
            (path for path in runs_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
    except FileNotFoundError:
        return None, None
    except OSError:
        logger.exception("Không đọc được thư mục báo cáo lần chạy: %s", runs_dir)
        return None, "Không đọc được báo cáo lần chạy gần nhất."

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
        except (OSError, json.JSONDecodeError):
            logger.exception("Không đọc được báo cáo lần chạy: %s", summary_path)
            return None, "Báo cáo lần chạy mới nhất bị lỗi hoặc không đọc được."

        if not isinstance(summary, dict) or not isinstance(summary.get("last_counts"), dict):
            return None, "Báo cáo lần chạy mới nhất thiếu dữ liệu thống kê."

        counts = summary["last_counts"]
        if any(
            isinstance(counts.get(name), bool)
            or not isinstance(counts.get(name), int)
            or counts[name] < 0
            for name in COUNT_NAMES
        ):
            return None, "Báo cáo lần chạy mới nhất có số liệu thống kê không hợp lệ."
        run_numbers = {}
        for name in ("processed_frames", "processed_inferences"):
            value = summary.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None, f"Báo cáo lần chạy mới nhất thiếu số liệu hợp lệ: {name}."
            run_numbers[name] = value

        source = summary.get("source")
        source_name = (
            source.replace("\\", "/").rsplit("/", 1)[-1] if isinstance(source, str) else ""
        )
        run = {
            "run_id": run_dir.name,
            "source": source_name,
            **run_numbers,
            "confidence_threshold": summary.get("confidence_threshold"),
            "counts": {name: counts[name] for name in COUNT_NAMES},
        }
        return run, None

    return None, None


def build_dashboard_insights(reports_dir):
    """Return user-safe notices and summaries from local report artifacts."""
    reports_dir = Path(reports_dir)
    notices = []
    model, model_warning = _load_model_evaluation(reports_dir / "metrics" / "cnn" / "metrics.json")
    if model is None:
        notices.append(
            {
                "level": "info",
                "title": "Chưa có báo cáo đánh giá CNN",
                "message": model_warning
                or "Chạy scripts/evaluate_model.py để tạo báo cáo đánh giá.",
            }
        )
    elif model_warning:
        notices.append(
            {
                "level": "warning",
                "title": "Cần kiểm tra báo cáo CNN",
                "message": model_warning,
            }
        )
    else:
        notices.append(
            {
                "level": "success",
                "title": "Báo cáo CNN có đủ hai lớp",
                "message": (
                    f"Đánh giá trên {model['supports']['empty'] + model['supports']['occupied']} "
                    f"mẫu: accuracy {model['accuracy']:.1%}, "
                    f"recall lớp có xe {model['classes']['occupied']['recall']:.1%}."
                ),
            }
        )

    latest_run, run_warning = _load_latest_run(reports_dir / "runs")
    if latest_run is None:
        notices.append(
            {
                "level": "info" if run_warning is None else "warning",
                "title": "Chưa có báo cáo lần chạy",
                "message": run_warning or "Chưa tìm thấy summary.json trong reports/runs.",
            }
        )
    else:
        counts = latest_run["counts"]
        if counts["invalid"]:
            notices.append(
                {
                    "level": "warning",
                    "title": "Có ô không xử lý được",
                    "message": (
                        f"Lần chạy {latest_run['run_id']} có {counts['invalid']} ô lỗi "
                        "ở frame cuối; hãy kiểm tra cấu hình vùng đỗ và video."
                    ),
                }
            )
        if counts["unknown"]:
            notices.append(
                {
                    "level": "warning",
                    "title": "Một số ô chưa đủ độ tin cậy",
                    "message": (
                        f"{counts['unknown']} ô ở frame cuối có trạng thái chưa xác định; "
                        "hãy kiểm tra ảnh crop và điều kiện ánh sáng."
                    ),
                }
            )
        if counts["configured"] != counts["valid"]:
            notices.append(
                {
                    "level": "warning",
                    "title": "Số ô hợp lệ không khớp cấu hình",
                    "message": (
                        f"Cấu hình có {counts['configured']} ô nhưng chỉ "
                        f"{counts['valid']} ô hợp lệ ở frame cuối."
                    ),
                }
            )
        if not counts["invalid"] and not counts["unknown"]:
            notices.append(
                {
                    "level": "success",
                    "title": "Lần chạy gần nhất không có ô lỗi/chưa xác định",
                    "message": (
                        "Đây là thống kê ở frame cuối; không đồng nghĩa mọi dự đoán "
                        "đều chính xác."
                    ),
                }
            )

    return {
        "model_evaluation": model,
        "latest_run": latest_run,
        "notices": notices,
    }
