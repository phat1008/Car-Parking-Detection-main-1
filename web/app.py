"""Web dashboard cho hệ thống nhận diện chỗ đỗ xe."""

import base64
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.parking_counter import ParkingCounter  # noqa: E402
from src.smoother import SlotSmoother  # noqa: E402
from web.inference import classify_frame_chunked, sample_video_frames  # noqa: E402
from web.insights import build_dashboard_insights  # noqa: E402
from web.layout import build_layout  # noqa: E402
from web.report import build_history_csv  # noqa: E402
from web.storage import DetectionStore  # noqa: E402

logger = logging.getLogger(__name__)


def availability(empty, occupied):
    total = empty + occupied
    return round(empty / total, 4) if total else ""


MAX_UPLOAD_MB = 60
DEFAULT_INFERENCE_INTERVAL = 2


def _encode_processed_frame(frame, results, counts, fps=None):
    output = frame.copy()
    image_height, image_width = output.shape[:2]

    for item in results:
        x, y, width, height = item["box"]
        state = item["state"]
        color = {
            "empty": (0, 255, 0),
            "occupied": (0, 0, 255),
            "invalid": (128, 128, 128),
        }.get(state, (0, 255, 255))

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            color,
            1,
        )
        slot_id = str(item["index"])
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            slot_id,
            font,
            0.5,
            1,
        )
        label_x = max(0, min(x + 2, image_width - text_width - 1))
        text_y = min(
            max(y + text_height, text_height),
            image_height - baseline - 1,
        )
        cv2.putText(
            output,
            slot_id,
            (label_x, text_y),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if image_width >= 180 and image_height >= 55:
        panel_width = min(220, image_width - 16)
        panel_height = 72 if fps is not None else 48
        panel_x = image_width - panel_width - 8
        panel_y = 8
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (16, 19, 25),
            -1,
        )
        cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output,
            f"Co xe: {counts['occupied']}/{counts['configured']}",
            (panel_x + 8, panel_y + 17),
            font,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if fps is not None:
            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (panel_x + 8, panel_y + 34),
                font,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        legend = (
            ("Trong", (0, 255, 0)),
            ("Xe", (0, 0, 255)),
            ("Chua ro", (0, 255, 255)),
            ("Loi", (128, 128, 128)),
        )
        legend_x = panel_x + 8
        legend_y = panel_y + (58 if fps is not None else 37)
        for label, color in legend:
            cv2.rectangle(
                output,
                (legend_x, legend_y - 7),
                (legend_x + 7, legend_y),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (legend_x + 10, legend_y),
                font,
                0.28,
                (230, 233, 238),
                1,
                cv2.LINE_AA,
            )
            (label_width, _), _ = cv2.getTextSize(label, font, 0.28, 1)
            legend_x += label_width + 18

    scale = min(960 / image_width, 540 / image_height, 1.0)
    if scale < 1:
        output = cv2.resize(
            output,
            (round(image_width * scale), round(image_height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(
        ".jpg",
        output,
        [cv2.IMWRITE_JPEG_QUALITY, 75],
    )
    if not success:
        raise RuntimeError("Không thể mã hóa khung hình video đã xử lý.")

    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _default_detector():
    from parking_detector import CNNParkingDetector

    return CNNParkingDetector(
        model_path=os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "parking_cnn.keras")),
        config_path=os.getenv("CONFIG_PATH", str(BASE_DIR / "config" / "parking_config.json")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.60")),
    )


def create_app(detector=None, store=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    if detector is None:
        detector = _default_detector()
    if store is None:
        store = DetectionStore.from_env()

    camera_id = detector.slot_manager.camera_id
    layout = build_layout(detector.slots)
    slot_column = {
        slot_id: col["column"] for col in layout for lane in col["lanes"] for slot_id in lane
    }

    def column_stats(results):
        stats = {
            col["column"]: {"column": col["column"], "empty": 0, "occupied": 0, "unknown": 0}
            for col in layout
        }
        for item in results:
            column = slot_column.get(item["index"])
            if column is not None and item["state"] in ("empty", "occupied", "unknown"):
                stats[column][item["state"]] += 1
        return [stats[c] for c in sorted(stats)]

    def slot_list(results):
        return [
            {
                "id": item["index"],
                "state": item["state"],
                "confidence": round(item["confidence"], 4),
            }
            for item in results
        ]

    def summary(
        counts, results, processed_frames=1, processed_inferences=1, inference_interval=None
    ):
        empty, occupied = counts.get("empty", 0), counts.get("occupied", 0)
        return {
            "schema_version": 2,
            "counts": counts,
            "availability": availability(empty, occupied),
            "column_stats": column_stats(results),
            "slots": slot_list(results),
            "processed_frames": processed_frames,
            "processed_inferences": processed_inferences,
            "inference_interval": inference_interval,
        }

    def timestamp_utc():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @app.get("/")
    def dashboard():
        return render_template("index.html", camera_id=camera_id)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "slots": len(detector.slots)})

    @app.get("/api/insights")
    def report_insights():
        return jsonify(build_dashboard_insights(BASE_DIR / "reports"))

    @app.get("/api/slots")
    def slots():
        return jsonify(
            {
                "camera_id": camera_id,
                "slot_ids": [slot["id"] for slot in detector.slots],
                "persistent_storage": store.persistent,
            }
        )

    @app.get("/api/report.csv")
    def report_csv():
        try:
            limit = max(1, min(int(request.args.get("limit", 200)), 1000))
            records = store.history(limit=limit)
        except Exception:
            logger.exception("Không đọc được dữ liệu cho báo cáo")
            return jsonify({"error": "Không kết nối được cơ sở dữ liệu."}), 503
        delimiter = ";" if request.args.get("delimiter") == ";" else ","
        body = build_history_csv(records, delimiter=delimiter)
        return Response(
            body,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=parking_report.csv"},
        )

    @app.get("/api/layout")
    def layout_view():
        boxes = [
            {
                "id": slot["id"],
                "x": slot["x"],
                "y": slot["y"],
                "width": slot["width"],
                "height": slot["height"],
            }
            for slot in detector.slots
        ]
        return jsonify(
            {
                "columns": layout,
                "boxes": boxes,
                "source_width": detector.slot_manager.source_width,
                "source_height": detector.slot_manager.source_height,
                "total_slots": len(boxes),
            }
        )

    @app.post("/api/video")
    def detect_video():
        file = request.files.get("video")
        if file is None or file.filename == "":
            return jsonify({"error": "Thiếu file video (field 'video')."}), 400

        try:
            inference_interval = int(
                request.form.get(
                    "inference_interval",
                    DEFAULT_INFERENCE_INTERVAL,
                )
            )
            if inference_interval < 1:
                raise ValueError("inference_interval phải lớn hơn 0")
        except ValueError:
            return jsonify({"error": "Tham số inference_interval không hợp lệ."}), 400

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            file.save(tmp.name)
        except Exception:
            os.remove(tmp.name)
            logger.exception("Không lưu được video tải lên")
            return jsonify({"error": "Không thể lưu video tải lên."}), 500

        def generate():
            timeline = []
            last = None
            smoother = SlotSmoother(window_size=7, min_stable_count=5)
            frame_index = 0
            processed_inferences = 0
            previous_time = None
            display_fps = 0.0
            last_counts = ParkingCounter.count_slots([], detector.slots)

            try:
                for time_sec, frame in sample_video_frames(
                    tmp.name,
                    0,
                ):
                    frame_index += 1
                    now = time.perf_counter()
                    if previous_time is not None and now > previous_time:
                        display_fps = 1.0 / (now - previous_time)
                    previous_time = now

                    should_infer = frame_index == 1 or frame_index % inference_interval == 0

                    if should_infer:
                        output = classify_frame_chunked(detector, frame)
                        for item in output["results"]:
                            raw_state = item["state"]
                            stable_state = smoother.update(
                                slot_id=item["index"],
                                current_state=raw_state,
                                confidence=item.get("confidence"),
                                min_confidence=getattr(
                                    getattr(detector, "classifier", None),
                                    "confidence_threshold",
                                    0.60,
                                ),
                            )
                            item["raw_state"] = raw_state
                            item["state"] = stable_state
                            item["label"] = stable_state

                        last = output
                        last_counts = ParkingCounter.count_slots(
                            last["results"],
                            detector.slots,
                        )
                        timeline_item = {
                            "time_sec": round(time_sec, 2),
                            "counts": {
                                k: last_counts[k]
                                for k in ("empty", "occupied", "unknown", "invalid")
                            },
                        }
                        timeline.append(timeline_item)
                        processed_inferences += 1

                    if last is None:
                        raise RuntimeError("Không có kết quả inference cho frame đầu.")

                    yield json.dumps(
                        {
                            "type": "frame",
                            "frame": frame_index,
                            "time_sec": round(time_sec, 2),
                            "inference": should_infer,
                            "counts": last_counts,
                            "preview": _encode_processed_frame(
                                frame,
                                last["results"],
                                last_counts,
                                display_fps,
                            ),
                        },
                        ensure_ascii=False,
                    ) + "\n"

                if last is None:
                    yield json.dumps(
                        {
                            "type": "error",
                            "error": "Video không có khung hình nào.",
                        },
                        ensure_ascii=False,
                    ) + "\n"
                    return

                record = {
                    "camera_id": camera_id,
                    "source": "video",
                    "video_name": file.filename,
                    "created_at": timestamp_utc(),
                    "counts": last_counts,
                    **summary(
                        last_counts,
                        last["results"],
                        processed_frames=frame_index,
                        processed_inferences=processed_inferences,
                        inference_interval=inference_interval,
                    ),
                    "timeline": timeline,
                }

                saved = True
                try:
                    store.save(record)
                except Exception:
                    logger.exception("Không lưu được kết quả video vào MongoDB")
                    saved = False

                yield json.dumps(
                    {
                        "type": "complete",
                        **record,
                        "saved": saved,
                    },
                    ensure_ascii=False,
                ) + "\n"
            except ValueError as exc:
                yield json.dumps(
                    {"type": "error", "error": str(exc)},
                    ensure_ascii=False,
                ) + "\n"
            except Exception:
                logger.exception("Lỗi khi xử lý video")
                yield json.dumps(
                    {"type": "error", "error": "Lỗi khi xử lý video."},
                    ensure_ascii=False,
                ) + "\n"
            finally:
                if os.path.exists(tmp.name):
                    os.remove(tmp.name)

        return Response(
            generate(),
            mimetype="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/detect")
    def detect():
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify({"error": "Thiếu file ảnh (field 'image')."}), 400

        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return (
                jsonify({"error": "Không đọc được ảnh. Hãy dùng JPG hoặc PNG."}),
                400,
            )

        try:
            output = classify_frame_chunked(detector, frame)
        except Exception:
            logger.exception("Lỗi khi nhận diện ảnh")
            return jsonify({"error": "Lỗi khi chạy model."}), 500

        counts = ParkingCounter.count_slots(output["results"], detector.slots)
        record = {
            "camera_id": camera_id,
            "source": "image",
            "created_at": timestamp_utc(),
            "counts": counts,
            **summary(counts, output["results"]),
        }

        saved = True
        try:
            store.save(record)
        except Exception:
            logger.exception("Không lưu được kết quả vào MongoDB")
            saved = False

        return jsonify({**record, "saved": saved})

    @app.get("/api/status")
    def status():
        try:
            latest = store.latest()
        except Exception:
            logger.exception("Không đọc được trạng thái gần nhất")
            return jsonify({"error": "Không kết nối được cơ sở dữ liệu."}), 503
        return jsonify({"camera_id": camera_id, "latest": latest})

    @app.get("/api/history")
    def history():
        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 200))
        try:
            items = store.history(limit=limit)
        except Exception:
            logger.exception("Không đọc được lịch sử")
            return jsonify({"error": "Không kết nối được cơ sở dữ liệu."}), 503
        return jsonify({"items": items})

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": f"File vượt quá {MAX_UPLOAD_MB} MB."}), 413

    return app
