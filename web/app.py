"""Web dashboard cho hệ thống nhận diện chỗ đỗ xe."""

import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.parking_counter import ParkingCounter  # noqa: E402
from web.inference import classify_frame_chunked, sample_video_frames  # noqa: E402
from web.layout import build_layout  # noqa: E402
from web.storage import DetectionStore  # noqa: E402

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 60
MAX_VIDEO_FRAMES = 60


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

    @app.get("/")
    def dashboard():
        return render_template("index.html", camera_id=camera_id)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "slots": len(detector.slots)})

    @app.get("/api/slots")
    def slots():
        return jsonify(
            {
                "camera_id": camera_id,
                "slot_ids": [slot["id"] for slot in detector.slots],
                "persistent_storage": store.persistent,
            }
        )

    @app.get("/api/layout")
    def layout_view():
        return jsonify({"columns": layout, "total_slots": sum(
            len(lane) for col in layout for lane in col["lanes"]
        )})

    @app.post("/api/video")
    def detect_video():
        file = request.files.get("video")
        if file is None or file.filename == "":
            return jsonify({"error": "Thiếu file video (field 'video')."}), 400

        try:
            every_sec = min(max(float(request.form.get("every_sec", 2)), 0.5), 30.0)
            max_frames = min(max(int(request.form.get("max_frames", 30)), 1), MAX_VIDEO_FRAMES)
        except ValueError:
            return jsonify({"error": "Tham số every_sec hoặc max_frames không hợp lệ."}), 400

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            file.save(tmp.name)
            timeline = []
            last = None
            for time_sec, frame in sample_video_frames(tmp.name, every_sec, max_frames):
                output = classify_frame_chunked(detector, frame)
                counts = ParkingCounter.count_slots(output["results"], detector.slots)
                timeline.append({"time_sec": round(time_sec, 2), "counts": counts})
                last = output
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            logger.exception("Lỗi khi xử lý video")
            return jsonify({"error": "Lỗi khi xử lý video."}), 500
        finally:
            os.remove(tmp.name)

        if last is None:
            return jsonify({"error": "Video không có khung hình nào."}), 400

        record = {
            "camera_id": camera_id,
            "source": "video",
            "video_name": file.filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "counts": timeline[-1]["counts"],
            "slots": [
                {
                    "id": item["index"],
                    "state": item["state"],
                    "confidence": item["confidence"],
                    "probability_occupied": item["probability_occupied"],
                }
                for item in last["results"]
            ],
            "timeline": timeline,
        }

        saved = True
        try:
            store.save(record)
        except Exception:
            logger.exception("Không lưu được kết quả video vào MongoDB")
            saved = False

        return jsonify({**record, "saved": saved})

    @app.post("/api/detect")
    def detect():
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify({"error": "Thiếu file ảnh (field 'image')."}), 400

        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Không đọc được ảnh. Hãy dùng JPG hoặc PNG."}), 400

        try:
            output = classify_frame_chunked(detector, frame)
        except Exception:
            logger.exception("Lỗi khi nhận diện ảnh")
            return jsonify({"error": "Lỗi khi chạy model."}), 500

        counts = ParkingCounter.count_slots(output["results"], detector.slots)
        record = {
            "camera_id": camera_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "slots": [
                {
                    "id": item["index"],
                    "state": item["state"],
                    "confidence": item["confidence"],
                    "probability_occupied": item["probability_occupied"],
                }
                for item in output["results"]
            ],
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

