import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# Allow this script to be run directly from the repository root:
# `python scripts/run.py`.
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from parking_detector import CNNParkingDetector  # noqa: E402
from src.parking_counter import ParkingCounter  # noqa: E402
from src.smoother import SlotSmoother  # noqa: E402


class PrefetchedVideoReader:
    _END_OF_STREAM = object()

    def __init__(self, capture, buffer_size):
        self.capture = capture
        self.frames = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _enqueue(self, item):
        while not self.stop_event.is_set():
            try:
                self.frames.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _read_frames(self):
        try:
            while not self.stop_event.is_set():
                success, frame = self.capture.read()
                if not success:
                    break
                if not self._enqueue(frame):
                    return
        except Exception as error:
            self._enqueue(error)
        finally:
            self._enqueue(self._END_OF_STREAM)

    def read(self):
        item = self.frames.get()
        if item is self._END_OF_STREAM:
            return False, None
        if isinstance(item, Exception):
            raise item
        return True, item

    def close(self):
        self.stop_event.set()
        self.capture.release()
        self.thread.join(timeout=1.0)


def parse_source(value):
    return int(value) if value.isdigit() else value


def draw_slot_id(output, slot_id, box):
    """Draw a white slot ID label without a background."""
    x, y = box[:2]
    image_height, image_width = output.shape[:2]
    text = str(slot_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        0.3,
        1,
    )
    label_x = max(0, min(x + 2, image_width - text_width - 1))
    text_y = min(
        max(y + text_height, text_height),
        image_height - baseline - 1,
    )
    cv2.putText(
        output,
        text,
        (label_x, text_y),
        font,
        0.3,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_results(frame, results, counts, fps):
    output = frame.copy()

    for item in results:
        x, y, width, height = item["box"]
        state = item["state"]

        if state == "empty":
            color = (0, 255, 0)
        elif state in ("occupied", "not_empty"):
            color = (0, 0, 255)
        elif state == "invalid":
            color = (128, 128, 128)
        else:
            color = (0, 255, 255)

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            color,
            1,
        )
        draw_slot_id(output, item["index"], item["box"])

    lines = [
        f"Configured: {counts['configured']}",
        f"Empty: {counts['empty']}",
        f"Occupied: {counts['occupied']}",
        f"Unknown: {counts['unknown']}",
        f"Invalid: {counts['invalid']}",
        f"FPS: {fps:.1f}",
    ]

    x = 20
    y = 35

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (x, y + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    legend_items = [
        ("Not empty", (0, 0, 255)),
        ("Unknown", (0, 255, 255)),
        ("Empty", (0, 255, 0)),
    ]
    legend_x = 20
    legend_y = y + len(lines) * 25 + 18
    legend_width = 170
    legend_height = len(legend_items) * 23 + 12

    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (legend_x - 8, legend_y - 18),
        (legend_x + legend_width, legend_y + legend_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

    for index, (label, color) in enumerate(legend_items):
        item_y = legend_y + index * 23
        cv2.rectangle(
            output,
            (legend_x, item_y - 11),
            (legend_x + 12, item_y + 1),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (legend_x + 20, item_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def resize_for_display(frame, max_width=1280, max_height=720):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale == 1.0:
        return frame

    return cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def display_size(width, height, max_width=1280, max_height=720):
    scale = min(max_width / width, max_height / height, 1.0)
    return round(width * scale), round(height * scale)


def create_output_dir():
    runs_dir = BASE_DIR / "reports" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = runs_dir / timestamp
    suffix = 1

    while True:
        try:
            output_dir.mkdir()
        except FileExistsError:
            output_dir = runs_dir / f"{timestamp}_{suffix}"
            suffix += 1
        else:
            return output_dir


def main():
    parser = argparse.ArgumentParser(description="CNN production parking detection")

    parser.add_argument(
        "source_positional",
        nargs="?",
        help="Video path or webcam index (positional shortcut)",
    )

    parser.add_argument(
        "--source",
        default=None,
        help="Video path or webcam index",
    )

    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config" / "parking_config.json"),
    )

    parser.add_argument(
        "--model",
        default=str(BASE_DIR / "models" / "parking_cnn.keras"),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--inference-interval",
        type=int,
        default=2,
        help="Run CNN every N frames",
    )

    parser.add_argument(
        "--prefetch-frames",
        type=int,
        default=4,
        help="Number of frames buffered while inference is running",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
    )

    args = parser.parse_args()

    if args.inference_interval < 1:
        parser.error("--inference-interval must be >= 1")
    if args.prefetch_frames < 1:
        parser.error("--prefetch-frames must be >= 1")

    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        parser.error("--confidence must be a finite number between 0 and 1")

    source_value = args.source if args.source is not None else args.source_positional
    source = parse_source(
        source_value if source_value is not None else str(BASE_DIR / "carPark.mp4")
    )
    output_dir = create_output_dir()

    detector = CNNParkingDetector(
        model_path=args.model,
        config_path=args.config,
        confidence_threshold=args.confidence,
    )

    smoother = SlotSmoother(
        window_size=7,
        min_stable_count=5,
    )

    counter = ParkingCounter()

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_source = capture.get(cv2.CAP_PROP_FPS)

    if not fps_source or fps_source <= 0:
        fps_source = 25.0

    output_video = output_dir / "result.mp4"
    video_writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_source,
        (width, height),
    )

    if not video_writer.isOpened():
        video_writer.release()
        capture.release()
        raise RuntimeError(
            f"Cannot open output video writer: {output_video} "
            f"(codec=mp4v, size={width}x{height}, fps={fps_source})"
        )

    slot_csv_path = output_dir / "slot_predictions.csv"
    frame_csv_path = output_dir / "frame_statistics.csv"
    summary_path = output_dir / "summary.json"

    frame_index = 0
    processed_inferences = 0
    previous_time = time.perf_counter()
    display_fps = 0.0

    last_results = []
    last_counts = {
        "configured": len(detector.slots),
        "valid": 0,
        "empty": 0,
        "occupied": 0,
        "unknown": len(detector.slots),
        "invalid": 0,
    }

    if not args.no_display:
        cv2.namedWindow(
            "CNN Parking Production",
            cv2.WINDOW_NORMAL,
        )
        display_width, display_height = display_size(width, height)
        cv2.resizeWindow(
            "CNN Parking Production",
            display_width,
            display_height,
        )

    with (
        slot_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as slot_file,
        frame_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as frame_file,
    ):
        slot_writer = csv.writer(slot_file)
        frame_writer = csv.writer(frame_file)

        slot_writer.writerow(
            [
                "frame",
                "slot_id",
                "raw_state",
                "stable_state",
                "confidence",
                "probability_occupied",
            ]
        )

        frame_writer.writerow(
            [
                "frame",
                "timestamp",
                "configured",
                "empty",
                "occupied",
                "unknown",
                "invalid",
                "availability",
            ]
        )

        frame_reader = PrefetchedVideoReader(capture, args.prefetch_frames)

        try:
            while True:
                success, frame = frame_reader.read()

                if not success:
                    break

                frame_index += 1

                now = time.perf_counter()
                elapsed = now - previous_time

                if elapsed > 0:
                    display_fps = 1.0 / elapsed

                previous_time = now

                should_infer = frame_index == 1 or frame_index % args.inference_interval == 0

                if should_infer:
                    analysis = detector.classify_frame(frame)

                    smoothed_results = []

                    for item in analysis["results"]:
                        raw_state = item["state"]

                        stable_state = smoother.update(
                            slot_id=item["index"],
                            current_state=raw_state,
                            confidence=item.get("confidence"),
                            min_confidence=args.confidence,
                        )

                        item["raw_state"] = raw_state
                        item["state"] = stable_state
                        item["label"] = stable_state

                        smoothed_results.append(item)

                        slot_writer.writerow(
                            [
                                frame_index,
                                item["index"],
                                raw_state,
                                stable_state,
                                item.get("confidence"),
                                item.get("probability_occupied"),
                            ]
                        )

                    last_results = smoothed_results
                    last_counts = counter.count_slots(
                        last_results,
                        detector.slots,
                    )

                    known_count = last_counts["empty"] + last_counts["occupied"]

                    availability = last_counts["empty"] / known_count if known_count else 0.0

                    timestamp = (frame_index - 1) / fps_source

                    frame_writer.writerow(
                        [
                            frame_index,
                            round(timestamp, 3),
                            last_counts["configured"],
                            last_counts["empty"],
                            last_counts["occupied"],
                            last_counts["unknown"],
                            last_counts["invalid"],
                            round(availability, 4),
                        ]
                    )

                    processed_inferences += 1

                output_frame = draw_results(
                    frame,
                    last_results,
                    last_counts,
                    display_fps,
                )

                video_writer.write(output_frame)

                if not args.no_display:
                    display_frame = resize_for_display(output_frame)
                    cv2.imshow(
                        "CNN Parking Production",
                        display_frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        break

        finally:
            frame_reader.close()
            video_writer.release()
            cv2.destroyAllWindows()

    summary = {
        "source": str(source),
        "model": str(args.model),
        "config": str(args.config),
        "frame_width": width,
        "frame_height": height,
        "source_fps": fps_source,
        "processed_frames": frame_index,
        "processed_inferences": processed_inferences,
        "inference_interval": args.inference_interval,
        "prefetch_frames": args.prefetch_frames,
        "confidence_threshold": args.confidence,
        "last_counts": last_counts,
        "output_video": str(output_video),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Output directory: {output_dir}")
    print(f"Video: {output_video}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
