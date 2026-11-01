"""Nhận diện theo nhóm nhỏ để giảm đỉnh RAM và hỗ trợ xử lý video."""

import queue
import threading

import cv2

CHUNK_SIZE = 32
VIDEO_PREFETCH_FRAMES = 4


def classify_frame_chunked(detector, frame, chunk_size=CHUNK_SIZE):
    """Giống detector.classify_frame nhưng gọi model theo từng nhóm ô."""
    slot_items = detector.slot_manager.crop_slots(frame)
    results = []

    for start in range(0, len(slot_items), chunk_size):
        batch = slot_items[start : start + chunk_size]
        crops = [item["crop"] for item in batch]
        predictions = detector.classifier.predict_batch(crops)

        for item, prediction in zip(batch, predictions):
            results.append(
                {
                    "index": item["id"],
                    "box": item["box"],
                    "type": item["type"],
                    **prediction,
                }
            )

    return {"results": results, "total": len(results)}


def sample_video_frames(
    path,
    every_sec,
    max_frames=None,
    buffer_size=VIDEO_PREFETCH_FRAMES,
):
    """Đọc trước video trên luồng nền và trả về frame theo thứ tự."""
    if buffer_size < 1:
        raise ValueError("buffer_size phải lớn hơn 0")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Không mở được file video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_sec)))
    frames = queue.Queue(maxsize=buffer_size)
    stop_event = threading.Event()
    end_of_stream = object()

    def enqueue(item):
        while not stop_event.is_set():
            try:
                frames.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def read_frames():
        index = 0
        yielded = 0

        try:
            while (max_frames is None or yielded < max_frames) and not stop_event.is_set():
                if not cap.grab():
                    break
                if every_sec <= 0 or index % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    if not enqueue((index / fps, frame)):
                        return
                    yielded += 1
                index += 1
        except Exception as error:
            enqueue(error)
        finally:
            enqueue(end_of_stream)

    reader = threading.Thread(target=read_frames, daemon=True)
    reader.start()

    try:
        while True:
            item = frames.get()
            if item is end_of_stream:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        reader.join(timeout=1.0)
        cap.release()
        reader.join(timeout=1.0)
