"""Nhận diện theo nhóm nhỏ để giảm đỉnh RAM và hỗ trợ xử lý video."""

import cv2

CHUNK_SIZE = 32


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


def sample_video_frames(path, every_sec, max_frames):
    """Đọc video và trả về (thời điểm giây, khung hình) cách nhau every_sec."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Không mở được file video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_sec)))
    index = 0
    yielded = 0

    try:
        while yielded < max_frames:
            if not cap.grab():
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index / fps, frame
                yielded += 1
            index += 1
    finally:
        cap.release()
