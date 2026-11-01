import base64
import json
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import web.app as web_app
import web.inference as web_inference


class FakeStore:
    persistent = False

    def __init__(self):
        self.records = []

    def save(self, record):
        self.records.append(record)

    def latest(self):
        return None

    def history(self, limit=50):
        return self.records[:limit]


def create_test_app():
    detector = SimpleNamespace(
        slot_manager=SimpleNamespace(
            camera_id="test-camera",
            source_width=8,
            source_height=8,
        ),
        slots=[
            {
                "id": 2,
                "x": 1,
                "y": 1,
                "width": 5,
                "height": 5,
                "enabled": True,
            }
        ],
    )
    store = FakeStore()
    return web_app.create_app(detector=detector, store=store), store


def test_sample_video_frames_prefetches_in_background_and_preserves_samples(monkeypatch):
    second_frame_ready = Event()

    class FakeCapture:
        def __init__(self, path):
            self.index = -1
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 30.0

        def grab(self):
            self.index += 1
            return self.index < 10

        def retrieve(self):
            if self.index == 3:
                second_frame_ready.set()
            return True, np.full((1, 1, 3), self.index, dtype=np.uint8)

        def release(self):
            self.released = True

    capture = FakeCapture("parking.mp4")
    monkeypatch.setattr(web_inference.cv2, "VideoCapture", lambda path: capture)

    frames = web_inference.sample_video_frames(
        "parking.mp4",
        every_sec=0.1,
        max_frames=2,
        buffer_size=2,
    )

    try:
        first_time, first_frame = next(frames)
        assert second_frame_ready.wait(timeout=1)
        second_time, second_frame = next(frames)

        assert (first_time, second_time) == (0.0, 0.1)
        assert (int(first_frame[0, 0, 0]), int(second_frame[0, 0, 0])) == (0, 3)
        with pytest.raises(StopIteration):
            next(frames)
    finally:
        frames.close()

    assert capture.released


def test_sample_video_frames_propagates_decoder_errors(monkeypatch):
    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 30.0

        def grab(self):
            raise OSError("decoder failed")

        def release(self):
            pass

    monkeypatch.setattr(
        web_inference.cv2,
        "VideoCapture",
        lambda path: FakeCapture(),
    )

    with pytest.raises(OSError, match="decoder failed"):
        next(web_inference.sample_video_frames("parking.mp4", 0.1, 2))


def test_video_endpoint_streams_annotated_frames_and_saves_summary(monkeypatch):
    app, store = create_test_app()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    sample_args = []

    def fake_sample_video_frames(*args, **kwargs):
        sample_args.append(args)
        sample_args.append(kwargs)
        return iter((index / 30, frame) for index in range(8))

    inference_calls = []
    monkeypatch.setattr(
        web_app,
        "sample_video_frames",
        fake_sample_video_frames,
    )

    def fake_classify_frame(detector, image):
        inference_calls.append(image)
        return {
            "results": [
                {
                    "index": 2,
                    "box": (1, 1, 5, 5),
                    "state": "occupied",
                    "confidence": 0.95,
                    "probability_occupied": 0.95,
                }
            ],
            "total": 1,
        }

    monkeypatch.setattr(
        web_app,
        "classify_frame_chunked",
        fake_classify_frame,
    )

    response = app.test_client().post(
        "/api/video",
        data={"video": (BytesIO(b"video"), "parking.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "application/x-ndjson"
    events = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
    frame_events = [event for event in events if event["type"] == "frame"]
    completed = events[-1]

    assert [event["frame"] for event in frame_events] == list(range(1, 9))
    assert [event["inference"] for event in frame_events] == [
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert sample_args[0][1] == 0
    assert sample_args[1] == {}
    assert len(inference_calls) == 5
    assert len(completed["timeline"]) == 5
    assert base64.b64decode(frame_events[0]["preview"].split(",", 1)[1]).startswith(b"\xff\xd8")
    assert completed["type"] == "complete"
    assert completed["counts"]["occupied"] == 1
    assert completed["processed_frames"] == 8
    assert completed["processed_inferences"] == 5
    assert completed["saved"] is True
    assert len(store.records) == 1
    assert "preview" not in store.records[0]


def test_video_endpoint_rejects_non_positive_inference_interval():
    app, _ = create_test_app()

    response = app.test_client().post(
        "/api/video",
        data={
            "video": (BytesIO(b"video"), "parking.mp4"),
            "inference_interval": "0",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Tham số inference_interval không hợp lệ."


def test_processed_frame_uses_compact_slot_labels_and_status_legend(monkeypatch):
    labels = []
    text_calls = []
    put_text = web_app.cv2.putText

    def record_text(*args, **kwargs):
        labels.append(args[1])
        text_calls.append(args)
        return put_text(*args, **kwargs)

    monkeypatch.setattr(web_app.cv2, "putText", record_text)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    results = [
        {
            "index": 2,
            "box": (20, 30, 40, 20),
            "state": "occupied",
        }
    ]

    preview = web_app._encode_processed_frame(
        frame,
        results,
        {"configured": 10, "occupied": 3},
    )

    assert preview.startswith("data:image/jpeg;base64,")
    assert "2" in labels
    id_label = next(call for call in text_calls if call[1] == "2")
    assert id_label[4] == 0.22
    assert id_label[5] == (255, 255, 255)
    assert id_label[6] == 1
    assert "Co xe: 3/10" in labels
    assert {"Trong", "Xe", "Chua ro", "Loi"}.issubset(labels)
    assert "occupied" not in labels


def test_dashboard_hides_image_recognition_and_removes_image_api():
    app, _ = create_test_app()
    client = app.test_client()

    page = client.get("/")

    assert page.status_code == 200
    assert "Nhận diện từ ảnh" not in page.get_data(as_text=True)
    assert 'id="every-sec"' not in page.get_data(as_text=True)
    assert client.post("/api/detect").status_code == 404


def test_insights_endpoint_reads_reports_from_application_directory(tmp_path, monkeypatch):
    app, _ = create_test_app()
    monkeypatch.setattr(web_app, "BASE_DIR", tmp_path)

    response = app.test_client().get("/api/insights")

    assert response.status_code == 200
    data = response.get_json()
    assert data["model_evaluation"] is None
    assert data["latest_run"] is None
    assert len(data["notices"]) == 2
