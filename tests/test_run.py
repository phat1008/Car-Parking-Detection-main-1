from datetime import datetime as DateTime

import numpy as np
import pytest

import scripts.run as run


def test_prefetched_video_reader_preserves_order_and_releases_capture():
    class FakeCapture:
        def __init__(self):
            self.frames = ["frame-1", "frame-2", "frame-3"]
            self.released = False

        def read(self):
            if self.frames:
                return True, self.frames.pop(0)
            return False, None

        def release(self):
            self.released = True

    capture = FakeCapture()
    reader = run.PrefetchedVideoReader(capture, buffer_size=1)

    try:
        assert [reader.read() for _ in range(3)] == [
            (True, "frame-1"),
            (True, "frame-2"),
            (True, "frame-3"),
        ]
        assert reader.read() == (False, None)
    finally:
        reader.close()

    assert capture.released


def test_prefetched_video_reader_propagates_capture_errors():
    class FakeCapture:
        def read(self):
            raise OSError("decoder failed")

        def release(self):
            pass

    reader = run.PrefetchedVideoReader(FakeCapture(), buffer_size=1)

    try:
        with pytest.raises(OSError, match="decoder failed"):
            reader.read()
    finally:
        reader.close()


def test_draw_results_shows_small_white_slot_id_without_background(monkeypatch):
    labels = []
    put_text = run.cv2.putText

    def record_text(*args, **kwargs):
        labels.append(args)
        return put_text(*args, **kwargs)

    monkeypatch.setattr(run.cv2, "putText", record_text)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    results = [{"index": 104, "box": (20, 30, 40, 20), "state": "occupied"}]

    output = run.draw_results(
        frame,
        results,
        {
            "configured": 1,
            "empty": 0,
            "occupied": 1,
            "unknown": 0,
            "invalid": 0,
        },
        30.0,
    )

    id_label = next(call for call in labels if call[1] == "104")
    assert output.shape == frame.shape
    assert id_label[4] == 0.5
    assert id_label[5] == (255, 255, 255)
    assert id_label[6] == 1


def test_create_output_dir_uses_a_unique_directory(tmp_path, monkeypatch):
    class FixedDatetime:
        @staticmethod
        def now():
            return DateTime(2026, 11, 1, 15, 0, 0)

    monkeypatch.setattr(run, "BASE_DIR", tmp_path)
    monkeypatch.setattr(run, "datetime", FixedDatetime)

    first_dir = run.create_output_dir()
    sentinel = first_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    second_dir = run.create_output_dir()

    assert first_dir != second_dir
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert second_dir.exists()


@pytest.mark.parametrize("confidence", ["-0.01", "1.01", "nan", "inf", "-inf"])
def test_main_rejects_invalid_confidence(confidence, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--confidence", confidence],
    )

    with pytest.raises(SystemExit) as exc_info:
        run.main()

    assert exc_info.value.code == 2


def test_main_rejects_invalid_prefetch_size(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--prefetch-frames", "0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run.main()

    assert exc_info.value.code == 2


def test_main_fails_and_releases_resources_when_video_writer_is_unavailable(tmp_path, monkeypatch):
    class FakeDetector:
        slots = []

        def __init__(self, **kwargs):
            pass

    class FakeCapture:
        released = False

        def __init__(self, source):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                run.cv2.CAP_PROP_FRAME_WIDTH: 640,
                run.cv2.CAP_PROP_FRAME_HEIGHT: 480,
                run.cv2.CAP_PROP_FPS: 30,
            }
            return values[prop]

        def release(self):
            self.released = True

    class FakeWriter:
        released = False

        def __init__(self, *args):
            pass

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    capture = FakeCapture("source.mp4")
    writer = FakeWriter()
    monkeypatch.setattr(run, "CNNParkingDetector", FakeDetector)
    monkeypatch.setattr(run.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(run.cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(run, "create_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--source", "source.mp4", "--no-display"],
    )

    with pytest.raises(RuntimeError, match="Cannot open output video writer"):
        run.main()

    assert capture.released
    assert writer.released
