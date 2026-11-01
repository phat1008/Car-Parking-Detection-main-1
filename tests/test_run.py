from datetime import datetime as DateTime

import pytest

import scripts.run as run


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
