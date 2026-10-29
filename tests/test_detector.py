import json

import numpy as np
import pytest

import parking_detector
from parking_detector import CNNParkingDetector


class StubClassifier:
    """Fake classifier that records crops without loading a real CNN model."""

    def __init__(self, **kwargs):
        self.crops = None

    def predict_batch(self, crops):
        self.crops = crops
        return [
            {
                "state": "empty",
                "label": "empty",
                "confidence": 0.9,
                "probability_occupied": 0.1,
                "is_empty": True,
            }
            for _ in crops
        ]


def create_config(path):
    """Write a small camera layout with two enabled and one disabled slot."""
    config = {
        "source_width": 100,
        "source_height": 50,
        "boxes": [
            {
                "id": 10,
                "x": 10,
                "y": 5,
                "width": 20,
                "height": 10,
                "type": "regular",
                "enabled": True,
            },
            {
                "id": 11,
                "x": 60,
                "y": 20,
                "width": 10,
                "height": 10,
                "type": "compact",
                "enabled": True,
            },
            {
                "id": 12,
                "x": 80,
                "y": 20,
                "width": 10,
                "height": 10,
                "enabled": False,
            },
        ],
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_classify_frame_crops_enabled_slots_and_preserves_metadata(tmp_path, monkeypatch):
    """Verify frame classification scales boxes, skips disabled slots, and returns metadata."""
    monkeypatch.setattr(parking_detector, "CNNParkingClassifier", StubClassifier)
    config_path = create_config(tmp_path / "parking.json")
    detector = CNNParkingDetector("unused-model.keras", config_path)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    analysis = detector.classify_frame(frame)

    assert analysis["total"] == 2
    assert [item["index"] for item in analysis["results"]] == [10, 11]
    assert analysis["results"][0]["box"] == (20, 10, 40, 20)
    assert analysis["results"][1]["box"] == (120, 40, 20, 20)
    assert analysis["results"][1]["type"] == "compact"
    assert all(item["state"] == "empty" for item in analysis["results"])
    assert [crop.shape[:2] for crop in detector.classifier.crops] == [
        (20, 40),
        (20, 20),
    ]


@pytest.mark.parametrize(
    "frame",
    [None, np.empty((0, 0, 3), dtype=np.uint8)],
    ids=["none", "empty-array"],
)
def test_classify_frame_rejects_empty_input(tmp_path, monkeypatch, frame):
    """Ensure invalid video frames fail clearly instead of producing empty predictions."""
    monkeypatch.setattr(parking_detector, "CNNParkingClassifier", StubClassifier)
    detector = CNNParkingDetector(
        "unused-model.keras",
        create_config(tmp_path / "parking.json"),
    )

    with pytest.raises(ValueError, match="Frame đầu vào rỗng"):
        detector.classify_frame(frame)
