import json

import pytest

from scripts import annotate_slots


def valid_config():
    """Return a minimal valid parking layout for annotation tests."""
    return {
        "camera_id": "camera_test",
        "source_width": 100,
        "source_height": 80,
        "boxes": [
            {
                "id": 0,
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 15,
                "enabled": True,
            }
        ],
    }


def test_preview_size_preserves_aspect_ratio_and_respects_maximums():
    """Ensure large annotation images are reduced without changing aspect ratio."""
    width, height = annotate_slots.calculate_display_size(1920, 1080)

    assert (width, height) == (1280, 720)


def test_mouse_coordinates_map_back_to_source_image():
    """Verify scaled-preview clicks are converted into original-image coordinates."""
    annotate_slots.source_width = 1920
    annotate_slots.source_height = 1080
    annotate_slots.scale_x = 1280 / 1920
    annotate_slots.scale_y = 720 / 1080

    assert annotate_slots.to_source_coordinates(640, 360) == (960, 540)
    assert annotate_slots.to_source_coordinates(1280, 720) == (1919, 1079)


def test_config_validation_accepts_valid_layout():
    """Confirm a bounded layout with at least one enabled slot can be saved."""
    assert annotate_slots.validate_config(valid_config()) is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["boxes"][0].update(x=-1), "tọa độ âm"),
        (lambda config: config["boxes"][0].update(width=0), "width/height"),
        (lambda config: config["boxes"][0].update(x=90), "vượt ngoài ảnh"),
        (
            lambda config: config["boxes"].append(config["boxes"][0].copy()),
            "ID slot bị trùng",
        ),
        (
            lambda config: config["boxes"][0].update(enabled=False),
            "ít nhất một parking slot enabled",
        ),
    ],
)
def test_config_validation_rejects_invalid_layouts(mutate, message):
    """Check invalid coordinates, dimensions, duplicate IDs, and disabled layouts."""
    config = valid_config()
    mutate(config)

    assert message in annotate_slots.validate_config(config)


def test_save_config_does_not_write_invalid_layout(tmp_path, monkeypatch):
    """Ensure validation runs before a config file is created or overwritten."""
    output_path = tmp_path / "parking.json"
    output_path.write_text('{"existing": true}', encoding="utf-8")
    monkeypatch.setattr(annotate_slots, "source_width", 100)
    monkeypatch.setattr(annotate_slots, "source_height", 80)
    monkeypatch.setattr(annotate_slots, "boxes", [])

    assert annotate_slots.save_config(output_path, "camera_test") is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"existing": True}
