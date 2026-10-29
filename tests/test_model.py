import numpy as np
import pytest
import tensorflow as tf

from src.cnn_classifier import CNNParkingClassifier


class MeanProbabilityModel:
    """Provide deterministic probabilities while exercising the classifier's TensorFlow call."""

    def __call__(self, batch, training=False):
        mean = tf.reduce_mean(batch, axis=(1, 2, 3))
        return tf.expand_dims(mean / 255.0, axis=-1)


def create_classifier(tmp_path, monkeypatch, model=None):
    """Construct a classifier with a lightweight deterministic model for unit tests."""
    model_path = tmp_path / "fake-model.keras"
    model_path.touch()
    monkeypatch.setattr(
        tf.keras.models,
        "load_model",
        lambda path: model if model is not None else MeanProbabilityModel(),
    )
    return CNNParkingClassifier(model_path, image_size=(4, 4))


def test_decode_probability_maps_classes_and_uncertainty():
    """Verify positive, negative, and below-threshold outputs map to documented states."""
    classifier = object.__new__(CNNParkingClassifier)
    classifier.confidence_threshold = 0.60

    occupied = classifier._decode_probability(0.8)
    empty = classifier._decode_probability(0.2)
    uncertain = classifier._decode_probability(0.5)

    assert occupied == {
        "state": "occupied",
        "label": "occupied",
        "confidence": 0.8,
        "probability_occupied": 0.8,
        "is_empty": False,
    }
    assert empty["state"] == "empty"
    assert empty["is_empty"] is True
    assert uncertain["state"] == "unknown"
    assert uncertain["confidence"] == 0.5


def test_predict_batch_keeps_results_aligned_and_marks_invalid_crops(
    tmp_path, monkeypatch
):
    """Check batched inference retains input order and does not infer for empty crops."""
    classifier = create_classifier(tmp_path, monkeypatch)
    crops = [
        np.full((8, 8, 3), 255, dtype=np.uint8),
        None,
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 128, dtype=np.uint8),
    ]

    results = classifier.predict_batch(crops)

    assert len(results) == len(crops)
    assert results[0]["state"] == "occupied"
    assert results[1] == {
        "state": "invalid",
        "label": "invalid",
        "confidence": 0.0,
        "probability_occupied": 0.0,
        "is_empty": False,
    }
    assert results[2]["state"] == "empty"
    assert results[3]["state"] == "unknown"
    assert results[3]["confidence"] < classifier.confidence_threshold


def test_predict_returns_invalid_for_empty_crop(tmp_path, monkeypatch):
    """Ensure a single empty crop uses the same invalid result contract as batch inference."""
    classifier = create_classifier(tmp_path, monkeypatch)

    result = classifier.predict(np.empty((0, 0, 3), dtype=np.uint8))

    assert result["state"] == "invalid"
    assert result["label"] == "invalid"
    assert result["is_empty"] is False
    assert result["confidence"] == 0.0


def test_missing_model_fails_with_actionable_error(tmp_path):
    """Confirm model loading fails immediately when its file is missing."""
    with pytest.raises(FileNotFoundError, match="Không tìm thấy model CNN"):
        CNNParkingClassifier(tmp_path / "missing.keras")
