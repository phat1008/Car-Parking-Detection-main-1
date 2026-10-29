import numpy as np
import pytest

from src.cnn_classifier import CNNParkingClassifier


def create_uninitialized_classifier(image_size):
    """Create the preprocessing component without loading a model."""
    classifier = object.__new__(CNNParkingClassifier)
    classifier.image_size = image_size
    return classifier


def test_preprocess_resizes_converts_bgr_to_rgb_and_keeps_float32():
    """Verify crop preprocessing returns the model's RGB float32 input format."""
    classifier = create_uninitialized_classifier((1, 1))
    bgr_crop = np.array([[[10, 20, 30]]], dtype=np.uint8)

    processed = classifier.preprocess(bgr_crop)

    assert processed.shape == (1, 1, 3)
    assert processed.dtype == np.float32
    np.testing.assert_array_equal(processed[0, 0], [30.0, 20.0, 10.0])


def test_preprocess_resizes_to_configured_width_and_height():
    """Check OpenCV resize uses the configured (width, height) convention."""
    classifier = create_uninitialized_classifier((8, 6))
    crop = np.zeros((20, 10, 3), dtype=np.uint8)

    processed = classifier.preprocess(crop)

    assert processed.shape == (6, 8, 3)


@pytest.mark.parametrize(
    "crop",
    [None, np.empty((0, 5, 3), dtype=np.uint8)],
    ids=["none", "empty-array"],
)
def test_preprocess_rejects_empty_crop(crop):
    """Ensure malformed slot crops raise an explicit validation error."""
    classifier = create_uninitialized_classifier((64, 64))

    with pytest.raises(ValueError, match="Parking slot crop rỗng"):
        classifier.preprocess(crop)
