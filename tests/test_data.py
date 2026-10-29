from pathlib import Path

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CLASSES = ("empty", "not_empty")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(directory):
    """Return supported image files directly contained in a dataset class folder."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("class_name", CLASSES)
def test_dataset_split_contains_readable_images(split, class_name):
    """Check each train/validation/test class folder exists and its images decode."""
    class_dir = DATA_ROOT / split / class_name
    assert class_dir.is_dir(), f"Missing dataset folder: {class_dir}"

    files = image_files(class_dir)
    assert files, f"No image files found in {class_dir}"

    unreadable = [path for path in files if cv2.imread(str(path)) is None]
    assert not unreadable, f"Unreadable images in {class_dir}: {unreadable[:5]}"


@pytest.mark.parametrize("class_name", CLASSES)
def test_video_groups_do_not_leak_between_dataset_splits(class_name):
    """Ensure frames sharing a filename group stay in only one split per class."""
    groups_by_split = {}
    for split in SPLITS:
        class_dir = DATA_ROOT / split / class_name
        assert class_dir.is_dir(), f"Missing dataset folder: {class_dir}"
        groups_by_split[split] = {
            path.stem.split("_")[0] for path in image_files(class_dir)
        }

    for index, split in enumerate(SPLITS):
        for other_split in SPLITS[index + 1 :]:
            shared_groups = groups_by_split[split] & groups_by_split[other_split]
            assert not shared_groups, (
                f"Video groups leak between {split} and {other_split} "
                f"for class {class_name}: {sorted(shared_groups)}"
            )
