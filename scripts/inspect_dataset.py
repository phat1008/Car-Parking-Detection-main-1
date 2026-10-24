from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/clf-data")
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(folder: Path):
    return [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]


def main():
    counts = {}

    for label in ("empty", "not_empty"):
        folder = DATASET_DIR / label
        images = collect_images(folder)
        counts[label] = len(images)
        print(f"{label}: {len(images)} images")

    total = sum(counts.values())
    print(f"Total: {total} images")

    if total == 0:
        raise RuntimeError("Dataset is empty")

    for label, count in counts.items():
        if count == 0:
            raise RuntimeError(f"Missing class or empty folder: {label}")


if __name__ == "__main__":
    main()