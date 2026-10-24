from pathlib import Path
import cv2

DATASET_DIR = Path("data/clf-data")


def main():
    invalid = []

    for path in DATASET_DIR.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(path))

        if image is None:
            invalid.append(path)

    print(f"Invalid images: {len(invalid)}")

    for path in invalid[:20]:
        print(path)


if __name__ == "__main__":
    main()