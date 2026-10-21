from pathlib import Path

import cv2


DATASET_DIR = Path("data/clf-data")

for class_name in ["empty", "not_empty"]:
    folder = DATASET_DIR / class_name
    files = list(folder.glob("*.jpg"))

    print(f"{class_name}: {len(files)} images")

    invalid_count = 0

    for image_path in files[:100]:
        image = cv2.imread(str(image_path))

        if image is None:
            invalid_count += 1
            continue

        if len(image.shape) != 3 or image.shape[2] != 3:
            invalid_count += 1

    print(f"Invalid sample images: {invalid_count}")