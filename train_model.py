import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASET_DIR = Path("data/clf-data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.p"

IMAGE_SIZE = (15, 15)
RANDOM_STATE = 42


def load_dataset():
    features = []
    labels = []

    for label in ["empty", "not_empty"]:
        folder = DATASET_DIR / label

        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        image_paths = sorted(
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
        )

        print(f"Loading {label}: {len(image_paths)} images")

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"Warning: cannot read {image_path}")
                continue

            if image.shape[2] != 3:
                print(f"Warning: invalid channels: {image_path}")
                continue

            image = cv2.resize(
                image,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA,
            )

            # Giữ BGR vì cả train và inference đều dùng OpenCV.
            image = image.astype(np.float32) / 255.0

            # 15x15x3 -> 675 features
            vector = image.reshape(-1)

            features.append(vector)
            labels.append(label)

    if not features:
        raise RuntimeError("Dataset is empty")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    return X, y


def main():
    X, y = load_dataset()

    print("X shape:", X.shape)
    print("Labels:", np.unique(y, return_counts=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["empty", "not_empty"],
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["empty", "not_empty"],
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Input format: 15x15x3")
    print("Color format: BGR")
    print("Normalization: pixel / 255.0")
    print("Labels: empty, not_empty")


if __name__ == "__main__":
    main()