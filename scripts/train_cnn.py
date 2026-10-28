from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42

MODEL_PATH = Path("models/parking_cnn.keras")
METADATA_PATH = Path("models/parking_cnn.metadata.json")
HISTORY_PATH = Path("reports/metrics/cnn_training_history.json")

CLASS_NAMES = ["empty", "not_empty"]


def create_small_cnn():
    model = models.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),

            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),

            layers.Rescaling(1.0 / 255.0),

            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def create_model():
    return create_small_cnn()


def load_dataset(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {TRAIN_DIR}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {VAL_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {TEST_DIR}")

    train_ds = load_dataset(TRAIN_DIR, shuffle=True)
    val_ds = load_dataset(VAL_DIR, shuffle=False)
    test_ds = load_dataset(TEST_DIR, shuffle=False)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = create_small_cnn()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    test_metrics = model.evaluate(test_ds, return_dict=True)

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "history": {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
                "test_metrics": {
                    key: float(value)
                    for key, value in test_metrics.items()
                },
            },
            file,
            indent=2,
        )

    metadata = {
        "model_type": "cnn",
        "image_size": list(IMG_SIZE),
        "color_format": "RGB",
        "normalization": "model_internal_rescaling",
        "classes": CLASS_NAMES,
        "positive_class": "not_empty",
        "production_threshold": 0.60,
        "test_metrics": {
            key: float(value)
            for key, value in test_metrics.items()
        },
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Metrics saved to: {HISTORY_PATH}")


if __name__ == "__main__":
    main()