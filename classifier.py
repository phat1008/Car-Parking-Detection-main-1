import pickle
from pathlib import Path

import cv2
import numpy as np


class ParkingClassifier:
    def __init__(self, model_path):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with model_path.open("rb") as file:
            self.model = pickle.load(file)

    def preprocess(self, crop):
        if crop is None or crop.size == 0:
            raise ValueError("Empty parking-space crop")

        crop = cv2.resize(
            crop,
            (15, 15),
            interpolation=cv2.INTER_AREA,
        )

        # Phải giống train_model.py
        crop = crop.astype(np.float32) / 255.0

        features = crop.reshape(1, -1)

        return features

    def predict(self, crop):
        features = self.preprocess(crop)

        label = self.model.predict(features)[0]

        confidence = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

        return {
            "label": str(label),
            "is_empty": str(label) == "empty",
            "confidence": confidence,
        }