from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class CNNParkingClassifier:
    def __init__(
        self,
        model_path,
        image_size=(64, 64),
        confidence_threshold=0.60,
    ):
        self.model_path = Path(model_path)
        self.image_size = tuple(image_size)
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model CNN: {self.model_path}"
            )

        self.model = tf.keras.models.load_model(self.model_path)

    def preprocess(self, crop):
        if crop is None or crop.size == 0:
            raise ValueError("Parking slot crop rỗng")

        resized = cv2.resize(
            crop,
            self.image_size,
            interpolation=cv2.INTER_AREA,
        )

        # OpenCV đọc ảnh theo BGR.
        # CNN được train theo RGB.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Model đã có layers.Rescaling(1/255),
        # nên không chia /255 tại đây.
        return rgb.astype(np.float32)

    def _decode_probability(self, probability):
        probability = float(probability)
        confidence = max(probability, 1.0 - probability)

        if confidence < self.confidence_threshold:
            state = "unknown"
        elif probability >= 0.5:
            state = "occupied"
        else:
            state = "empty"

        return {
            "state": state,
            "label": state,
            "confidence": round(confidence, 4),
            "probability_occupied": round(probability, 4),
            "is_empty": state == "empty",
        }

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return {
                "state": "invalid",
                "label": "invalid",
                "confidence": 0.0,
                "probability_occupied": 0.0,
                "is_empty": False,
            }

        image = self.preprocess(crop)
        batch = np.expand_dims(image, axis=0)

        probability = self.model.predict(
            batch,
            verbose=0,
        )[0][0]

        return self._decode_probability(probability)

    def predict_batch(self, crops):
        results = [
            {
                "state": "invalid",
                "label": "invalid",
                "confidence": 0.0,
                "probability_occupied": 0.0,
                "is_empty": False,
            }
            for _ in crops
        ]

        valid_images = []
        valid_indices = []

        for index, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue

            valid_images.append(self.preprocess(crop))
            valid_indices.append(index)

        if not valid_images:
            return results

        batch = np.stack(valid_images, axis=0)

        probabilities = self.model.predict(
            batch,
            verbose=0,
        ).reshape(-1)

        for index, probability in zip(valid_indices, probabilities):
            results[index] = self._decode_probability(probability)

        return results