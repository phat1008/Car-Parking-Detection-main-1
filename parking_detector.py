from src.cnn_classifier import CNNParkingClassifier
from src.slot_manager import SlotManager


class CNNParkingDetector:
    def __init__(
        self,
        model_path,
        config_path,
        confidence_threshold=0.60,
    ):
        self.slot_manager = SlotManager(config_path)

        self.classifier = CNNParkingClassifier(
            model_path=model_path,
            image_size=(64, 64),
            confidence_threshold=confidence_threshold,
        )

        self.slots = self.slot_manager.slots

    def classify_frame(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Frame đầu vào rỗng")

        slot_items = self.slot_manager.crop_slots(frame)
        crops = [item["crop"] for item in slot_items]

        predictions = self.classifier.predict_batch(crops)

        results = []

        for slot_item, prediction in zip(
            slot_items,
            predictions,
        ):
            results.append(
                {
                    "index": slot_item["id"],
                    "box": slot_item["box"],
                    "type": slot_item["type"],
                    **prediction,
                }
            )

        return {
            "results": results,
            "total": len(results),
        }
