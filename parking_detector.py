import json
from pathlib import Path

import cv2

from classifier import ParkingClassifier


class ParkingDetector:
    def __init__(self, model_path, boxes_path):
        self.classifier = ParkingClassifier(model_path)
        self.boxes = self.load_boxes(boxes_path)

    @staticmethod
    def load_boxes(boxes_path):
        boxes_path = Path(boxes_path)

        with boxes_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        return [tuple(box) for box in config["boxes"]]

    def classify_frame(self, frame):
        results = []
        empty_count = 0

        for index, box in enumerate(self.boxes):
            x, y, width, height = box

            crop = frame[y:y + height, x:x + width]

            if crop.size == 0:
                results.append(
                    {
                        "index": index,
                        "box": box,
                        "label": "invalid",
                        "is_empty": False,
                        "confidence": None,
                    }
                )
                continue

            prediction = self.classifier.predict(crop)

            if prediction["is_empty"]:
                empty_count += 1

            results.append(
                {
                    "index": index,
                    "box": box,
                    **prediction,
                }
            )

        total = len(self.boxes)
        occupied_count = total - empty_count

        return {
            "results": results,
            "total": total,
            "empty": empty_count,
            "occupied": occupied_count,
        }

    def draw_results(self, frame, analysis):
        output = frame.copy()

        for item in analysis["results"]:
            x, y, width, height = item["box"]

            if item["label"] == "empty":
                color = (0, 255, 0)
                text = "EMPTY"
            elif item["label"] == "not_empty":
                color = (0, 0, 255)
                text = "OCCUPIED"
            else:
                color = (128, 128, 128)
                text = "INVALID"

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                color,
                2,
            )

            cv2.putText(
                output,
                f"{item['index']}: {text}",
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            output,
            f"Empty: {analysis['empty']}/{analysis['total']}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"Occupied: {analysis['occupied']}/{analysis['total']}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return output