import json
from pathlib import Path
import cv2

from classifier import ParkingClassifier


class ParkingDetector:
    def __init__(self, model_path, boxes_path):
        self.classifier = ParkingClassifier(model_path)
        # Tải cấu hình file JSON chuẩn hóa
        config_data = self.load_config(boxes_path)
        self.camera_id = config_data.get("camera_id", "camera_01")
        self.source_width = config_data.get("source_width")
        self.source_height = config_data.get("source_height")
        self.boxes = config_data.get("boxes", [])

    @staticmethod
    def validate_slot(slot_data, frame_width=None, frame_height=None, seen_ids=None):
        """
        Kiểm tra tính hợp lệ của từng slot dictionary.
        Ném ra ValueError kèm thông báo chi tiết nếu config sai.
        """
        required_keys = ["id", "x", "y", "width", "height"]
        for key in required_keys:
            if key not in slot_data:
                raise ValueError(f"Thiếu trường bắt buộc '{key}' trong slot: {slot_data}")

        slot_id = slot_data["id"]

        # 1. Kiểm tra ID trùng lặp
        if seen_ids is not None:
            if slot_id in seen_ids:
                raise ValueError(f"ID slot bị trùng lặp: {slot_id}")
            seen_ids.add(slot_id)

        x, y = slot_data["x"], slot_data["y"]
        width, height = slot_data["width"], slot_data["height"]

        # 2. Kiểm tra tọa độ âm
        if x < 0 or y < 0:
            raise ValueError(f"Slot ID {slot_id} có tọa độ âm: (x={x}, y={y})")

        # 3. Kiểm tra kích thước lớn hơn 0
        if width <= 0 or height <= 0:
            raise ValueError(f"Slot ID {slot_id} có kích thước không hợp lệ: (width={width}, height={height})")

        # 4. Kiểm tra ô có vượt quá kích thước frame hay không (nếu có thông tin)
        if frame_width is not None and frame_height is not None:
            if x + width > frame_width or y + height > frame_height:
                raise ValueError(
                    f"Slot ID {slot_id} vượt quá kích thước frame! "
                    f"Slot bounds: (x_max={x+width}, y_max={y+height}), Frame: ({frame_width}x{frame_height})"
                )

        return True

    @classmethod
    def load_config(cls, boxes_path):
        """
        Đọc và validate toàn bộ cấu hình từ file JSON.
        Chỉ giữ lại các slot có enabled == True.
        """
        boxes_path = Path(boxes_path)
        if not boxes_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {boxes_path}")

        with boxes_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        frame_width = config.get("source_width")
        frame_height = config.get("source_height")
        raw_boxes = config.get("boxes", [])

        valid_boxes = []
        seen_ids = set()

        for idx, box in enumerate(raw_boxes):
            try:
                # Validate từng box theo tiêu chí kỹ thuật
                cls.validate_slot(box, frame_width, frame_height, seen_ids)
                
                # 5. Hỗ trợ trường 'enabled' (mặc định True nếu không khai báo)
                if box.get("enabled", True):
                    valid_boxes.append(box)
            except ValueError as e:
                # 6. In lỗi rõ ràng thay vì làm crash chương trình
                print(f"[ERROR CONFIG] Lỗi tại box index {idx} (ID: {box.get('id', 'unknown')}): {e}")
                raise e

        config["boxes"] = valid_boxes
        return config

    def classify_frame(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Frame đầu vào rỗng")

        frame_height, frame_width = frame.shape[:2]
        results = []
        empty_count = 0

        for box in self.boxes:
            # Lấy thông tin tọa độ trực tiếp từ dictionary thay vì dùng tuple index
            slot_id = box["id"]
            x, y, width, height = self._box_for_frame(box, frame_width, frame_height)

            crop = frame[y:y + height, x:x + width]

            if crop.size == 0:
                results.append(
                    {
                        "index": slot_id,
                        "box": (x, y, width, height),
                        "label": "invalid",
                        "state": "invalid",
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
                    "index": slot_id,
                    "box": (x, y, width, height),
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

    def _box_for_frame(self, box, frame_width, frame_height):
        """Scale configured coordinates when the input frame was resized."""
        scale_x = (
            frame_width / self.source_width
            if self.source_width and self.source_width != frame_width
            else 1.0
        )
        scale_y = (
            frame_height / self.source_height
            if self.source_height and self.source_height != frame_height
            else 1.0
        )

        x = round(box["x"] * scale_x)
        y = round(box["y"] * scale_y)
        width = round(box["width"] * scale_x)
        height = round(box["height"] * scale_y)

        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return x, y, width, height

        # Clamp only the visible crop; the original config remains unchanged.
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)
        return x, y, width, height

    def draw_results(self, frame, analysis):
        output = frame.copy()

        for item in analysis["results"]:
            x, y, width, height = item["box"]

            if item["label"] == "empty":
                color = (0, 255, 0)
                text = "EMPTY"
            elif item["label"] in ("occupied", "not_empty"):
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