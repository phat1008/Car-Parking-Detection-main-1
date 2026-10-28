import json
from pathlib import Path


class SlotManager:
    def __init__(self, config_path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file config: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.source_width = int(self.config["source_width"])
        self.source_height = int(self.config["source_height"])
        self.camera_id = self.config.get("camera_id", "camera_01")

        self.slots = self._validate_slots(
            self.config.get("boxes", [])
        )

    def _validate_slots(self, raw_slots):
        seen_ids = set()
        valid_slots = []

        for index, slot in enumerate(raw_slots):
            required_keys = [
                "id",
                "x",
                "y",
                "width",
                "height",
            ]

            missing_keys = [
                key for key in required_keys
                if key not in slot
            ]

            if missing_keys:
                raise ValueError(
                    f"Slot index {index} thiếu: {missing_keys}"
                )

            slot_id = slot["id"]

            if slot_id in seen_ids:
                raise ValueError(
                    f"ID slot bị trùng: {slot_id}"
                )

            seen_ids.add(slot_id)

            x = slot["x"]
            y = slot["y"]
            width = slot["width"]
            height = slot["height"]

            if not all(
                isinstance(value, (int, float))
                for value in [x, y, width, height]
            ):
                raise ValueError(
                    f"Slot {slot_id} có tọa độ/kích thước không phải số"
                )

            if x < 0 or y < 0:
                raise ValueError(
                    f"Slot {slot_id} có tọa độ âm"
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Slot {slot_id} có kích thước không hợp lệ"
                )

            if x + width > self.source_width:
                raise ValueError(
                    f"Slot {slot_id} vượt source_width"
                )

            if y + height > self.source_height:
                raise ValueError(
                    f"Slot {slot_id} vượt source_height"
                )

            if slot.get("enabled", True):
                valid_slots.append(slot)

        if not valid_slots:
            raise ValueError(
                "Config không có parking slot enabled"
            )

        return valid_slots

    def scaled_box(
        self,
        slot,
        frame_width,
        frame_height,
    ):
        scale_x = frame_width / self.source_width
        scale_y = frame_height / self.source_height

        x = round(slot["x"] * scale_x)
        y = round(slot["y"] * scale_y)
        width = round(slot["width"] * scale_x)
        height = round(slot["height"] * scale_y)

        x = max(0, min(x, frame_width))
        y = max(0, min(y, frame_height))

        width = max(
            0,
            min(width, frame_width - x),
        )

        height = max(
            0,
            min(height, frame_height - y),
        )

        return x, y, width, height

    def crop_slots(self, frame):
        frame_height, frame_width = frame.shape[:2]
        slot_items = []

        for slot in self.slots:
            box = self.scaled_box(
                slot,
                frame_width,
                frame_height,
            )

            x, y, width, height = box

            crop = frame[
                y:y + height,
                x:x + width,
            ]

            slot_items.append(
                {
                    "id": slot["id"],
                    "box": box,
                    "crop": crop,
                    "type": slot.get("type", "regular"),
                }
            )

        return slot_items