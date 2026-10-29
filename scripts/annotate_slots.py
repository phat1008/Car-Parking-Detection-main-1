import argparse
import json
import math
from pathlib import Path

import cv2

WINDOW_NAME = "Annotate Slots"
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 850
MIN_SLOT_SIZE = 10

drawing = False
ix = iy = -1
boxes = []
current_frame = None
display_image = None
source_width = source_height = 0
display_width = display_height = 0
scale_x = scale_y = 1.0


def calculate_display_size(width, height):
    """Fit the source image within a bounded preview while preserving its aspect ratio."""
    scale = min(
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
        1.0,
    )
    return max(1, round(width * scale)), max(1, round(height * scale))


def to_source_coordinates(x, y):
    """Convert preview mouse coordinates to clamped source-image coordinates."""
    source_x = round(x / scale_x)
    source_y = round(y / scale_y)
    return (
        max(0, min(source_x, source_width - 1)),
        max(0, min(source_y, source_height - 1)),
    )


def validate_config(config):
    """Return a validation message, or None when a parking layout can be saved."""
    width = config.get("source_width")
    height = config.get("source_height")
    for name, value in (("source_width", width), ("source_height", height)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            return f"{name} phải là số hữu hạn lớn hơn 0."

    raw_boxes = config.get("boxes")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        return "Config phải có ít nhất một parking slot."

    seen_ids = set()
    has_enabled_slot = False
    for index, box in enumerate(raw_boxes):
        if not isinstance(box, dict):
            return f"Slot index {index} phải là object."

        missing_keys = [
            key for key in ("id", "x", "y", "width", "height") if key not in box
        ]
        if missing_keys:
            return f"Slot index {index} thiếu: {missing_keys}"

        slot_id = box["id"]
        if isinstance(slot_id, bool) or not isinstance(slot_id, (int, str)):
            return f"ID slot index {index} phải là số nguyên hoặc chuỗi."
        if slot_id in seen_ids:
            return f"ID slot bị trùng: {slot_id}"
        seen_ids.add(slot_id)

        coordinates = {}
        for key in ("x", "y", "width", "height"):
            value = box[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return f"Slot {slot_id}: {key} phải là số hữu hạn."
            coordinates[key] = value

        x = coordinates["x"]
        y = coordinates["y"]
        slot_width = coordinates["width"]
        slot_height = coordinates["height"]
        if x < 0 or y < 0:
            return f"Slot {slot_id} có tọa độ âm."
        if slot_width <= 0 or slot_height <= 0:
            return f"Slot {slot_id} phải có width/height lớn hơn 0."
        if x + slot_width > width or y + slot_height > height:
            return f"Slot {slot_id} vượt ngoài ảnh nguồn " f"({width}x{height})."

        enabled = box.get("enabled", True)
        if not isinstance(enabled, bool):
            return f"Slot {slot_id}: enabled phải là true hoặc false."
        has_enabled_slot = has_enabled_slot or enabled

    if not has_enabled_slot:
        return "Config phải có ít nhất một parking slot enabled."

    return None


def draw_grid_ui():
    """Render source-coordinate boxes and controls on the scaled preview."""
    preview = cv2.resize(
        current_frame,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )

    for box in boxes:
        x = round(box["x"] * scale_x)
        y = round(box["y"] * scale_y)
        width = round(box["width"] * scale_x)
        height = round(box["height"] * scale_y)
        color = (0, 255, 0) if box.get("enabled", True) else (128, 128, 128)

        cv2.rectangle(
            preview,
            (x, y),
            (x + width, y + height),
            color,
            2,
        )
        cv2.putText(
            preview,
            f"ID:{box['id']}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        f"Total Slots: {len(boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    instructions = (
        "Left click + drag: draw slot | z: undo | r: reset",
        "s: save config | q: quit",
    )
    for index, text in enumerate(instructions):
        cv2.putText(
            preview,
            text,
            (20, 75 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return preview


def show_current_layout():
    """Refresh the annotation window from the current source frame and boxes."""
    global display_image
    display_image = draw_grid_ui()
    cv2.imshow(WINDOW_NAME, display_image)


def mouse_callback(event, x, y, flags, param):
    """Draw slots in source coordinates even when the preview is resized."""
    global drawing, ix, iy, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = to_source_coordinates(x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        preview = draw_grid_ui()
        end_x, end_y = to_source_coordinates(x, y)
        cv2.rectangle(
            preview,
            (round(ix * scale_x), round(iy * scale_y)),
            (round(end_x * scale_x), round(end_y * scale_y)),
            (0, 165, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, preview)
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        end_x, end_y = to_source_coordinates(x, y)
        drawing = False

        x_min, x_max = sorted((ix, end_x))
        y_min, y_max = sorted((iy, end_y))
        width = x_max - x_min
        height = y_max - y_min

        if width > MIN_SLOT_SIZE and height > MIN_SLOT_SIZE:
            new_id = (
                max(
                    (box["id"] for box in boxes if isinstance(box.get("id"), int)),
                    default=-1,
                )
                + 1
            )
            boxes.append(
                {
                    "id": new_id,
                    "x": x_min,
                    "y": y_min,
                    "width": width,
                    "height": height,
                    "type": "regular",
                    "enabled": True,
                }
            )
        else:
            print(f"[WARNING] Slot phải lớn hơn {MIN_SLOT_SIZE}x{MIN_SLOT_SIZE} pixel.")

        show_current_layout()


def save_config(output_path, camera_id):
    """Validate the parking layout before atomically writing its JSON config."""
    config = {
        "camera_id": camera_id,
        "source_width": source_width,
        "source_height": source_height,
        "boxes": boxes,
    }
    error = validate_config(config)
    if error:
        print(f"[ERROR] Không lưu cấu hình: {error}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)
    temporary_path.replace(output_path)
    print(f"[SUCCESS] Đã lưu {len(boxes)} slot vào {output_path}")
    return True


def load_source(source_path):
    """Read the first video frame or load an image for slot annotation."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    if source_path.suffix.lower() not in video_extensions:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {source_path}")
        return frame

    capture = cv2.VideoCapture(str(source_path))
    try:
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Không thể đọc frame từ video: {source_path}")
        return frame
    finally:
        capture.release()


def main():
    global current_frame, boxes
    global source_width, source_height, display_width, display_height
    global scale_x, scale_y

    parser = argparse.ArgumentParser(
        description="Annotate parking slots on an image or video."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("carPark.mp4"),
        help="Image or video used to create the parking layout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/parking_config.json"),
        help="Path to the output JSON configuration.",
    )
    args = parser.parse_args()

    current_frame = load_source(args.source)
    source_height, source_width = current_frame.shape[:2]
    display_width, display_height = calculate_display_size(
        source_width,
        source_height,
    )
    scale_x = display_width / source_width
    scale_y = display_height / source_height

    camera_id = "camera_01"
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as file:
            existing_config = json.load(file)
        if not isinstance(existing_config, dict):
            raise ValueError(f"Config phải là JSON object: {args.output}")
        camera_id = existing_config.get("camera_id", camera_id)
        existing_boxes = existing_config.get("boxes", [])
        if not isinstance(existing_boxes, list):
            raise ValueError(f"Trường boxes phải là một danh sách: {args.output}")
        boxes = existing_boxes
        print(f"[INFO] Đã tải {len(boxes)} slot từ {args.output}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    show_current_layout()

    print("Kéo chuột trái để vẽ; z: undo, r: reset, s: save, q: quit.")
    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("z"):
                if boxes:
                    removed = boxes.pop()
                    print(f"[INFO] Đã undo slot ID {removed.get('id')}")
                    show_current_layout()
            elif key == ord("r"):
                boxes.clear()
                print("[INFO] Đã xóa toàn bộ slot.")
                show_current_layout()
            elif key == ord("s"):
                save_config(args.output, camera_id)
            elif key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
