import json
import os
import cv2

def validate_slot(slot_data, frame_width=None, frame_height=None, seen_ids=None):
    """
    Kiểm tra tính hợp lệ của 1 slot dictionary.
    Ném ra Exception với thông báo rõ ràng nếu cấu hình sai.
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
    w, h = slot_data["width"], slot_data["height"]

    # 2. Kiểm tra tọa độ âm
    if x < 0 or y < 0:
        raise ValueError(f"Slot ID {slot_id} có tọa độ âm: (x={x}, y={y})")

    # 3. Kiểm tra kích thước lớn hơn 0
    if w <= 0 or h <= 0:
        raise ValueError(f"Slot ID {slot_id} có kích thước không hợp lệ: (width={w}, height={h})")

    # 4. Kiểm tra ô có vượt quá kích thước frame hay không (nếu có truyền kích thước frame)
    if frame_width is not None and frame_height is not None:
        if x + w > frame_width or y + h > frame_height:
            raise ValueError(
                f"Slot ID {slot_id} vượt quá kích thước frame! "
                f"Slot bounds: (x_max={x+w}, y_max={y+h}), Frame: ({frame_width}x{frame_height})"
            )

    return True


def load_slots(json_path, frame_width=None, frame_height=None):
    """
    Đọc file JSON cấu hình và trả về danh sách các slot đã qua kiểm tra.
    Hỗ trợ trường 'enabled' (chỉ trả về các slot có enabled == True).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Lấy thông số frame từ file config nếu không truyền vào
    if frame_width is None:
        frame_width = data.get("source_width")
    if frame_height is None:
        frame_height = data.get("source_height")

    raw_boxes = data.get("boxes", [])
    valid_slots = []
    seen_ids = set()

    for idx, box in enumerate(raw_boxes):
        try:
            validate_slot(box, frame_width, frame_height, seen_ids)
            # 5. Hỗ trợ enabled: true/false (mặc định là True nếu không khai báo)
            if box.get("enabled", True):
                valid_slots.append(box)
        except ValueError as e:
            # 6. In lỗi rõ ràng khi config sai
            print(f"[ERROR CONFIG] Lỗi tại phần tử thứ {idx}: {e}")
            raise e

    return valid_slots


def get_slot_crop(frame, slot):
    """
    Cắt vùng ảnh (crop) của slot từ frame dựa trên thông tin dict slot.
    """
    x = int(slot["x"])
    y = int(slot["y"])
    w = int(slot["width"])
    h = int(slot["height"])

    crop = frame[y:y+h, x:x+w]
    return crop