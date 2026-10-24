import json
from pathlib import Path

def validate_parking_config(config_path="config/parking_config.json"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    # 1. Kiểm tra camera có kích thước hợp lệ
    width = config.get("source_width", 0)
    height = config.get("source_height", 0)
    if width <= 0 or height <= 0:
        raise ValueError("Lỗi: Kích thước camera (source_width/source_height) phải lớn hơn 0.")
        
    boxes = config.get("boxes", [])
    seen_ids = set()
    
    for box in boxes:
        box_id = box.get("id")
        x = box.get("x")
        y = box.get("y")
        w = box.get("width")
        h = box.get("height")
        
        # 2. ID không trùng
        if box_id in seen_ids:
            raise ValueError(f"Lỗi: Phát hiện ID box bị trùng lặp: {box_id}")
        seen_ids.add(box_id)
        
        # 3. Box không âm và width/height > 0
        if x < 0 or y < 0:
            raise ValueError(f"Lỗi: Box ID {box_id} có tọa độ âm (x={x}, y={y}).")
        if w <= 0 or h <= 0:
            raise ValueError(f"Lỗi: Box ID {box_id} có width/height phải > 0.")
            
        # 4. Box nằm trong ảnh nguồn
        if (x + w) > width or (y + h) > height:
            raise ValueError(f"Lỗi: Box ID {box_id} vượt quá giới hạn khung hình nguồn ({width}x{height}).")
            
    print(f"[Thành công] File cấu hình '{config_path}' hợp lệ với {len(boxes)} parking box!")

if __name__ == "__main__":
    validate_parking_config()