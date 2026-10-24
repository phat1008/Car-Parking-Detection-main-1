import cv2
import json
from pathlib import Path

# Cấu hình đường dẫn ảnh tĩnh và file config
IMAGE_PATH = "video_first_frame.jpg"
CONFIG_DIR = Path("config")
CONFIG_PATH = CONFIG_DIR / "parking_config.json"

# Biến toàn cục phục vụ cho việc kéo thả chuột
drawing = False
ix, iy = -1, -1
current_boxes = []
base_frame = None
scale_percent = 100  # Sẽ được tính tự động bên dưới
orig_w, orig_h = 0, 0

def validate_config(config_data):
    """Validate cấu hình box theo đúng tiêu chuẩn yêu cầu trước khi lưu"""
    width = config_data.get("source_width", 0)
    height = config_data.get("source_height", 0)
    
    if width <= 0 or height <= 0:
        return "Kích thước camera (source_width/source_height) phải lớn hơn 0."
        
    boxes = config_data.get("boxes", [])
    seen_ids = set()
    
    for box in boxes:
        box_id = box.get("id")
        x = box.get("x")
        y = box.get("y")
        w = box.get("width")
        h = box.get("height")
        
        # ID không trùng lặp
        if box_id in seen_ids:
            return f"Phát hiện ID box bị trùng lặp: {box_id}"
        seen_ids.add(box_id)
        
        # Box không âm, width/height > 0
        if x < 0 or y < 0:
            return f"Box ID {box_id} có tọa độ âm (x={x}, y={y})."
        if w <= 0 or h <= 0:
            return f"Box ID {box_id} có kích thước width/height phải > 0."
            
        # Box nằm hoàn toàn trong ảnh nguồn
        if (x + w) > width or (y + h) > height:
            return f"Box ID {box_id} vượt quá giới hạn khung hình nguồn ({width}x{height})."
            
    return None

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_boxes, base_frame, scale_percent
    
    # Quy đổi tọa độ từ màn hình hiển thị (đã scale) về kích thước ảnh gốc
    orig_x = int(x * (100 / scale_percent))
    orig_y = int(y * (100 / scale_percent))
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = orig_x, orig_y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_temp = base_frame.copy()
            # Vẽ lại tất cả các box cũ trên ảnh gốc
            for box in current_boxes:
                bx, by, bw, bh = box["x"], box["y"], box["width"], box["height"]
                cv2.rectangle(frame_temp, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            # Vẽ box đang kéo
            cv2.rectangle(frame_temp, (ix, iy), (orig_x, orig_y), (0, 255, 0), 2)
            
            # Resize hiển thị và vẽ hướng dẫn
            display_img(frame_temp)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, x_max = min(ix, orig_x), max(ix, orig_x)
        y_min, y_max = min(iy, orig_y), max(iy, orig_y)
        
        if (x_max - x_min > 5) and (y_max - y_min > 5):
            new_box = {
                "id": len(current_boxes),
                "x": int(x_min),
                "y": int(y_min),
                "width": int(x_max - x_min),
                "height": int(y_max - y_min),
                "type": "regular"
            }
            current_boxes.append(new_box)
            print(f"Đã thêm Box ID {new_box['id']} tại tọa độ: {[new_box['x'], new_box['y'], new_box['width'], new_box['height']]}")
            
        update_display()

def draw_instructions(img):
    instructions = [
        "1. Keo chuot trai: chon vung do",
        "2. Nhan 's': luu layout & validate",
        "3. Nhan 'z': undo vung vua ve",
        "4. Nhan 'r': xoa toan bo",
        "5. Nhan 'q': thoat"
    ]
    h, w, _ = img.shape
    start_x = w - 340
    start_y = 35
    
    # Vẽ nền mờ cho bảng hướng dẫn
    overlay = img.copy()
    cv2.rectangle(overlay, (start_x - 10, start_y - 20), (w - 15, start_y + len(instructions) * 28 + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (start_x, start_y + i * 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def display_img(img):
    """Hàm resize ảnh tự động theo tỷ lệ tối ưu để lấp đầy màn hình"""
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    draw_instructions(resized)
    cv2.imshow("Parking Space Annotator (Image)", resized)

def update_display():
    global base_frame
    frame_temp = base_frame.copy()
    for box in current_boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        box_id = box["id"]
        cv2.rectangle(frame_temp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_temp, str(box_id), (x + 5, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    display_img(frame_temp)

def main():
    global base_frame, current_boxes, scale_percent, orig_w, orig_h
    
    base_frame = cv2.imread(IMAGE_PATH)
    if base_frame is None:
        print(f"[Lỗi] Không thể đọc hình ảnh từ đường dẫn: {IMAGE_PATH}")
        return

    orig_h, orig_w = base_frame.shape[:2]
    target_height = 850 
    scale_percent = (target_height / orig_h) * 100

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    current_boxes = []
    
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Tương thích ngược: Hỗ trợ cả định dạng mới (dict) lẫn định dạng cũ (list)
                if isinstance(data, dict) and "boxes" in data:
                    raw_boxes = data["boxes"]
                elif isinstance(data, list):
                    raw_boxes = data
                else:
                    raw_boxes = []
                    
                for idx, box in enumerate(raw_boxes):
                    if isinstance(box, dict):
                        current_boxes.append({
                            "id": box.get("id", idx),
                            "x": box["x"],
                            "y": box["y"],
                            "width": box["width"],
                            "height": box["height"],
                            "type": box.get("type", "regular")
                        })
                    elif isinstance(box, list) and len(box) >= 4:
                        current_boxes.append({
                            "id": idx,
                            "x": box[0],
                            "y": box[1],
                            "width": box[2],
                            "height": box[3],
                            "type": "regular"
                        })
                print(f"Đã tải và chuẩn hóa {len(current_boxes)} vùng đỗ xe từ cấu hình cũ.")
            except json.JSONDecodeError:
                current_boxes = []

    # Sử dụng WINDOW_NORMAL để tự do co giãn cửa sổ
    cv2.namedWindow("Parking Space Annotator (Image)", cv2.WINDOW_NORMAL)
    
    win_w = int(orig_w * scale_percent / 100)
    win_h = int(orig_h * scale_percent / 100)
    cv2.resizeWindow("Parking Space Annotator (Image)", win_w, win_h)
    
    cv2.setMouseCallback("Parking Space Annotator (Image)", mouse_callback)
    
    update_display()
    
    print("=== BẮT ĐẦU CÔNG CỤ TẠO BOUNDING BOX TRÊN ẢNH TĨNH (CHUẨN STAGE 7) ===")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Đảm bảo ID liên tục theo thứ tự
            for idx, box in enumerate(current_boxes):
                box["id"] = idx
                
            config_data = {
                "camera_id": "camera_01",
                "source_width": orig_w,
                "source_height": orig_h,
                "boxes": current_boxes
            }
            
            # Validate trước khi lưu
            error = validate_config(config_data)
            if error:
                print(f"[Lỗi Validate]: {error}")
            else:
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=4)
                print(f"[Thành công] Đã validate và lưu {len(current_boxes)} vùng đỗ xe vào {CONFIG_PATH}")
                
        elif key == ord('z'):
            if current_boxes:
                removed = current_boxes.pop()
                print(f"Đã xóa Box ID {removed['id']}")
                update_display()
        elif key == ord('r'):
            current_boxes.clear()
            print("Đã xóa toàn bộ các vùng đỗ xe.")
            update_display()
        elif key == ord('q'):
            print("Đang thoát chương trình...")
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()