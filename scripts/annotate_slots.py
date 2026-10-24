import os
import json
import argparse
from pathlib import Path
import cv2

# Biến toàn cục phục vụ tương tác chuột
drawing = False
ix, iy = -1, -1
boxes = []  # Danh sách lưu các ô hiện tại: chứa dict thông tin ô
current_frame = None
display_image = None

def draw_grid_ui(img):
    """Vẽ lại toàn bộ các ô đã lưu và thông tin hướng dẫn lên khung hình."""
    overlay = img.copy()
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        box_id = box["id"]
        enabled = box.get("enabled", True)
        
        # Màu sắc: Xanh lá nếu bật, Xám nếu tắt
        color = (0, 255, 0) if enabled else (128, 128, 128)
        
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, f"ID:{box_id}", (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Hiển thị tổng số ô hiện tại
    cv2.putText(overlay, f"Total Slots: {len(boxes)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Hướng dẫn phím tắt trên màn hình
    instructions = [
        "Controls:",
        "Left Click + Drag: Draw slot",
        "z: Undo last",
        "r: Reset all",
        "s: Save config",
        "q: Quit"
    ]
    y_offset = 80
    for text in instructions:
        cv2.putText(overlay, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 25

    return overlay

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, display_image, current_frame, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = draw_grid_ui(current_frame)
            # Vẽ hình chữ nhật tạm thời khi đang kéo chuột
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 165, 255), 2)
            cv2.imshow("Annotate Slots", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min = min(ix, x)
        y_min = min(iy, y)
        width = abs(ix - x)
        height = abs(iy - y)

        # Kiểm tra kích thước tối thiểu (ví dụ: width > 10 và height > 10)
        if width > 10 and height > 10:
            # Tự động cấp ID mới tăng dần
            new_id = len(boxes) if not boxes else max(b["id"] for b in boxes) + 1
            
            new_box = {
                "id": new_id,
                "x": int(x_min),
                "y": int(y_min),
                "width": int(width),
                "height": int(height),
                "type": "regular",
                "enabled": True
            }
            boxes.append(new_box)
        
        display_image = draw_grid_ui(current_frame)
        cv2.imshow("Annotate Slots", display_image)

def main():
    global current_frame, display_image, boxes

    parser = argparse.ArgumentParser(description="Công cụ gán nhãn ô đỗ xe (Parking Slot Annotator)")
    parser.add_argument("--source", type=str, default="carPark.mp4", help="Đường dẫn đến file video hoặc ảnh")
    parser.add_argument("--output", type=str, default="config/parking_config.json", help="Đường dẫn file JSON đầu ra")
    args = parser.parse_args()

    source_path = args.source
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Đọc frame đầu tiên của video (hoặc đọc ảnh nếu source là ảnh)
    if os.path.exists(source_path) and source_path.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(source_path)
        ret, current_frame = cap.read()
        cap.release()
        if not ret:
            print(f"[ERROR] Không thể đọc frame từ video: {source_path}")
            return
    else:
        # Thử đọc trực tiếp bằng cv2.imread nếu là file ảnh hoặc đường dẫn khác
        current_frame = cv2.imread(source_path)
        if current_frame is None:
            print(f"[ERROR] Không tìm thấy hoặc không đọc được nguồn: {source_path}")
            return

    source_height, source_width = current_frame.shape[:2]

    # Nếu file JSON cấu hình đã tồn tại, load trước để không bị mất vị trí đã lưu cũ
    camera_id = "camera_01"
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                camera_id = existing_data.get("camera_id", "camera_01")
                boxes = existing_data.get("boxes", [])
                print(f"[INFO] Đã tải thành công {len(boxes)} ô từ cấu hình cũ: {output_path}")
        except Exception as e:
            print(f"[WARNING] Không thể đọc file cấu hình cũ, tạo mới từ đầu: {e}")

    cv2.namedWindow("Annotate Slots", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate Slots", mouse_callback)

    display_image = draw_grid_ui(current_frame)
    cv2.imshow("Annotate Slots", display_image)

    print("[INFO] Đã khởi động công cụ. Sử dụng chuột và các phím tắt (z, r, s, q).")

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Phím 'z': Undo ô cuối cùng
        if key == ord('z'):
            if boxes:
                removed = boxes.pop()
                print(f"[INFO] Đã xóa (Undo) ô ID: {removed['id']}")
                display_image = draw_grid_ui(current_frame)
                cv2.imshow("Annotate Slots", display_image)

        # Phím 'r': Reset / xóa tất cả các ô
        elif key == ord('r'):
            boxes.clear()
            print("[INFO] Đã xóa toàn bộ các ô.")
            display_image = draw_grid_ui(current_frame)
            cv2.imshow("Annotate Slots", display_image)

        # Phím 's': Lưu cấu hình ra file JSON
        elif key == ord('s'):
            config_data = {
                "camera_id": camera_id,
                "source_width": source_width,
                "source_height": source_height,
                "boxes": boxes
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            print(f"[SUCCESS] Đã lưu thành công {len(boxes)} ô vào {output_path}")

        # Phím 'q': Thoát chương trình
        elif key == ord('q'):
            print("[INFO] Thoát chương trình gán nhãn.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()