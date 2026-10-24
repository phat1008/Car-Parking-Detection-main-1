import cv2
import json
from pathlib import Path

# Cấu hình đường dẫn
VIDEO_PATH = "carPark.mp4"
CONFIG_DIR = Path("config")
CONFIG_PATH = CONFIG_DIR / "parking_config.json"

# Biến toàn cục phục vụ cho việc kéo thả chuột
drawing = False
ix, iy = -1, -1
current_boxes = []
base_frame = None
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    """Hàm xử lý sự kiện chuột để vẽ bounding box (kéo thả chuột trái)"""
    global drawing, ix, iy, current_boxes, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Tạo bản sao tạm để vẽ khung chữ nhật đang kéo
            frame_temp = frame_copy.copy()
            cv2.rectangle(frame_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            draw_instructions(frame_temp)
            cv2.imshow("Parking Space Annotator", frame_temp)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, x_max = min(ix, x), max(ix, x)
        y_min, y_max = min(iy, y), max(iy, y)
        
        # Chỉ lưu nếu box có kích thước hợp lệ (tránh click nhầm)
        if (x_max - x_min > 5) and (y_max - y_min > 5):
            current_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            print(f"Đã thêm vùng đỗ xe: {[x_min, y_min, x_max - x_min, y_max - y_min]}")
            
        update_display()

def draw_instructions(img):
    """Vẽ bảng hướng dẫn thao tác ở góc phải màn hình"""
    instructions = [
        "1. Keo chuot trai: chon vung do",
        "2. Nhan 's': luu layout",
        "3. Nhan 'd': nhan dien xe & report",
        "4. Nhan 'z': undo vung vua ve",
        "5. Nhan 'r': xoa toan bo",
        "6. Nhan 'q': thoat"
    ]
    
    # Thiết lập vị trí hiển thị ở góc phải phía trên
    h, w, _ = img.shape
    start_x = w - 380
    start_y = 40
    
    # Vẽ nền mờ cho bảng hướng dẫn dễ nhìn
    overlay = img.copy()
    cv2.rectangle(overlay, (start_x - 15, start_y - 25), (w - 20, start_y + len(instructions) * 30 + 10), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # In nội dung hướng dẫn lên khung hình
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (start_x, start_y + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def update_display():
    """Cập nhật lại hiển thị khung hình kèm theo tất cả các box đã vẽ"""
    global frame_copy
    frame_copy = base_frame.copy()
    
    # Vẽ lại toàn bộ các box đã lưu
    for box in current_boxes:
        x, y, w, h = box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    draw_instructions(frame_copy)
    cv2.imshow("Parking Space Annotator", frame_copy)

def main():
    global base_frame, frame_copy, current_boxes
    
    # Đọc frame đầu tiên từ video carPark.mp4
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, base_frame = cap.read()
    cap.release()
    
    if not ret:
        print("[Lỗi] Không thể đọc video từ đường dẫn:", VIDEO_PATH)
        return

    # Load layout cũ và chuẩn hóa lại cấu trúc [x, y, w, h] an toàn
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    current_boxes = []
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                raw_boxes = json.load(f)
                for box in raw_boxes:
                    if isinstance(box, list) and len(box) >= 4:
                        current_boxes.append(box[:4])
                print(f"Đã tải và chuẩn hóa {len(current_boxes)} vùng đỗ xe từ cấu hình cũ.")
            except json.JSONDecodeError:
                print("[Cảnh báo] File config lỗi định dạng, tạo mới danh sách vùng đỗ.")
                current_boxes = []

    frame_copy = base_frame.copy()
    update_display()
    
    cv2.namedWindow("Parking Space Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parking Space Annotator", 1280, 720)
    cv2.setMouseCallback("Parking Space Annotator", mouse_callback)
    
    print("=== BẮT ĐẦU CÔNG CỤ TẠO BOUNDING BOX ===")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Nhấn 's' để lưu layout vào config.json
        if key == ord('s'):
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(current_boxes, f, indent=4)
            print(f"[Thành công] Đã lưu {len(current_boxes)} vùng đỗ xe vào {CONFIG_PATH}")
            
        # Nhấn 'z' để Undo (xóa box vẽ gần nhất)
        elif key == ord('z'):
            if current_boxes:
                removed = current_boxes.pop()
                print(f"Đã xóa vùng vừa vẽ: {removed}")
                update_display()
                
        # Nhấn 'r' để xóa toàn bộ các box
        elif key == ord('r'):
            current_boxes.clear()
            print("Đã xóa toàn bộ các vùng đỗ xe.")
            update_display()
            
        # Nhấn 'q' để thoát
        elif key == ord('q'):
            print("Đang thoát chương trình...")
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()