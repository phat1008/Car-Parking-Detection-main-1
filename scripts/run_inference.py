import cv2
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib
from collections import deque, Counter
from datetime import datetime
import csv

# 1. Tải cấu hình và mô hình đã huấn luyện
CONFIG_PATH = Path("config/parking_config.json")
MODEL_CNN_PATH = Path("models/parking_cnn.keras")
VIDEO_PATH = "carPark.mp4"

# Tạo thư mục chạy riêng biệt theo timestamp (Giai đoạn 11)
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
RUN_DIR = Path(f"reports/runs/{run_timestamp}")
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = RUN_DIR / "result.mp4"
SUMMARY_JSON_PATH = RUN_DIR / "summary.json"
FRAME_CSV_PATH = RUN_DIR / "frame_statistics.csv"
SLOT_CSV_PATH = RUN_DIR / "slot_predictions.csv"
CONFIG_COPY_PATH = RUN_DIR / "run_config.json"

class SlotSmoother:
    """Lớp làm mượt trạng thái qua các frame sử dụng thuật toán Majority Vote (Stage 9)"""
    def __init__(self, window_size=5):
        self.history = {}
        self.window_size = window_size

    def update(self, slot_id, label):
        if slot_id not in self.history:
            self.history[slot_id] = deque(maxlen=self.window_size)
        self.history[slot_id].append(label)
        
        counts = Counter(self.history[slot_id])
        return counts.most_common(1)[0][0]

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

class ParkingPredictor:
    def __init__(self, model_path):
        if str(model_path).endswith(".keras"):
            self.model = tf.keras.models.load_model(model_path)
            self.is_cnn = True
        else:
            self.model = joblib.load(model_path)
            self.is_cnn = False

    def predict_slots(self, frame, boxes):
        """Dự đoán trạng thái từng slot kết hợp kiểm tra ngưỡng Confidence (Stage 10) kèm giá trị confidence"""
        predictions = {}
        confidences = {}
        for box in boxes:
            box_id = box["id"]
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            
            slot_crop = frame[y:y+h, x:x+w]
            if slot_crop.size == 0:
                predictions[box_id] = "invalid"
                confidences[box_id] = 0.0
                continue
                
            try:
                resized = cv2.resize(slot_crop, (64, 64))
                if self.is_cnn:
                    input_data = np.expand_dims(resized / 255.0, axis=0)
                    pred_prob = self.model.predict(input_data, verbose=0)[0][0]
                    
                    confidence = float(max(pred_prob, 1 - pred_prob))
                    confidences[box_id] = confidence
                    
                    if confidence < 0.60:
                        status = "unknown"
                    else:
                        status = "occupied" if pred_prob > 0.5 else "empty"
                else:
                    status = "empty"
                    confidences[box_id] = 1.0
                
                predictions[box_id] = status
            except Exception:
                predictions[box_id] = "error"
                confidences[box_id] = 0.0
                
        return predictions, confidences

def calculate_statistics(predictions):
    empty_count = 0
    occupied_count = 0
    unknown_count = 0
    invalid_count = 0
    
    for box_id, status in predictions.items():
        if status == "empty":
            empty_count += 1
        elif status == "occupied":
            occupied_count += 1
        elif status == "unknown":
            unknown_count += 1
        else:
            invalid_count += 1
            
    valid_count = empty_count + occupied_count
    occupancy_rate = (
        (occupied_count / valid_count)
        if valid_count > 0
        else 0.0
    )
    
    return {
        "empty_count": empty_count,
        "occupied_count": occupied_count,
        "unknown_count": unknown_count,
        "invalid_count": invalid_count,
        "occupancy_rate": occupancy_rate
    }

def draw_renderer(frame, boxes, predictions, stats):
    output_frame = frame.copy()
    
    for box in boxes:
        box_id = box["id"]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        status = predictions.get(box_id, "unknown")
        
        if status == "empty":
            color = (0, 255, 0)
        elif status == "occupied":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
            
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_frame, f"ID:{box_id}", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(output_frame, f"Empty: {stats['empty_count']}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Occupied: {stats['occupied_count']}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(output_frame, f"Unknown: {stats['unknown_count']}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output_frame, f"Occupancy Rate: {stats['occupancy_rate']*100:.1f}%", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return output_frame

def main():
    config = load_config()
    boxes = config.get("boxes", [])
    total_slots = len(boxes)
    camera_id = config.get("camera_id", "camera_01")
    
    # Sao lưu config chạy vào thư mục kết quả
    with open(CONFIG_COPY_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print(f"[Lỗi] Không thể mở video: {VIDEO_PATH}")
        return
        
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))
    
    detector = ParkingPredictor(MODEL_CNN_PATH)
    smoother = SlotSmoother(window_size=5)
    
    print(f"=== BẮT ĐẦU CHẠY INFERENCE VÀ LƯU BÁO CÁO TẠI: {RUN_DIR} ===")
    
    frame_index = 0
    all_frame_stats = []
    
    # Khởi tạo ghi file CSV cho slot predictions
    with open(SLOT_CSV_PATH, mode="w", newline="", encoding="utf-8") as slot_file:
        slot_writer = csv.writer(slot_file)
        slot_writer.writerow(["frame", "slot_id", "label", "confidence"])
        
        with open(FRAME_CSV_PATH, mode="w", newline="", encoding="utf-8") as frame_file:
            frame_writer = csv.writer(frame_file)
            frame_writer.writerow(["frame", "timestamp", "total", "empty", "occupied", "unknown", "invalid", "occupancy_rate"])
            
            while True:
                success, frame = capture.read()
                if not success:
                    break
                    
                timestamp = round(frame_index / fps, 2)
                
                # 1. Dự đoán thô & confidence
                raw_predictions, confidences = detector.predict_slots(frame, boxes)
                
                # 2. Temporal Smoothing
                smoothed_predictions = {}
                for box_id, raw_status in raw_predictions.items():
                    smoothed_status = smoother.update(box_id, raw_status)
                    smoothed_predictions[box_id] = smoothed_status
                    
                    # Ghi nhận vào slot_predictions.csv
                    slot_writer.writerow([frame_index + 1, box_id, smoothed_status, round(confidences.get(box_id, 0.0), 2)])
                
                # 3. Tính toán thống kê
                stats = calculate_statistics(smoothed_predictions)
                
                # Ghi nhận vào frame_statistics.csv
                frame_writer.writerow([
                    frame_index + 1, timestamp, total_slots,
                    stats['empty_count'], stats['occupied_count'],
                    stats['unknown_count'], stats['invalid_count'],
                    round(stats['occupancy_rate'], 4)
                ])
                
                all_frame_stats.append(stats)
                
                # 4. Render và ghi video
                output_frame = draw_renderer(frame, boxes, smoothed_predictions, stats)
                writer.write(output_frame)
                
                frame_index += 1
                if frame_index % 50 == 0:
                    print(f"Đã xử lý khung hình thứ {frame_index}...")

    capture.release()
    writer.release()

    # Tính toán các chỉ số trung bình cho summary.json
    processed_frames = frame_index
    if processed_frames > 0:
        avg_empty = sum(s['empty_count'] for s in all_frame_stats) / processed_frames
        avg_occupied = sum(s['occupied_count'] for s in all_frame_stats) / processed_frames
        avg_occupancy_rate = sum(s['occupancy_rate'] for s in all_frame_stats) / processed_frames
        total_unknown = sum(s['unknown_count'] for s in all_frame_stats)
        total_invalid = sum(s['invalid_count'] for s in all_frame_stats)
    else:
        avg_empty = avg_occupied = avg_occupancy_rate = total_unknown = total_invalid = 0.0

    # Tạo file summary.json chuẩn Stage 11
    summary_data = {
        "input": VIDEO_PATH,
        "model": str(MODEL_CNN_PATH),
        "camera_id": camera_id,
        "total_slots": total_slots,
        "processed_frames": processed_frames,
        "empty_average": round(avg_empty, 2),
        "occupied_average": round(avg_occupied, 2),
        "average_occupancy_rate": round(avg_occupancy_rate, 2),
        "unknown_count": total_unknown,
        "invalid_count": total_invalid
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"[Thành công] Đã hoàn tất! Toàn bộ báo cáo được lưu tại: {RUN_DIR}")

if __name__ == "__main__":
    main()