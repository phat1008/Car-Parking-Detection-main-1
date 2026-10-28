import cv2
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import csv
from src.slot_manager import SlotManager
from src.smoother import SlotSmoother
from skimage.feature import hog

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

class ParkingPredictor:
    def __init__(self, model_path):
        if str(model_path).endswith(".keras"):
            self.model = tf.keras.models.load_model(model_path)
            self.is_cnn = True
        else:
            self.model = joblib.load(model_path)
            self.is_cnn = False

    @staticmethod
    def preprocess_cnn_crop(crop):
        resized = cv2.resize(crop, (64, 64))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # The trained CNN contains its own Rescaling(1/255) layer.
        return resized.astype(np.float32)

    def predict_slots(self, frame, slot_manager):
        """Dự đoán trạng thái từng slot kết hợp kiểm tra ngưỡng Confidence (Stage 10) kèm giá trị confidence"""
        predictions = {}
        confidences = {}
        cnn_crops = []
        cnn_box_ids = []

        for item in slot_manager.crop_slots(frame):
            box_id = item["id"]
            slot_crop = item["crop"]
            if slot_crop.size == 0:
                predictions[box_id] = "invalid"
                confidences[box_id] = 0.0
                continue

            if self.is_cnn:
                cnn_crops.append(self.preprocess_cnn_crop(slot_crop))
                cnn_box_ids.append(box_id)
                continue

            try:
                resized = cv2.resize(slot_crop, (64, 64))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                features = hog(
                    gray,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                    visualize=False,
                ).reshape(1, -1)
                raw_label = self.model.predict(features)[0]
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(features)[0]
                    pred_prob = float(probabilities[1])
                else:
                    decision = float(self.model.decision_function(features)[0])
                    pred_prob = 1.0 / (1.0 + np.exp(-decision))

                confidence = max(pred_prob, 1.0 - pred_prob)
                confidences[box_id] = confidence
                if confidence < 0.60:
                    status = "unknown"
                else:
                    status = (
                        "occupied"
                        if str(raw_label).lower() in {"1", "not_empty", "occupied", "car"}
                        else "empty"
                    )
                predictions[box_id] = status
            except Exception:
                predictions[box_id] = "error"
                confidences[box_id] = 0.0

        if cnn_crops:
            batch = np.stack(cnn_crops, axis=0)
            probabilities = self.model.predict(batch, verbose=0).reshape(-1)
            for box_id, probability in zip(cnn_box_ids, probabilities):
                probability = float(probability)
                confidence = max(probability, 1.0 - probability)
                confidences[box_id] = confidence
                if confidence < 0.60:
                    predictions[box_id] = "unknown"
                elif probability > 0.5:
                    predictions[box_id] = "occupied"
                else:
                    predictions[box_id] = "empty"

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

def draw_renderer(frame, slot_manager, predictions, stats):
    output_frame = frame.copy()
    
    for item in slot_manager.crop_slots(frame):
        box_id = item["id"]
        x, y, w, h = item["box"]
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
    slot_manager = SlotManager(CONFIG_PATH)
    config = slot_manager.config
    total_slots = len(slot_manager.slots)
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
    if not fps or fps <= 0:
        fps = 25.0
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))
    
    detector = ParkingPredictor(MODEL_CNN_PATH)
    smoother = SlotSmoother(window_size=7, min_stable_count=5)
    
    print(f"=== BẮT ĐẦU CHẠY INFERENCE VÀ LƯU BÁO CÁO TẠI: {RUN_DIR} ===")
    
    frame_index = 0
    all_frame_stats = []
    previous_predictions = {}
    
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
                raw_predictions, confidences = detector.predict_slots(frame, slot_manager)
                
                # 2. Temporal Smoothing
                smoothed_predictions = {}
                for box_id, raw_status in raw_predictions.items():
                    smoothed_status = smoother.update(
                        box_id,
                        raw_status,
                        confidence=confidences.get(box_id),
                    )
                    smoothed_predictions[box_id] = smoothed_status

                    if previous_predictions.get(box_id) != smoothed_status:
                        slot_writer.writerow(
                            [
                                frame_index + 1,
                                box_id,
                                smoothed_status,
                                round(confidences.get(box_id, 0.0), 2),
                            ]
                        )

                    previous_predictions[box_id] = smoothed_status
                
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
                output_frame = draw_renderer(frame, slot_manager, smoothed_predictions, stats)
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