import pandas as pd
from pathlib import Path
import json

def evaluate_video():
    # Đường dẫn file Ground Truth (do bạn tạo thủ công các frame đại diện) và file Prediction hệ thống xuất ra
    gt_path = Path("reports/ground_truth.csv")
    pred_path = Path("reports/runs/latest_run/slot_predictions.csv") # Hoặc trỏ tới thư mục chạy gần nhất của bạn
    
    # Tìm tự động file slot_predictions.csv mới nhất trong reports/runs/ nếu có
    runs_dir = Path("reports/runs")
    if runs_dir.exists():
        subdirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        if subdirs:
            pred_path = subdirs[0] / "slot_predictions.csv"

    if not gt_path.exists():
        print(f"[Cảnh báo] Chưa tìm thấy file Ground Truth tại: {gt_path}")
        print("Vui lòng tạo file ground_truth.csv với các cột: frame, slot_id, label trước khi chạy đánh giá!")
        return

    if not pred_path.exists():
        print(f"[Lỗi] Không tìm thấy file dự đoán tại: {pred_path}. Hãy chạy run_inference.py trước.")
        return

    df_gt = pd.read_csv(gt_path)
    df_pred = pd.read_csv(pred_path)

    # Gộp 2 DataFrame dựa trên frame và slot_id
    merged = pd.merge(df_gt, df_pred, on=["frame", "slot_id"], suffixes=("_gt", "_pred"))

    if merged.empty:
        print("[Lỗi] Không khớp frame hoặc slot_id nào giữa Ground Truth và Prediction!")
        return

    # Chuẩn hóa nhãn (đảm bảo 'not_empty' tương đương 'occupied')
    merged['label_gt'] = merged['label_gt'].replace({'not_empty': 'occupied'})
    merged['label_pred'] = merged['label_pred'].replace({'not_empty': 'occupied'})

    # Tính toán các metric
    total_samples = len(merged)
    correct_predictions = (merged['label_gt'] == merged['label_pred']).sum()
    slot_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    # False Empty: Thực tế là occupied nhưng dự đoán là empty
    false_empty = ((merged['label_gt'] == 'occupied') & (merged['label_pred'] == 'empty')).sum()
    total_occupied_gt = (merged['label_gt'] == 'occupied').sum()
    false_empty_rate = false_empty / total_occupied_gt if total_occupied_gt > 0 else 0.0

    # False Occupied: Thực tế là empty nhưng dự đoán là occupied
    false_occupied = ((merged['label_gt'] == 'empty') & (merged['label_pred'] == 'occupied')).sum()
    total_empty_gt = (merged['label_gt'] == 'empty').sum()
    false_occupied_rate = false_occupied / total_empty_gt if total_empty_gt > 0 else 0.0

    metrics_result = {
        "total_evaluated_samples": int(total_samples),
        "slot_accuracy": round(float(slot_accuracy), 4),
        "false_empty_rate": round(float(false_empty_rate), 4),
        "false_occupied_rate": round(float(false_occupied_rate), 4)
    }

    print("=== KẾT QUẢ ĐÁNH GIÁ VIDEO (VIDEO METRICS) ===")
    for k, v in metrics_result.items():
        print(f" - {k}: {v}")

    output_metrics_path = Path("reports/metrics/video_evaluation_metrics.json")
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=4)
    print(f"\n[Thành công] Đã lưu báo cáo đánh giá video tại: {output_metrics_path}")

if __name__ == "__main__":
    evaluate_video()