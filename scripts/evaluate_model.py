import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình bãi đỗ xe")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn tới file mô hình (.keras hoặc .joblib)")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn tới thư mục test hoặc file CSV chứa dữ liệu test")
    parser.add_argument("--output", type=str, default="reports/metrics/eval_result", help="Thư mục lưu kết quả đánh giá")
    return parser.parse_args()

def load_model(model_path):
    path = Path(model_path)
    if str(path).endswith(".keras"):
        return tf.keras.models.load_model(path), "keras"
    else:
        return joblib.load(path), "joblib"

def evaluate():
    args = parse_args()
    model, model_type = load_model(args.model)
    output_dir = Path(args.output)
    
    # Tạo cấu trúc thư mục báo cáo theo yêu cầu Stage 12
    output_dir.mkdir(parents=True, exist_ok=True)
    misclassified_dir = output_dir / "misclassified"
    (misclassified_dir / "false_empty").mkdir(parents=True, exist_ok=True)
    (misclassified_dir / "false_occupied").mkdir(parents=True, exist_ok=True)
    (misclassified_dir / "low_confidence").mkdir(parents=True, exist_ok=True)

    # Giả định dữ liệu test được tổ chức theo thư mục: data/test/empty và data/test/occupied
    test_dir = Path(args.data)
    if not test_dir.exists():
        print(f"[Lỗi] Không tìm thấy thư mục/file dữ liệu test tại: {test_dir}")
        return

    images = []
    labels = []
    image_paths = []
    
    for category in ["empty", "occupied"]:
        cat_dir = test_dir / category
        if cat_dir.exists():
            for img_path in cat_dir.glob("*.*"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))
                    images.append(img_resized)
                    labels.append(0 if category == "empty" else 1)
                    image_paths.append(img_path)

    if not images:
        print("[Lỗi] Không tìm thấy ảnh nào trong thư mục test!")
        return

    X = np.array(images) / 255.0
    y_true = np.array(labels)

    # Dự đoán
    if model_type == "keras":
        y_probs = model.predict(X, verbose=0).flatten()
        y_pred = (y_probs > 0.5).astype(int)
    else:
        # Dành cho SVM (nếu có trích xuất đặc trưng phù hợp)
        # Flatten ảnh cho SVM cơ bản
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        y_probs = model.predict_proba(X_flat)[:, 1]
        y_pred = model.predict(X_flat)

    # Tính toán các chỉ số
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    report = classification_report(y_true, y_pred, target_names=["empty", "occupied"])
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Đánh giá hoàn tất! Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    # 1. Lưu classification_report.txt
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 2. Lưu metrics.json
    metrics_data = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    # 3. Lưu Confusion Matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["empty", "occupied"], yticklabels=["empty", "occupied"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    # 4. Lưu ROC Curve & Precision-Recall Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(output_dir / "precision_recall_curve.png")
    plt.close()

    # 5. Lưu wrong_predictions.csv và gom ảnh dự đoán sai / confidence thấp
    wrong_records = []
    for idx, (path, true_lbl, pred_lbl, prob) in enumerate(zip(image_paths, y_true, y_pred, y_probs)):
        confidence = float(max(prob, 1 - prob))
        img_original = cv2.imread(str(path))
        
        # Ghi nhận các trường hợp dự đoán sai hoặc độ tin cậy thấp (< 0.6)
        if true_lbl != pred_lbl or confidence < 0.60:
            wrong_records.append({
                "image_path": str(path),
                "true_label": "occupied" if true_lbl == 1 else "empty",
                "predicted_label": "occupied" if pred_lbl == 1 else "empty",
                "confidence": round(confidence, 4)
            })
            
            # Phân loại lưu vào thư mục misclassified tương ứng
            if confidence < 0.60:
                save_path = misclassified_dir / "low_confidence" / path.name
            elif true_lbl == 0 and pred_lbl == 1:
                save_path = misclassified_dir / "false_occupied" / path.name
            else:
                save_path = misclassified_dir / "false_empty" / path.name
                
            if img_original is not None:
                cv2.imwrite(str(save_path), img_original)

    df_wrong = pd.DataFrame(wrong_records)
    df_wrong.to_csv(output_dir / "wrong_predictions.csv", index=False)
    
    print(f"[Thành công] Toàn bộ báo cáo và hình ảnh phân tích lỗi đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    evaluate()