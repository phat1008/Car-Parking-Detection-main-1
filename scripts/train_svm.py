import os
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    balanced_accuracy_score,
    ConfusionMatrixDisplay
)
from skimage.feature import hog
import matplotlib.pyplot as plt
import joblib

# --- Cấu hình thông số ---
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
IMG_SIZE = (64, 64)
CLASSES = ["empty", "not_empty"]

# Thư mục lưu metric theo yêu cầu
METRICS_DIR = Path("reports/metrics/svm")
MODELS_DIR = Path("models")

def extract_features(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=False
    )
    return features

def load_dataset(split_dir: Path):
    X, y = [], []
    class_counts = {cls: 0 for cls in CLASSES}
    
    for label_idx, cls_name in enumerate(CLASSES):
        cls_folder = split_dir / cls_name
        if not cls_folder.exists():
            continue
            
        for img_path in cls_folder.glob("*.*"):
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)
                class_counts[cls_name] += 1
                
    return np.array(X), np.array(y), class_counts

def main():
    print("=== Đang tải dữ liệu và trích xuất đặc trưng HOG ===")
    X_train, y_train, train_counts = load_dataset(TRAIN_DIR)
    X_val, y_val, val_counts = load_dataset(VAL_DIR)
    
    print(f"Số mẫu Train: {len(X_train)} | Số mẫu Val: {len(X_val)}")
    print(f"Thống kê ảnh Train: {train_counts}")
    print(f"Thống kê ảnh Val (support): {val_counts}")

    # Huấn luyện mô hình SVM Pipeline
    print("=== Đang huấn luyện mô hình SVM ===")
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='scale')
    )
    model.fit(X_train, y_train)
    
    # Dự đoán trên tập Validation
    y_pred = model.predict(X_val)
    
    # --- Tính toán các metric yêu cầu ---
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
    
    cm = confusion_matrix(y_val, y_pred)
    report_text = classification_report(y_val, y_pred, target_names=CLASSES)

    print(f"\nAccuracy: {acc:.4f} | Balanced Accuracy: {bal_acc:.4f} | Macro F1: {macro_f1:.4f}")

    # --- Lưu trữ các file metric và cấu hình ---
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Lưu classification_report.txt
    report_path = METRICS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 2. Lưu confusion_matrix.png
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d", colorbar=False)
    plt.title("SVM Confusion Matrix (Validation)")
    plt.savefig(METRICS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Lưu metrics.json
    metrics_data = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "per_class": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            } for i in range(len(CLASSES))
        },
        "confusion_matrix": cm.tolist(),
        "train_image_counts": train_counts,
        "val_image_counts": val_counts
    }
    with open(METRICS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    # 4. Lưu config.json
    config_data = {
        "model_name": "SVM_RBF_HOG",
        "img_size": list(IMG_SIZE),
        "C": 10.0,
        "gamma": "scale",
        "kernel": "rbf",
        "feature_extractor": "HOG (orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))"
    }
    with open(METRICS_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    # Lưu trọng số mô hình
    model_path = MODELS_DIR / "parking_svm_optimized.joblib"
    joblib.dump(model, model_path)

    print(f"\n[Thành công] Đã lưu đầy đủ metric, báo cáo, biểu đồ và config tại thư mục: {METRICS_DIR}")

if __name__ == "__main__":
    main()