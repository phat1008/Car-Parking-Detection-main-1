import os
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from skimage.feature import hog
import joblib

# Đường dẫn dữ liệu
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
IMG_SIZE = (64, 64)
CLASSES = ["empty", "not_empty"]

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
    for label_idx, cls_name in enumerate(CLASSES):
        cls_folder = split_dir / cls_name
        if not cls_folder.exists():
            continue
        for img_path in cls_folder.glob("*.*"):
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)
    return np.array(X), np.array(y)

def main():
    print("=== Đang tải tập Train và Val để chạy Grid Search ===")
    X_train, y_train = load_dataset(TRAIN_DIR)
    X_val, y_val = load_dataset(VAL_DIR)
    
    print(f"Số mẫu Train: {len(X_train)} | Số mẫu Val: {len(X_val)}")

    # Xây dựng pipeline gồm StandardScaler và SVC
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(probability=True)
    )

    # Thiết lập không gian tham số Grid Search theo tài liệu yêu cầu
    param_grid = {
        "svc__C": [0.1, 1, 10, 50, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        "svc__kernel": ["rbf", "linear"]
    }

    # Sử dụng validation set để chọn model (dùng GridSearchCV)[cite: 20]
    print("=== Đang tiến hành Grid Search tối ưu hóa SVM ===")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='recall_macro', # Hoặc f1 để tối ưu cân bằng các class
        cv=3,                   # Cross-validation trên tập train (tránh data leakage)
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n[Kết quả] Tham số tốt nhất tìm được: {grid_search.best_params_}")
    
    # Đánh giá mô hình tốt nhất trên tập Validation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None)

    print(f"\nĐộ chính xác trên Validation (Accuracy): {acc * 100:.2f}%")
    print(f"Recall của class 'not_empty': {recall[1] * 100:.2f}% (Mục tiêu >= 95%)")
    print(f"Precision của class 'empty': {precision[0] * 100:.2f}%")
    
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

   # Lưu mô hình tối ưu và metadata theo đúng chuẩn yêu cầu
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/parking_svm.joblib"
    metadata_path = "models/parking_svm.metadata.json"
    
    # Lưu file joblib
    joblib.dump(best_model, model_path)
    
    # Tạo nội dung metadata chuẩn theo yêu cầu dự án
    metadata = {
        "model_type": "svm",
        "image_size": [64, 64],
        "color_format": "BGR",
        "normalization": "StandardScaler",
        "feature_type": "hog",
        "classes": CLASSES,
        "kernel": grid_search.best_params_["svc__kernel"],
        "C": grid_search.best_params_["svc__C"],
        "gamma": grid_search.best_params_["svc__gamma"],
        "python_version": "3.11",
        "opencv_version": cv2.__version__,
        "sklearn_version": "1.3.0"
    }
    
    # Lưu file JSON metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n[Thành công] Đã lưu mô hình tại: {model_path}")
    print(f"[Thành công] Đã lưu metadata tại: {metadata_path}")

if __name__ == "__main__":
    main()