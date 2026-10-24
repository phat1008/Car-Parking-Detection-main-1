import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Cấu hình đường dẫn và thông số
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30

def create_small_cnn():
    """Xây dựng kiến trúc CNN nhỏ gọn theo đúng tiêu chuẩn tài liệu"""
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        
        # Classifier Head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Nhị phân: empty (0) / not_empty (1)
    ])
    return model

def main():
    print("=== BẮT ĐẦU HUẤN LUYỆN CNN ===")
    
    # 1. Tải và chuẩn bị dữ liệu từ thư mục train/val bằng tf.keras
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    # Chuẩn hóa pixel về khoảng [0, 1]
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Tối ưu hiệu suất đọc dữ liệu
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Khởi tạo mô hình CNN
    model = create_small_cnn()
    
    # 3. Compile mô hình với loss và các metrics yêu cầu
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    model.summary()

    # 4. Cấu hình Early Stopping và lưu model tốt nhất
    os.makedirs("models", exist_ok=True)
    model_path = "models/parking_cnn.keras"
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Tiến hành huấn luyện
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"\n[Thành công] Đã huấn luyện xong và lưu mô hình tại: {model_path}")

    # 6. Lưu metadata kèm theo mô hình CNN
    metadata_path = "models/parking_cnn.metadata.json"
    metadata = {
        "model_type": "cnn",
        "image_size": [64, 64],
        "color_format": "RGB",
        "normalization": "divide_by_255",
        "feature_type": "raw_pixels_cnn",
        "classes": ["empty", "not_empty"],
        "python_version": "3.11",
        "tensorflow_version": tf.__version__
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Thành công] Đã lưu metadata tại: {metadata_path}")

if __name__ == "__main__":
    main()