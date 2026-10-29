import logging
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog

# Cấu hình logging để ghi log khi model không tồn tại hoặc lỗi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParkingClassifier:
    def __init__(self, model_path, confidence_threshold=0.6):
        """
        Khởi tạo bộ phân loại ô đỗ xe.
        :param model_path: Đường dẫn tới file model joblib (.joblib)
        :param confidence_threshold: Ngưỡng confidence tối thiểu (dưới mức này trả về 'unknown')
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()

    def _load_model(self):
        """Load a compatible HOG model or fail fast with an actionable error."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file model tại đường dẫn: {self.model_path}")

        try:
            model = joblib.load(self.model_path)
        except (OSError, EOFError, ImportError, AttributeError, ValueError) as exc:
            raise RuntimeError(f"Không thể load model {self.model_path}: {exc}") from exc

        if not hasattr(model, "predict"):
            raise TypeError(f"Model {self.model_path} không hợp lệ: thiếu phương thức predict")

        logger.info(f"[INFO] Tải thành công model từ {self.model_path}")
        return model

    def preprocess(self, crop):
        """
        Resize ảnh về kích thước của pipeline HOG rồi trích xuất đặc trưng
        grayscale giống scripts/train_svm.py.
        """
        if crop is None or crop.size == 0:
            raise ValueError("Empty parking-space crop")

        resized = cv2.resize(
            crop,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
        )
        return features.reshape(1, -1)

    def predict(self, crop):
        """
        Thực hiện phân loại ô crop với các tiêu chí chuẩn hóa nhãn:
        - 'empty', 'occupied', 'unknown'
        - Xử lý crop rỗng trả về 'unknown' thay vì crash hoặc coi là occupied.
        """
        # Crop rỗng là lỗi hình học của slot, không phải dự đoán không chắc chắn.
        if crop is None or crop.size == 0:
            return {
                "label": "invalid",
                "state": "invalid",
                "is_empty": False,
                "confidence": 0.0,
            }

        try:
            features = self.preprocess(crop)

            raw_label = self.model.predict(features)[0]
            confidence = None

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 1.0  # Mặc định nếu model không hỗ trợ predict_proba

            # 2 & 3. Chuẩn hóa nhãn: Chuyển 'not_empty' (hoặc các biến thể tương đương) thành 'occupied'
            raw_str = str(raw_label).lower()
            if raw_str in ["empty", "0"]:
                normalized_label = "empty"
                is_empty = True
            elif raw_str in ["not_empty", "occupied", "car", "1"]:
                normalized_label = "occupied"
                is_empty = False
            else:
                normalized_label = "unknown"
                is_empty = False

            # 4. Nếu confidence thấp hơn ngưỡng, trả về 'unknown'
            if confidence is not None and confidence < self.confidence_threshold:
                normalized_label = "unknown"
                is_empty = False

            return {
                "label": normalized_label,
                "state": normalized_label,  # Hỗ trợ thêm trường 'state' theo chuẩn kết quả mong muốn
                "is_empty": is_empty,
                "confidence": round(confidence, 4) if confidence is not None else None,
            }

        except (ValueError, TypeError, IndexError, RuntimeError) as exc:
            raise RuntimeError(
                f"Lỗi trong quá trình predict crop bằng model {self.model_path}: {exc}"
            ) from exc
