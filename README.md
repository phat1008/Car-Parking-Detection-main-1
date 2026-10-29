# Car Parking Detection

Dự án nhận diện trạng thái các ô đỗ xe trong video hoặc webcam. Người dùng định
nghĩa trước vị trí ô đỗ trong một file cấu hình; hệ thống crop từng ô, phân loại
`empty` / `occupied` bằng CNN, làm mượt dự đoán theo thời gian và ghi video cùng
các báo cáo CSV/JSON.

> Đây là dự án thử nghiệm với các vùng đỗ được cấu hình thủ công. Hệ thống không
> tự phát hiện vị trí ô đỗ và chưa phải dịch vụ quản lý nhiều camera hoàn chỉnh.

## Chức năng

- Vẽ và kiểm tra vùng đỗ xe từ ảnh hoặc video.
- Chia ảnh theo lớp `empty` và `not_empty` thành train/validation/test.
- Huấn luyện CNN; có thêm các script thử nghiệm SVM với đặc trưng HOG.
- Chạy nhận diện trên file video hoặc webcam, hiển thị và lưu video kết quả.
- Xuất dự đoán theo từng ô, thống kê theo lần inference và summary của lần chạy.
- Đánh giá trên tập ảnh hoặc đối chiếu dự đoán video với ground truth.

## Yêu cầu

- Python 3.10 trở lên được khuyến nghị. `scripts/run.py` dùng cú pháp yêu cầu
  Python 3.10+.
- Các thư viện được liệt kê trong `requirements.txt`.
- Với preview có giao diện, cần môi trường desktop hỗ trợ cửa sổ OpenCV.

TensorFlow có thể chạy trên CPU. Trên Windows, TensorFlow native thường không
dùng CUDA GPU; nếu cần tăng tốc GPU, cần thiết lập runtime/OS được TensorFlow
hỗ trợ và xác nhận GPU xuất hiện trong danh sách thiết bị TensorFlow.

## Cài đặt

Chạy các lệnh sau từ thư mục gốc của repository.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` cài thêm các công cụ kiểm tra và bao gồm
`requirements.txt`.

## Chạy nhận diện

Runner chính là `scripts/run.py`.

```powershell
# Dùng video mặc định carPark.mp4
python scripts/run.py

# Chọn video khác
python scripts/run.py --source carPark1.mp4

# Dùng webcam mặc định (index 0)
python scripts/run.py --source 0

# Chỉ định cấu hình, model, ngưỡng và chạy không mở cửa sổ
python scripts/run.py `
  --source carPark.mp4 `
  --config config/parking_config.json `
  --model models/parking_cnn.keras `
  --confidence 0.60 `
  --inference-interval 1 `
  --no-display
```

Các tham số:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `source` hoặc `--source` | `carPark.mp4` | Đường dẫn video hoặc chỉ số webcam. |
| `--config` | `config/parking_config.json` | Layout các ô đỗ. |
| `--model` | `models/parking_cnn.keras` | Model CNN đã huấn luyện. |
| `--confidence` | `0.60` | Ngưỡng để phân loại dự đoán thành `empty`/`occupied`; thấp hơn ngưỡng sẽ là `unknown`. |
| `--inference-interval` | `2` | Chạy CNN mỗi N frame; frame đầu tiên luôn được inference. `1` nghĩa là inference trên mọi frame. |
| `--no-display` | Tắt | Không mở cửa sổ preview. Video và báo cáo vẫn được ghi. |

Mỗi lần chạy tạo thư mục riêng dưới `reports/runs/<thời_gian>/`:

- `result.mp4`: video có vẽ trạng thái ô đỗ.
- `slot_predictions.csv`: một dòng cho mỗi ô tại mỗi frame được inference.
- `frame_statistics.csv`: số lượng trạng thái và tỷ lệ khả dụng tại mỗi frame được inference.
- `summary.json`: thông tin nguồn, model, số frame, số lần inference và thống kê cuối.

`slot_predictions.csv` có các cột `frame`, `slot_id`, `raw_state`,
`stable_state`, `confidence`, `probability_occupied`. `raw_state` là dự đoán
trực tiếp; `stable_state` là kết quả sau làm mượt. `confidence` là độ tin cậy
của lớp được chọn; `probability_occupied` là xác suất model dự đoán có xe.

`frame_statistics.csv` có các cột `frame`, `timestamp`, `configured`, `empty`,
`occupied`, `unknown`, `invalid`, `availability`. `availability` được tính bằng
`empty / (empty + occupied)`, không tính các ô `unknown` và `invalid`.

Runner ghi CSV tại các lần inference, không phải mọi frame nếu
`--inference-interval` lớn hơn 1. Video output vẫn chứa mọi frame; ở những frame
không inference, trạng thái hiển thị được giữ từ lần inference gần nhất.

## Cấu hình vị trí ô đỗ

File mặc định là `config/parking_config.json`. Các trường chính:

- `source_width`, `source_height`: kích thước ảnh/video dùng để vẽ layout.
- `camera_id`: định danh camera.
- `boxes`: danh sách vùng đỗ; mỗi vùng có `id`, `x`, `y`, `width`, `height`.
- `enabled`: tùy chọn bật/tắt vùng; nếu bỏ qua thì mặc định bật.
- `type`: metadata loại vùng, ví dụ `regular`.

Tọa độ phải nằm trong kích thước nguồn đã khai báo. Khi video có độ phân giải
khác, hệ thống scale box theo kích thước frame hiện tại.

Tạo/cập nhật cấu hình tương tác:

```powershell
python scripts/annotate_slots.py --source carPark.mp4 --output config/parking_config.json
```

Kiểm tra file cấu hình:

```powershell
python scripts/validate_config.py
```

`annotate_slots.py` tự kiểm tra kích thước nguồn, ID trùng, tọa độ/kích thước
box, giới hạn ảnh và yêu cầu có ít nhất một slot được bật trước khi lưu. Ảnh
preview được thu nhỏ để vừa cửa sổ; tọa độ chuột được quy đổi về kích thước ảnh
gốc, nên file config vẫn lưu tọa độ theo frame nguồn. Dùng `z` để undo, `r` để
xóa toàn bộ slot, `s` để lưu và `q` hoặc `Esc` để thoát.

## Dữ liệu và huấn luyện

Cấu trúc dữ liệu dùng bởi script huấn luyện:

```text
data/
  train/
    empty/
    not_empty/
  val/
    empty/
    not_empty/
  test/
    empty/
    not_empty/
```

Nguồn dữ liệu để chia tập nằm trong `data/clf-data/empty/` và
`data/clf-data/not_empty/`.

```powershell
# Kiểm tra số lượng mẫu và ảnh lỗi
python scripts/inspect_dataset.py
python scripts/check_images.py

# Chia dữ liệu theo nhóm tên file
python scripts/split_dataset.py

# Huấn luyện CNN
python scripts/train_cnn.py
```

`split_dataset.py` nhóm ảnh theo phần tên trước dấu `_` để tránh đưa các ảnh cùng
nhóm/video vào nhiều tập. Script sao chép vào các thư mục split hiện có và không
xóa ảnh cũ không còn thuộc split; cần lưu ý điều này khi thay đổi nguồn dữ liệu
và chia lại.

Model CNN được lưu tại `models/parking_cnn.keras`, metadata tại
`models/parking_cnn.metadata.json`, lịch sử và test metrics tại
`reports/metrics/cnn_training_history.json`.

### Các script SVM thử nghiệm

```powershell
python scripts/train_svm.py
python scripts/grid_search_svm.py
```

Các script này dùng HOG + SVM và lưu model/report riêng. Runner chính hiện được
cấu hình để dùng CNN; SVM không phải model mặc định của runner.

## Đánh giá

Đánh giá model trên thư mục ảnh test:

```powershell
python scripts/evaluate_model.py `
  --model models/parking_cnn.keras `
  --data data/test `
  --output reports/metrics/eval_result
```

Kết quả gồm metrics, classification report, confusion matrix, ROC/PR curves,
`wrong_predictions.csv` và ảnh phân loại sai hoặc confidence thấp.

Đánh giá theo video yêu cầu file `reports/ground_truth.csv` có các cột:

```csv
frame,slot_id,label
1,0,empty
1,1,occupied
```

`frame` phải là số frame 1-based khớp với CSV runner; chỉ các frame được inference
mới có prediction. `label` chấp nhận `empty`, `occupied` hoặc `not_empty`.

```powershell
# Mặc định dùng prediction CSV mới nhất trong reports/runs/
python scripts/evaluate_video.py

# Hoặc chỉ rõ run cần đánh giá
python scripts/evaluate_video.py `
  --ground-truth reports/ground_truth.csv `
  --predictions reports/runs/<run_timestamp>/slot_predictions.csv `
  --output reports/metrics/video_evaluation_metrics.json
```

Evaluator dùng `stable_state` của runner chính để so sánh. Với CSV từ runner cũ
có cột `label`, evaluator vẫn hỗ trợ tương thích.

## Chạy test

```powershell
python -m pytest -q
```

Test dữ liệu kiểm tra thư mục train/validation/test, khả năng đọc ảnh và rò rỉ
nhóm file; do đó cần có dữ liệu trong `data/`. Các test model/detector sử dụng
model giả để chạy nhanh, không cần tải model CNN thật.

## Cấu trúc mã nguồn

```text
parking_detector.py       Kết nối quản lý slot với CNN classifier
classifier.py             Classifier HOG/SVM cũ
src/
  cnn_classifier.py       Tiền xử lý crop và phân loại CNN theo batch
  slot_manager.py         Đọc config, scale box, crop slot từ frame
  parking_counter.py      Đếm trạng thái slot
  smoother.py             Làm mượt trạng thái theo thời gian
scripts/
  run.py                  Runner inference chính
  run_inference.py        Entry point tương thích, chuyển tiếp sang run.py
  train_cnn.py            Huấn luyện CNN
  train_svm.py            Huấn luyện SVM/HOG
  evaluate_model.py       Đánh giá trên ảnh
  evaluate_video.py       Đánh giá theo frame/slot với ground truth
tests/                    Unit và kiểm tra dữ liệu
```
