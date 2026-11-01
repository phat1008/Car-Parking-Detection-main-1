# Deploy dashboard: Docker Hub + MongoDB Atlas + Render

## 1. Chạy local (kiểm tra trước khi build image)

```bash
python -m pip install -r web/requirements-web.txt
python -c "from web.app import create_app; create_app().run(host='127.0.0.1', port=8000)"
```

Mở http://localhost:8000. Lệnh Flask ở trên chạy được trên Windows, macOS và Linux
để kiểm tra local. Khi chạy production trên Linux, có thể dùng Gunicorn:

```bash
gunicorn -w 1 -b 0.0.0.0:8000 'web.app:create_app()'
```

Không đặt `MONGO_URI` thì kết quả chỉ lưu trong bộ nhớ.

## 2. Build và đẩy image lên Docker Hub

Chạy từ thư mục gốc repo. Build context phải có `models/parking_cnn.keras`;
file model hiện được theo dõi trong repository này. Nếu dùng bản sao/fork không
có model, hãy đặt model tại đường dẫn đó trước khi build image.

```bash
docker build -t <dockerhub_user>/car-parking-dashboard:latest .
docker login
docker push <dockerhub_user>/car-parking-dashboard:latest
```

Kiểm tra image chạy được trước khi push:

```bash
docker run --rm -p 8000:8000 -e PORT=8000 <dockerhub_user>/car-parking-dashboard:latest
```

## 3. MongoDB Atlas

1. Tạo cluster miễn phí, tạo user database và lấy connection string dạng `mongodb+srv://...`.
2. Network Access: thêm `0.0.0.0/0` để Render kết nối được (hoặc dùng IP tĩnh nếu có gói trả phí).

## 4. Render

1. New > Web Service > Deploy an existing image from a registry.
2. Image URL: `docker.io/<dockerhub_user>/car-parking-dashboard:latest`.
3. Environment variables:
   - `MONGO_URI` = connection string Atlas (đánh dấu là Secret).
   - `MONGO_DB` = `parking` (tùy chọn).
   - `CONFIDENCE_THRESHOLD` = `0.60` (tùy chọn).
4. Health Check Path: `/healthz`.
5. Render tự cấp `PORT`, container đã đọc biến này.

Lưu ý: gói miễn phí của Render ngủ sau một thời gian không dùng và không có ổ đĩa bền vững. Dữ liệu lịch sử nằm trên Atlas nên không mất; model nằm trong image.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PORT` | `8000` | Cổng web (Render tự đặt). |
| `MODEL_PATH` | `/app/models/parking_cnn.keras` | Model CNN. |
| `CONFIG_PATH` | `/app/config/parking_config.json` | Layout các ô đỗ. |
| `CONFIDENCE_THRESHOLD` | `0.60` | Ngưỡng để ra trạng thái `unknown`. |
| `MONGO_URI` | (trống) | Kết nối Atlas; trống thì lưu bộ nhớ tạm. |
| `MONGO_DB` | `parking` | Tên database. |
| `MONGO_COLLECTION` | `detections` | Tên collection. |

## API

- `GET /` giao diện dashboard.
- `POST /api/video` form-data field `video` (tối đa 60 MB), tùy chọn `every_sec` (mặc định 0,1 giây) và `max_frames`. Trả về luồng NDJSON gồm từng khung hình đã vẽ trạng thái, thống kê và kết quả cuối; lưu kết quả vào MongoDB nếu cấu hình.
- `GET /api/status` kết quả gần nhất.
- `GET /api/history?limit=50` lịch sử (tối đa 200).
- `GET /api/slots` danh sách ô và trạng thái lưu trữ.
- `GET /healthz` kiểm tra sống.
