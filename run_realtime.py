import argparse
import time
from pathlib import Path

import cv2

from parking_detector import ParkingDetector
from src.parking_counter import ParkingCounter
from src.smoother import SlotSmoother


BASE_DIR = Path(__file__).resolve().parent


def parse_source(value):
    return int(value) if value.isdigit() else value


def draw_results(frame, results, counts, availability, fps, show_score):
    output = frame.copy()

    for item in results:
        x, y, width, height = item["box"]
        state = item["state"]
        if state == "empty":
            color, text = (0, 255, 0), "EMPTY"
        elif state == "occupied":
            color, text = (0, 0, 255), "OCCUPIED"
        elif state == "invalid":
            color, text = (128, 128, 128), "INVALID"
        else:
            color, text = (0, 255, 255), "UNKNOWN"

        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        label = f"{item['index']} {text}"
        if show_score and item.get("confidence") is not None:
            label += f" ({item['confidence']:.2f})"
        cv2.putText(
            output,
            label,
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    overlay = output.copy()
    cv2.rectangle(overlay, (15, 15), (300, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        ("PARKING MONITOR", (255, 255, 255), 2),
        (f"Total: {counts['total']}", (255, 255, 255), 1),
        (f"Empty: {counts['empty']}", (0, 255, 0), 1),
        (f"Occupied: {counts['occupied']}", (0, 0, 255), 1),
        (f"Unknown: {counts['unknown']}", (0, 255, 255), 1),
        (f"Availability: {availability:.2f}%", (255, 200, 0), 1),
        (f"FPS: {fps:.1f}", (200, 200, 200), 1),
    ]
    for index, (text, color, thickness) in enumerate(lines):
        cv2.putText(
            output,
            text,
            (25, 45 + index * 25),
            font,
            0.55 if index else 0.6,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Chuong trinh nhan dien bai do xe realtime"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(BASE_DIR / "carPark.mp4"),
        help="Duong dan video hoac so webcam, vi du: 0",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "config" / "parking_config.json"),
        help="Duong dan file JSON cau hinh slot",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(BASE_DIR / "models" / "model.p"),
        help="Model pickle tuong thich voi preprocessing 15x15 BGR",
    )
    args = parser.parse_args()
    source = parse_source(args.source)

    print("[INFO] Dang khoi tao he thong...")
    print(f"- Nguon du lieu: {source}")
    print(f"- Cau hinh: {args.config}")
    print(f"- Model: {args.model}")

    try:
        detector = ParkingDetector(model_path=args.model, boxes_path=args.config)
    except (FileNotFoundError, TypeError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] Khong the khoi tao ParkingDetector: {exc}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Khong the mo nguon video/webcam: {source}")
        cap.release()
        return

    smoother = SlotSmoother(window_size=7, min_stable_count=5)
    counter = ParkingCounter()
    window_name = "Parking Monitor System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    show_score = True
    frame_idx = 0
    fps = 0.0
    previous_time = time.perf_counter()
    output_frame = None

    print("[INFO] q/ESC: thoat | p: pause | s: luu anh | r: reset | d: confidence | i: info")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Da doc xong video hoac mat ket noi nguon.")
                    break

                frame_idx += 1
                now = time.perf_counter()
                elapsed = now - previous_time
                if elapsed > 0:
                    fps = 1.0 / elapsed
                previous_time = now

                try:
                    analysis = detector.classify_frame(frame)
                    smoothed_results = []
                    for item in analysis["results"]:
                        state = smoother.update(
                            item["index"],
                            item.get("state", "invalid"),
                            confidence=item.get("confidence"),
                            min_confidence=0.5,
                        )
                        item["state"] = state
                        item["label"] = state
                        item["is_empty"] = state == "empty"
                        smoothed_results.append(item)

                    counts = counter.count_slots(smoothed_results, detector.boxes)
                    known_count = counts["empty"] + counts["occupied"]
                    availability = (
                        counts["empty"] / known_count * 100 if known_count else 0.0
                    )
                    output_frame = draw_results(
                        frame,
                        smoothed_results,
                        counts,
                        availability,
                        fps,
                        show_score,
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    print(f"[WARNING] Loi xu ly tai frame {frame_idx}: {exc}")
                    output_frame = frame.copy()

            if output_frame is None:
                continue
            cv2.imshow(window_name, output_frame)
            key = cv2.waitKey(25) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print(f"[INFO] Tam dung: {paused}")
            elif key == ord("s"):
                reports_dir = BASE_DIR / "reports"
                reports_dir.mkdir(exist_ok=True)
                image_path = reports_dir / f"captured_frame_{frame_idx}.jpg"
                if cv2.imwrite(str(image_path), output_frame):
                    print(f"[SUCCESS] Da luu anh: {image_path}")
                else:
                    print(f"[ERROR] Khong the luu anh: {image_path}")
            elif key == ord("r"):
                smoother.reset()
                print("[INFO] Da reset bo loc temporal smoothing.")
            elif key == ord("d"):
                show_score = not show_score
                print(f"[INFO] Hien thi confidence: {show_score}")
            elif key == ord("i"):
                print(
                    f"[SYSTEM INFO] Frame: {frame_idx} | "
                    f"Tong so slot: {len(detector.boxes)}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Da giai phong nguon video va dong cua so.")


if __name__ == "__main__":
    main()
