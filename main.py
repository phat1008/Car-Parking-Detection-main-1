import argparse
from pathlib import Path

import cv2

from parking_detector import ParkingDetector


def main():
    parser = argparse.ArgumentParser(
        description="Parking detection using custom pickle model"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--model",
        default="models/model.p",
        help="Pickle model path",
    )

    parser.add_argument(
        "--boxes",
        default="config/parking_config.json",
        help="Parking boxes JSON path",
    )

    parser.add_argument(
        "--output",
        default="reports/parking_result.mp4",
        help="Output video path",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    detector = ParkingDetector(
        model_path=args.model,
        boxes_path=args.boxes,
    )

    capture = cv2.VideoCapture(args.video)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    codec = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        args.output,
        codec,
        fps,
        (width, height),
    )

    frame_index = 0
    last_output = None

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_index += 1

            if frame_index % max(args.frame_skip, 1) == 0:
                analysis = detector.classify_frame(frame)
                last_output = detector.draw_results(frame, analysis)

            if last_output is None:
                last_output = frame

            cv2.imshow("Parking Detection", last_output)
            writer.write(last_output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()