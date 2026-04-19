#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2


def parse_device(device_arg):
    if isinstance(device_arg, str) and device_arg.isdigit():
        return int(device_arg)
    return device_arg


def ensure_paddle_jetson_importable(optional_path=""):
    if optional_path:
        normalized = os.path.abspath(os.path.expanduser(optional_path))
        search_path = (
            os.path.dirname(normalized)
            if os.path.basename(normalized) == "paddle_jetson"
            else normalized
        )
        if search_path not in sys.path:
            sys.path.insert(0, search_path)

    from paddle_jetson import YoloeInfer

    return YoloeInfer


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone task detection preview test."
    )
    parser.add_argument("--device", default="0", help="Camera device index or path")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=int, default=60, help="Capture FPS")
    parser.add_argument("--codec", default="MJPG", help="FOURCC codec")
    parser.add_argument("--model", default="Global_V2", help="Task model name or path")
    parser.add_argument(
        "--paddle-jetson-path",
        default="",
        help="Optional paddle_jetson directory or its parent directory",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if len(args.codec) != 4:
        raise ValueError("FOURCC codec must be exactly 4 characters")

    YoloeInfer = ensure_paddle_jetson_importable(args.paddle_jetson_path)
    detector = YoloeInfer(args.model)

    device = parse_device(args.device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.codec))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("=" * 60)
    print("Infer Test")
    print(f"Device: {args.device}")
    print(f"Model:  {args.model}")
    print("Quit:   press q")
    print("=" * 60)

    cv2.namedWindow("infer_test", cv2.WINDOW_NORMAL)
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[infer_test] Camera frame read failed.")
                break

            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            detections = detector.predict(frame)
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            if detections:
                for det in detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx, cy = map(int, det.center)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
                    text = f"{det.label} {det.score:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("infer_test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[infer_test] Exit requested by user.")
                break
    finally:
        close_fn = getattr(detector, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
