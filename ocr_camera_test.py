#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Open camera preview, press q to capture one frame and run OCR."
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Camera device index or path, e.g. 0 or /dev/video0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Requested camera FPS",
    )
    parser.add_argument(
        "--codec",
        default="MJPG",
        help="FOURCC codec, e.g. MJPG or YUYV",
    )
    parser.add_argument(
        "--task-model",
        default="Global_V2",
        help="Task detection model name or path for YoloeInfer",
    )
    parser.add_argument(
        "--ocr-model",
        default="fenge_mod",
        help="OCR model name or path for OCRReco",
    )
    parser.add_argument(
        "--label",
        default="text",
        help="Target detection label to crop before OCR",
    )
    parser.add_argument(
        "--paddle-jetson-path",
        default="/home/jetson/workspace/code_vehicle_wbt/paddle_jetson",
        help="Optional paddle_jetson directory or its parent directory",
    )
    return parser


def parse_device(device_arg):
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def draw_prompt(frame):
    lines = [
        "Press q: capture and OCR",
        "Press ESC: exit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    x = 12
    y = 24

    cv2.rectangle(frame, (5, 5), (320, 72), (0, 0, 0), -1)
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
        y += 28


def select_text_detections(detections, target_label):
    if not detections:
        return []

    filtered = [
        det for det in detections
        if getattr(det, "label", None) == target_label
    ]
    filtered.sort(key=lambda det: (float(det.center[1]), float(det.center[0])))
    return filtered


def crop_from_detection(frame, det, pad=12):
    x1, y1, x2, y2 = map(int, det.bbox)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)
    return frame[y1:y2, x1:x2]


def load_paddle_jetson(optional_path=""):
    if optional_path:
        optional_path = os.path.expanduser(optional_path)
        normalized = os.path.abspath(optional_path)
        module_dir = normalized
        if os.path.basename(normalized) == "paddle_jetson":
            module_dir = os.path.dirname(normalized)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

    try:
        from paddle_jetson import OCRReco, YoloeInfer
        return OCRReco, YoloeInfer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import paddle_jetson. "
            "Please activate the correct Python environment or pass "
            "--paddle-jetson-path /path/to/paddle_jetson or its parent directory"
        ) from exc


def patch_namespace_defaults():
    defaults = {
        "enable_mkldnn": False,
        "enable_mkldnn_bfloat16": False,
        "cpu_threads": 1,
        "use_gpu": True,
        "gpu_id": 0,
        "gpu_mem": 1024,
        "ir_optim": True,
        "use_tensorrt": False,
        "precision": "fp32",
        "min_subgraph_size": 3,
        "benchmark": False,
        "use_fp16": False,
        "device": "GPU",
        "run_mode": "paddle",
    }

    for key, value in defaults.items():
        if not hasattr(argparse.Namespace, key):
            setattr(argparse.Namespace, key, value)


def run_ocr_on_frame(frame, task_model, ocr_model, target_label, paddle_jetson_path=""):
    OCRReco, YoloeInfer = load_paddle_jetson(paddle_jetson_path)
    patch_namespace_defaults()

    detector = YoloeInfer(task_model)
    ocr = OCRReco(ocr_model)

    detections = detector.predict(frame)
    selected = select_text_detections(detections, target_label)

    if not selected:
        return [], detections

    texts = []
    for index, det in enumerate(selected, start=1):
        crop = crop_from_detection(frame, det)
        if crop.size == 0:
            continue
        text = ocr.predict(crop)
        texts.append(
            {
                "index": index,
                "label": det.label,
                "bbox": tuple(map(int, det.bbox)),
                "center": tuple(map(int, det.center)),
                "text": str(text).strip(),
            }
        )

    close_detector = getattr(detector, "close", None)
    if callable(close_detector):
        try:
            close_detector()
        except Exception:
            pass

    close_ocr = getattr(ocr, "close", None)
    if callable(close_ocr):
        try:
            close_ocr()
        except Exception:
            pass

    return texts, detections


def print_results(results):
    print("=" * 60)
    if not results:
        print("No OCR result found.")
        print("=" * 60)
        return

    print("OCR Results")
    for item in results:
        print(
            f"[{item['index']}] label={item['label']} bbox={item['bbox']} "
            f"center={item['center']} text={item['text']}"
        )
    print("=" * 60)


def main():
    args = build_parser().parse_args()
    device = parse_device(args.device)

    if len(args.codec) != 4:
        raise ValueError("FOURCC codec must be exactly 4 characters, e.g. MJPG")

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.codec))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("=" * 60)
    print("OCR Camera Test")
    print(f"Device:      {args.device}")
    print(f"Resolution:  {args.width}x{args.height}")
    print(f"FPS request: {args.fps}")
    print(f"Codec:       {args.codec}")
    print(f"Task model:  {args.task_model}")
    print(f"OCR model:   {args.ocr_model}")
    if args.paddle_jetson_path:
        print(f"Module dir:  {args.paddle_jetson_path}")
    print("Press q in the preview window to capture and OCR.")
    print("=" * 60)

    cv2.namedWindow("ocr_camera_test", cv2.WINDOW_NORMAL)

    captured_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            preview = frame.copy()
            draw_prompt(preview)
            cv2.imshow("ocr_camera_test", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                captured_frame = frame.copy()
                break
            if key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_frame is None:
        print("No frame captured.")
        return

    print("Captured one frame, running OCR...")
    start_time = time.time()
    results, detections = run_ocr_on_frame(
        captured_frame,
        task_model=args.task_model,
        ocr_model=args.ocr_model,
        target_label=args.label,
        paddle_jetson_path=args.paddle_jetson_path,
    )
    elapsed = time.time() - start_time

    print_results(results)
    print(f"OCR pipeline time: {elapsed:.2f}s")

    if detections:
        debug = captured_frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = getattr(det, "label", "unknown")
            cv2.putText(
                debug,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("ocr_capture_debug", debug)
        print("Close the debug window or press any key in it to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
