#!/usr/bin/env python3
import argparse
import time

import cv2


def parse_device(device_arg):
    if isinstance(device_arg, str) and device_arg.isdigit():
        return int(device_arg)
    return device_arg


def open_camera(device_arg, width, height, fps, codec):
    device = parse_device(device_arg)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device_arg}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone camera preview test. Supports one or two cameras."
    )
    parser.add_argument("--lane-device", default="0", help="Lane camera device index or path")
    parser.add_argument(
        "--task-device",
        default="",
        help="Optional task camera device index or path",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height")
    parser.add_argument("--fps", type=int, default=60, help="Requested capture FPS")
    parser.add_argument("--codec", default="MJPG", help="FOURCC codec, e.g. MJPG")
    return parser


def main():
    args = build_parser().parse_args()
    if len(args.codec) != 4:
        raise ValueError("FOURCC codec must be exactly 4 characters")

    lane_cap = open_camera(args.lane_device, args.width, args.height, args.fps, args.codec)
    task_cap = None
    if args.task_device:
        task_cap = open_camera(args.task_device, args.width, args.height, args.fps, args.codec)

    print("=" * 60)
    print("Camera Test")
    print(f"Lane device: {args.lane_device}")
    if args.task_device:
        print(f"Task device: {args.task_device}")
    print(f"Resolution:  {args.width}x{args.height}")
    print(f"FPS request: {args.fps}")
    print("Quit:        press q in any preview window")
    print("=" * 60)

    cv2.namedWindow("Lane Camera Test", cv2.WINDOW_NORMAL)
    if task_cap is not None:
        cv2.namedWindow("Task Camera Test", cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    try:
        while True:
            lane_ok, lane_frame = lane_cap.read()
            if not lane_ok:
                print("[camera_test] Lane camera frame read failed.")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            fps_text = frame_count / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                lane_frame,
                f"Lane FPS: {fps_text:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Lane Camera Test", lane_frame)

            if task_cap is not None:
                task_ok, task_frame = task_cap.read()
                if task_ok:
                    cv2.putText(
                        task_frame,
                        "Task Camera",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Task Camera Test", task_frame)
                else:
                    print("[camera_test] Task camera frame read failed.")
                    break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[camera_test] Exit requested by user.")
                break
    finally:
        lane_cap.release()
        if task_cap is not None:
            task_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
