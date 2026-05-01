#!/usr/bin/env python3
import argparse
import time

import cv2

from camera_device_utils import open_camera_from_device_arg
from test_shutdown import ShutdownController


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone camera preview test. Supports one or two cameras."
    )
    parser.add_argument(
        "--lane-device",
        default="1-2.3:1.0",
        help="Lane camera device path, /dev/videoX, or physical path like 1-2.3:1.0",
    )
    parser.add_argument(
        "--task-device",
        default="1-2.1:1.0",
        help="Optional task camera device path, /dev/videoX, or physical path like 1-2.1:1.0",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height")
    parser.add_argument("--fps", type=int, default=60, help="Requested capture FPS")
    parser.add_argument("--codec", default="MJPG", help="FOURCC codec, e.g. MJPG")
    return parser


def main():
    shutdown = ShutdownController("camera_test").install()
    args = build_parser().parse_args()
    if len(args.codec) != 4:
        raise ValueError("FOURCC codec must be exactly 4 characters")

    lane_cap, lane_device, lane_candidates = open_camera_from_device_arg(
        args.lane_device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        codec=args.codec,
        role_name="Lane",
    )
    task_cap = None
    task_device = None
    task_candidates = []
    if args.task_device:
        task_cap, task_device, task_candidates = open_camera_from_device_arg(
            args.task_device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            codec=args.codec,
            role_name="Task",
        )

    print("=" * 60)
    print("Camera Test")
    print(f"Lane path:   {args.lane_device}")
    print(f"Lane nodes:  {lane_candidates}")
    print(f"Lane using:  {lane_device}")
    if args.task_device:
        print(f"Task path:   {args.task_device}")
        print(f"Task nodes:  {task_candidates}")
        print(f"Task using:  {task_device}")
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
        while not shutdown.is_set():
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
                shutdown.request()
                break
    except KeyboardInterrupt:
        shutdown.request()
    finally:
        lane_cap.release()
        if task_cap is not None:
            task_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
