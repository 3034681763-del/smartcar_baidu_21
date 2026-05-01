#!/usr/bin/env python3
import argparse
import multiprocessing
import threading
import time

import cv2
import numpy as np

from Cam_cap import vidpub_course
from Process_manage import ProcessManager
from SerialCommunicate import serial_server_process
from infer_server_client import InferClient1, model_configs, serve_model_process
from model_path_config import get_lane_model_path, get_model_profile
from move_base import Base_func, Task_func
from shared_memory_manager import SharedMemoryManager
from test_shutdown import ShutdownController
from tool_func import select_detection_box


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visual-servo tracking test using the same runtime flow as main.py."
    )
    parser.add_argument("--lane-device", default="1-2.3:1.0", help="Lane camera physical path")
    parser.add_argument(
        "--task-device",
        "--device",
        dest="task_device",
        default="1-2.1:1.0",
        help="Task camera physical path",
    )
    parser.add_argument("--lane-model", default=get_lane_model_path(), help="Lane model path")
    parser.add_argument(
        "--physical-path",
        default="2.2:1.0",
        help="Serial physical path, for example 2.2:1.0 or /dev/ttyUSB0",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--target-x", type=float, default=320.0, help="Tracking target x")
    parser.add_argument("--target-y", type=float, default=240.0, help="Tracking target y")
    parser.add_argument(
        "--target-label",
        default=None,
        help="Model label to track, for example text, shelf, blue, yellow. Leave empty to use all labels.",
    )
    parser.add_argument(
        "--selector",
        default="biggest",
        choices=["first", "biggest", "biggest-complete", "leftmost", "leftmost-complete", "closest"],
        help="Detection box selection strategy.",
    )
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum detection confidence")
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=5.0,
        help="Margin used by complete-box selectors to reject boxes touching image edges.",
    )
    parser.add_argument("--cam-pose", default="L", choices=["L", "R", "shoot"], help="Tracking pose")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Tracking timeout in seconds. Use 0 to disable timeout and wait until aligned.",
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=0,
        help="Tracking stops only after this many consecutive missing frames. Use 0 to disable.",
    )
    return parser


def same_detection(left, right):
    if left is None or right is None:
        return False
    return (
        getattr(left, "label", None) == getattr(right, "label", None)
        and tuple(round(float(v), 2) for v in getattr(left, "bbox", ())) ==
        tuple(round(float(v), 2) for v in getattr(right, "bbox", ()))
    )


def draw_detection(frame, det, selected=False):
    x1, y1, x2, y2 = [int(round(float(value))) for value in det.bbox]
    cx, cy = [int(round(float(value))) for value in det.center]
    color = (0, 0, 255) if selected else (0, 255, 0)
    thickness = 3 if selected else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
    text = f"{getattr(det, 'label', None)} {float(getattr(det, 'score', 0.0)):.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def start_main_like_runtime(args):
    shm_manager = SharedMemoryManager()
    shm_manager.create_block("shm_0", size=640 * 640 * 3)
    shm_manager.create_block("shm_task", size=640 * 640 * 3)
    shm_manager.create_block("shm_lane", size=8)
    shm_manager.create_block("shm_crop", size=640 * 640 * 3)

    request_queue = multiprocessing.Queue()
    publish_queue = multiprocessing.Queue()
    action_done_event = multiprocessing.Event()

    mgr = ProcessManager(setup_signal_handlers=False)
    mgr.add_process(
        target=serial_server_process,
        args=(args.physical_path, args.baudrate, request_queue, publish_queue, action_done_event),
    )
    mgr.add_process(
        target=vidpub_course,
        args=(
            "shm_0",
            "shm_task",
            "shm_lane",
            args.lane_model,
            args.lane_device,
            args.task_device,
        ),
    )
    for cfg in model_configs:
        mgr.add_process(target=serve_model_process, args=(cfg, shm_manager))

    mgr.start_all()
    return shm_manager, mgr, request_queue, publish_queue, action_done_event


def main():
    multiprocessing.freeze_support()
    shutdown = ShutdownController("tracking_test").install()
    args = build_parser().parse_args()
    task_cfg = next(cfg for cfg in model_configs if cfg["name"] == "task")
    shm_manager, mgr, request_queue, publish_queue, action_done_event = start_main_like_runtime(args)
    shutdown.add_callback(action_done_event.set)
    task_client = None
    tracking_result = {"done": False, "ok": False}

    target_pose = (args.target_x, args.target_y)
    target_labels = [args.target_label] if args.target_label else None
    timeout_s = args.timeout if args.timeout and args.timeout > 0 else None
    max_missed_frames = args.max_missed_frames if args.max_missed_frames and args.max_missed_frames > 0 else None

    print("=" * 60)
    print("Tracking Test - Main Flow")
    print(f"Model profile: {get_model_profile()}")
    print(f"Lane model:    {args.lane_model}")
    print(f"Lane camera:   {args.lane_device}")
    print(f"Task camera:   {args.task_device}")
    print(f"Task params:   {task_cfg.get('params')}")
    print(f"Serial path:   {args.physical_path}")
    print(f"Target pose:   {target_pose}")
    print(f"Target label:  {args.target_label or 'any model label'}")
    print(f"Selector:      {args.selector}")
    print(f"Min score:     {args.min_score:.2f}")
    print(f"Cam pose:      {args.cam_pose}")
    print(f"Timeout:       {'disabled' if timeout_s is None else f'{timeout_s:.2f}s'}")
    print(f"Missed max:    {'disabled' if max_missed_frames is None else max_missed_frames}")
    print("Quit:          press q")
    print("=" * 60)

    try:
        task_client = InferClient1("task", shm_manager, task_cfg["port"])
        base = Base_func(
            request_queue=request_queue,
            use_lane_shm=False,
            action_done_event=action_done_event,
        )
        task = Task_func(
            base,
            ocr_reader=None,
            task_shm_key="shm_task",
            task_client=task_client,
        )

        def run_tracking():
            ok = task.tracking_executor(
                target_pose=target_pose,
                cam_pose=args.cam_pose,
                timeout_s=timeout_s,
                max_missed_frames=max_missed_frames,
                selector_strategy=args.selector,
                target_labels=target_labels,
                min_score=args.min_score,
                edge_margin=args.edge_margin,
            )
            tracking_result["ok"] = bool(ok)
            tracking_result["done"] = True

        tracking_thread = threading.Thread(target=run_tracking, daemon=True)
        tracking_thread.start()

        cv2.namedWindow("tracking_test", cv2.WINDOW_NORMAL)
        frame_count = 0
        fps_start = time.time()
        missed_frames = 0
        last_label_log_time = 0.0

        while not shutdown.is_set():
            frame = shm_manager.read_image("shm_task", (640, 640, 3), np.uint8)
            detections = task_client("shm_task", (640, 640, 3), np.uint8)
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            selected_box = select_detection_box(
                detections,
                strategy=args.selector,
                target_labels=target_labels,
                min_score=args.min_score,
                target_pose=target_pose,
                edge_margin=args.edge_margin,
                frame_width=640,
                frame_height=640,
            )

            if selected_box is None:
                missed_frames += 1
            else:
                missed_frames = 0

            now = time.time()
            if detections and now - last_label_log_time > 0.5:
                labels = [
                    f"{getattr(det, 'label', None)}:{float(getattr(det, 'score', 0.0)):.2f}"
                    for det in detections
                ]
                print(f"[tracking_test] model labels -> {labels}")
                last_label_log_time = now

            for det in detections or []:
                draw_detection(frame, det, selected=same_detection(det, selected_box))

            cv2.drawMarker(
                frame,
                (int(args.target_x), int(args.target_y)),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
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
            cv2.putText(
                frame,
                f"Missed: {missed_frames}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )
            if tracking_result["done"]:
                result_text = "OK" if tracking_result["ok"] else "FAILED"
                cv2.putText(
                    frame,
                    f"Tracking: {result_text}",
                    (12, 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if tracking_result["ok"] else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("tracking_test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[tracking_test] Exit requested by user.")
                shutdown.request()
                break
            if tracking_result["done"]:
                print(f"[tracking_test] result={tracking_result['ok']}")
                break
            shutdown.wait(0.02)
    except KeyboardInterrupt:
        shutdown.request()
    finally:
        action_done_event.set()
        if task_client is not None:
            task_client.close()
        mgr.terminate_all()
        shm_manager.release_all()
        cv2.destroyAllWindows()

    if shutdown.is_set() and not tracking_result["done"]:
        return 0
    return 0 if tracking_result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
