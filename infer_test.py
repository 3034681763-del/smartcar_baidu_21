#!/usr/bin/env python3
import argparse
import multiprocessing
import time

import cv2
import numpy as np

from Cam_cap import vidpub_course
from Process_manage import ProcessManager
from infer_server_client import InferClient1, model_configs, serve_model_process
from model_path_config import get_lane_model_path, get_model_profile
from shared_memory_manager import SharedMemoryManager
from test_shutdown import ShutdownController


def build_parser():
    parser = argparse.ArgumentParser(
        description="Task detection preview using the same camera, shared-memory, and model-server flow as main.py."
    )
    parser.add_argument(
        "--lane-device",
        default="1-2.3:1.0",
        help="Lane camera physical path, matching main.py by default.",
    )
    parser.add_argument(
        "--task-device",
        "--device",
        dest="task_device",
        default="1-2.1:1.0",
        help="Task camera physical path, matching main.py by default.",
    )
    parser.add_argument("--lane-model", default=get_lane_model_path(), help="Lane model path")
    parser.add_argument(
        "--task-only-model",
        action="store_true",
        help="Start only the task detection model service instead of all auxiliary model services.",
    )
    return parser


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

    mgr = ProcessManager(setup_signal_handlers=False)
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

    configs = [
        cfg for cfg in model_configs
        if not args.task_only_model or cfg.get("name") == "task"
    ]
    for cfg in configs:
        mgr.add_process(target=serve_model_process, args=(cfg, shm_manager))

    mgr.start_all()
    return shm_manager, mgr


def main():
    multiprocessing.freeze_support()
    shutdown = ShutdownController("infer_test").install()
    args = build_parser().parse_args()
    task_cfg = next(cfg for cfg in model_configs if cfg["name"] == "task")
    shm_manager, mgr = start_main_like_runtime(args)
    task_client = None

    print("=" * 60)
    print("Infer Test - Main Flow")
    print(f"Model profile: {get_model_profile()}")
    print(f"Lane model:    {args.lane_model}")
    print(f"Lane camera:   {args.lane_device}")
    print(f"Task camera:   {args.task_device}")
    print(f"Task params:   {task_cfg.get('params')}")
    print("Quit:          press q")
    print("=" * 60)

    try:
        task_client = InferClient1("task", shm_manager, task_cfg["port"])
        cv2.namedWindow("infer_test", cv2.WINDOW_NORMAL)
        frame_count = 0
        fps_start = time.time()
        last_label_log_time = 0.0

        while not shutdown.is_set():
            frame = shm_manager.read_image("shm_task", (640, 640, 3), np.uint8)
            detections = task_client("shm_task", (640, 640, 3), np.uint8)
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            now = time.time()
            if detections and now - last_label_log_time > 0.5:
                labels = [
                    f"{getattr(det, 'label', None)}:{float(getattr(det, 'score', 0.0)):.2f}"
                    for det in detections
                ]
                print(f"[infer_test] model labels -> {labels}")
                last_label_log_time = now

            for det in detections or []:
                draw_detection(frame, det)

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
                shutdown.request()
                break
            shutdown.wait(0.02)
    except KeyboardInterrupt:
        shutdown.request()
    finally:
        if task_client is not None:
            task_client.close()
        mgr.terminate_all()
        shm_manager.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
